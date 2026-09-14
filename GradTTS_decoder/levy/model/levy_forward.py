import math
import torch


def _cosine_alpha_bar(t: torch.Tensor, s: float = 0.008) -> torch.Tensor:
    """
    Cosine schedule alpha_bar(t) in [0,1], t in [0,1].
    Shape: t [B] -> alpha_bar [B]
    """
    # alpha_bar(t) = cos^2(((t+s)/(1+s)) * pi/2) / cos^2((s/(1+s)) * pi/2)
    c0 = math.cos((s / (1.0 + s)) * math.pi / 2.0) ** 2
    ct = torch.cos(((t + s) / (1.0 + s)) * math.pi / 2.0) ** 2
    return ct / c0


def sample_symmetric_alpha_stable(shape, alpha: float, device=None, dtype=None) -> torch.Tensor:
    """
    Chambers–Mallows–Stuck sampler for symmetric alpha-stable SαS with scale=1, skew=0.
    Returns i.i.d. samples with the given shape.

    Note: This is *componentwise* SαS. Many LIM-style implementations use i.i.d. noise per dimension.
    """
    if not (0.0 < alpha <= 2.0):
        raise ValueError(f"alpha must be in (0,2], got {alpha}")

    device = device or "cpu"
    dtype = dtype or torch.float32

    U = (torch.rand(shape, device=device, dtype=dtype) - 0.5) * math.pi  # Uniform(-pi/2, pi/2)
    W = torch.empty(shape, device=device, dtype=dtype).exponential_(1.0)  # Exp(1)

    if abs(alpha - 1.0) > 1e-6:
        # SαS: sin(alpha U) / (cos U)^(1/alpha) * (cos((1-alpha)U)/W)^((1-alpha)/alpha)
        numer = torch.sin(alpha * U)
        denom = torch.cos(U).clamp_min(1e-12).pow(1.0 / alpha)
        frac = numer / denom

        inner = torch.cos((1.0 - alpha) * U).clamp_min(1e-12) / W.clamp_min(1e-12)
        power = inner.pow((1.0 - alpha) / alpha)
        return frac * power
    else:
        # alpha == 1 (Cauchy) special case
        # X = (2/pi) * [ (pi/2 + U) * tan(U) - log( ( (pi/2) * W * cos(U) ) / (pi/2 + U) ) ]
        half_pi = math.pi / 2.0
        term1 = (half_pi + U) * torch.tan(U)
        term2 = torch.log((half_pi * W * torch.cos(U).clamp_min(1e-12)) / (half_pi + U).clamp_min(1e-12))
        return (2.0 / math.pi) * (term1 - term2)


@torch.no_grad()
def levy_forward_mel(
    X0: torch.Tensor,      # [B, 80, L]
    mask: torch.Tensor,    # [B, 1,  L]  (0/1 or bool)
    t: torch.Tensor,       # [B] in [0,1]
    alpha: float = 1.5,
    s: float = 0.008,
) -> torch.Tensor:
    """
    Forward Lévy (alpha-stable) corruption:
        x_t = a(t) x_0 + gamma(t) eps,  eps ~ SαS(1)
    with a(t) = alpha_bar(t)^(1/alpha), gamma(t) = (1 - alpha_bar(t))^(1/alpha),
    then apply mask: x_t := x_t * mask + x_0 * (1-mask).
    """
    if X0.ndim != 3 or X0.shape[1] != 80:
        raise ValueError(f"Expected X0 shape [B,80,L], got {tuple(X0.shape)}")
    if mask.shape[0] != X0.shape[0] or mask.shape[2] != X0.shape[2]:
        raise ValueError(f"mask must be [B,1,L] matching X0, got {tuple(mask.shape)}")
    if t.ndim != 1 or t.shape[0] != X0.shape[0]:
        raise ValueError(f"t must be [B], got {tuple(t.shape)}")

    B, C, L = X0.shape
    device, dtype = X0.device, X0.dtype

    t = t.clamp(0.0, 1.0).to(device=device, dtype=dtype)

    alpha_bar = _cosine_alpha_bar(t, s=s).clamp(0.0, 1.0)          # [B]
    a = alpha_bar.pow(1.0 / alpha)                                 # [B]
    gamma = (1.0 - alpha_bar).clamp_min(0.0).pow(1.0 / alpha)      # [B]

    a = a.view(B, 1, 1)
    gamma = gamma.view(B, 1, 1)

    eps = sample_symmetric_alpha_stable((B, C, L), alpha=alpha, device=device, dtype=dtype)
    Xt = a * X0 + gamma * eps

    # apply length mask (broadcast along mel bins)
    m = mask.to(device=device)
    if m.dtype != Xt.dtype:
        m = m.to(dtype=Xt.dtype)
    Xt = Xt * m + X0 * (1.0 - m)
    return Xt

