# SLT_26

This repository contains two decoder implementations for text-to-speech experiments. Each implementation has its own model code, experiment configurations, training scripts, and inference scripts.

## Decoder folders

| Folder | Decoder | Origin |
| --- | --- | --- |
| [Grad_TTS_decoder](Grad_TTS_decoder/) | Grad-TTS decoder using the GradLogPEstimator2d network |
| [SGMSEpls_decoder](SGMSEpls_decoder/) | SGMSE-based decoder using an lighter weight NCSN++ backbone|

Both folders contain the experiment variants `add`, `blur`, `fm`, `levy`, `mask`, `product`, and `score`. All work is isolated inside the selected variant directory.

```text
SLT_26/
├── README.md
├── Grad_TTS_decoder/
│   ├── add/
│   ├── blur/
│   ├── fm/
│   ├── levy/
│   ├── mask/
│   ├── product/
│   └── score/
└── SGMSEpls_decoder/
    ├── add/
    ├── blur/
    ├── fm/
    ├── levy/
    ├── mask/
    ├── product/
    └── score/
```

## Separate environments

**Use the existing, separate Conda environment for each decoder.** Activate the corresponding environment before training, inference, evaluation, or building native extensions.

| Decoder folder | Conda environment path |
| --- | --- |
| `Grad_TTS_decoder` | `/users/acp23xt/.conda/envs/grad-tts-masking` |
| `SGMSEpls_decoder` | `/mnt/parscratch/users/acp23xt/private/conda/envs/CUDA124_sgmsetts` |

The commands below assume Conda is available and initialized in your shell. These environments are already configured on the original system; creating new environments is not required there.

### Grad-TTS decoder

Activate the Grad-TTS environment:

```bash
conda activate /users/acp23xt/.conda/envs/grad-tts-masking
```

Then, from the repository root, enter the selected variant, for example:

```bash
cd Grad_TTS_decoder/add
```

Use this environment for all variants under `Grad_TTS_decoder`. Ensure the selected launch script activates this environment or uses its Python executable.

### SGMSE-based decoder

Activate the SGMSE environment:

```bash
conda activate /mnt/parscratch/users/acp23xt/private/conda/envs/CUDA124_sgmsetts
```

Then, from the repository root, enter the selected variant, for example:

```bash
cd SGMSEpls_decoder/add
```

Use this environment for all variants under `SGMSEpls_decoder`. Ensure the selected launch script activates this environment or uses its Python executable. This decoder uses the bundled SGMSE implementation, including the custom `sgmse.backbones.ncsnpp_v2_wote` backbone.

The checked-in [add/environment.yml](SGMSEpls_decoder/add/environment.yml) is a historical dependency reference: its `CUDA124_GradTTS` name and `prefix` do not identify the SGMSE runtime environment listed above. For setup on another machine, export the corresponding working environment and adapt machine-specific paths and CUDA dependencies. The bundled [SGMSE README](SGMSEpls_decoder/add/sgmse/README.md) provides additional package information.

### Native extensions

Rebuild any required native extensions inside the corresponding decoder environment. For example, from a variant directory containing `model/monotonic_align/`:

```bash
(cd model/monotonic_align && python setup.py build_ext --inplace)
```

This build requires NumPy, Cython, and a compatible compiler. Existing compiled files may need rebuilding for your Python version and platform.

## Running experiments

1. Activate the environment for the chosen decoder.
2. Enter its experiment variant directory, such as `Grad_TTS_decoder/add/` or `SGMSEpls_decoder/add/`.
3. Configure dataset paths, checkpoint paths, and output locations in the variant's configuration files and scripts.
4. Review `run.sh`, `inference.sh`, and the available metric scripts before launching the experiment. Adapt cluster-specific modules, Slurm settings, and absolute paths to your system.

Training and inference entry points are provided as `train.py` and `inference.py` within the variants. Their launch scripts provide the intended Hydra configuration arguments.

## Data and checkpoints

[Download model checkpoints from Google Drive](https://drive.google.com/drive/folders/1wwiHbAnYU7OO7Wj7gasy_ZGncf8rfpF3?usp=sharing).

The prepared folders exclude saved checkpoints and inference-output directories, along with `.out`/`.err` job logs, Git metadata, and Symphony logs. Supply the datasets and any required pretrained model or HiFi-GAN weights separately, and update their configured paths before running inference or resuming training.

Environment installation and experiment execution have not been validated as part of this repository preparation.
# SLT_26_284
