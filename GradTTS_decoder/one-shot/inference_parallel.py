import json, torch, numpy as np, sys, os
import matplotlib.pyplot as plot
from datetime import datetime
from tqdm import tqdm
from scipy.io.wavfile import write
from torch.utils.data import DataLoader
from omegaconf import DictConfig
import hydra

from model import GradTTS
from data import TextMelDataset, TextMelBatchCollate

sys.path.append('./hifi-gan/')
from env import AttrDict
from models import Generator as HiFiGAN


def _get_basename(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def get_integer_part(s):
    return int(''.join(filter(str.isdigit, s)))


def pt_to_pdf(pt, pdf, vmin=-12.5, vmax=0.0):
    fig = plot.figure(figsize=(20, 4), tight_layout=True)
    ax = fig.add_subplot()
    image = ax.imshow(pt, cmap="jet", origin="lower", aspect="equal",
                      interpolation="none", vmax=vmax, vmin=vmin)
    fig.colorbar(mappable=image, orientation='vertical', ax=ax, shrink=0.5)
    plot.savefig(pdf, format="pdf")
    plot.close()


def process_batch(generator, vocoder, batch, start_idx, device, cfg,
                  basenames, cwd, split, cvt_dir, index):
    x = batch['x'].to(torch.long).to(device)

    if 'x_lengths' in batch:
        x_lengths = batch['x_lengths'].to(torch.long).to(device)
    else:
        x_lengths = (x != 0).sum(dim=1).to(torch.long).to(device)

    y_enc, y_dec, attn = generator.forward(
        x, x_lengths, n_timesteps=cfg.eval.n_timesteps,
        temperature=1.5, stoc=False, length_scale=0.91
    )

    audio_batch = vocoder.forward(y_dec).cpu().clamp(-1, 1).numpy()

    for j in range(x.shape[0]):
        global_idx = start_idx + j
        if global_idx >= len(basenames):
            continue

        basename = basenames[global_idx]
        audio = (np.squeeze(audio_batch[j]) * 32768).astype(np.int16)

        wav_path = f'{cwd}/{split}/{cvt_dir}/Epoch_{index}/{basename}.wav'
        print(f'file is written to, {wav_path}')
        write(wav_path, cfg.data.sample_rate, audio)

        pt_to_pdf(y_dec[j].cpu(), f'{split}/{cvt_dir}/Epoch_{index}/dec_{basename}.pdf')
        pt_to_pdf(y_enc[j].cpu(), f'{split}/{cvt_dir}/Epoch_{index}/enc_{basename}.pdf')


@hydra.main(version_base=None, config_path='./config')
def main(cfg: DictConfig):
    cwd = os.getcwd()
    print(f'the current working directory is,{cwd}')

    gt_dir, cvt_dir = cfg.eval.gt_dir, cfg.eval.cvt_dir
    starting_epoch, ending_epoch = cfg.eval.starting_epoch, cfg.eval.ending_epoch
    checkpoint_dir, epoch_interval = cfg.eval.checkpoint_dir, cfg.eval.epoch_interval
    split = cfg.eval.split
    device = torch.device(f'cuda:{cfg.training.gpu}')
    print(f'Using device: {device}')

    hifigan_config = cfg.eval.code_dir + 'checkpts/hifigan-config.json'
    hifigan_checkpt = cfg.eval.code_dir + 'checkpts/hifigan.pt'

    print('Logging validation/test dataset...')
    valid_dataset = TextMelDataset(split, cfg)
    loader = DataLoader(
        dataset=valid_dataset, batch_size=cfg.eval.batch_size,
        collate_fn=TextMelBatchCollate(), drop_last=False,
        num_workers=cfg.training.num_workers, shuffle=False
    )

    checkpoint_files = sorted(
        [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith('.pt')],
        key=get_integer_part
    )
    ending_epoch = ending_epoch if ending_epoch is not None else len(checkpoint_files)
    print('checkpoint_dir', checkpoint_dir)

    os.makedirs(f'{split}/{gt_dir}', exist_ok=True)
    os.makedirs(f'{split}/{cvt_dir}', exist_ok=True)

    print('build Valid batch...')
    basenames, valid_batch_text = [], []

    for filepath, text in valid_dataset.filepaths_and_text:
        basenames.append(os.path.basename(filepath).split('.')[0])
        valid_batch_text.append(text)

    with open(f'{split}/{gt_dir}/text.txt', 'w') as text_file:
        for text in valid_batch_text:
            text_file.write(f"{text}\n")

    print('Initializing HiFi-GAN as vocoder')
    with open(hifigan_config) as f:
        h = AttrDict(json.load(f))

    vocoder = HiFiGAN(h)
    vocoder_ckpt = torch.load(hifigan_checkpt, map_location=device, weights_only=False)
    vocoder.load_state_dict(vocoder_ckpt['generator'])
    vocoder = vocoder.to(device).eval()
    vocoder.remove_weight_norm()

    generator = GradTTS(cfg).to(device)

    for epk_idx in range(starting_epoch, ending_epoch, epoch_interval):
        checkpoint_name = _get_basename(checkpoint_files[epk_idx])
        index = int(get_integer_part(checkpoint_name))

        checkpoint = torch.load(checkpoint_files[epk_idx], map_location=device, weights_only=False)
        generator.load_state_dict(checkpoint)
        generator = generator.to(device).eval()

        os.makedirs(f'{split}/{cvt_dir}/Epoch_{index}', exist_ok=True)

        now = datetime.now()
        print(f'begin {index}st epoch', now.strftime("%Y-%m-%d %H:%M:%S"))

        with torch.no_grad():
            leftover_batch, leftover_start_idx = None, None

            with tqdm(loader, total=len(loader)) as progress_bar:
                for batch_idx, batch in enumerate(progress_bar):
                    B = batch['x'].shape[0]
                    start_idx = batch_idx * cfg.eval.batch_size

                    if B < cfg.eval.batch_size:
                        leftover_batch, leftover_start_idx = batch, start_idx
                        continue

                    process_batch(generator, vocoder, batch, start_idx, device, cfg,
                                  basenames, cwd, split, cvt_dir, index)

            if leftover_batch is not None:
                print("Processing leftover batch after main loop...")
                process_batch(generator, vocoder, leftover_batch, leftover_start_idx,
                              device, cfg, basenames, cwd, split, cvt_dir, index)

        now = datetime.now()
        print(f'end {index}st epoch', now.strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == '__main__':
    main()
