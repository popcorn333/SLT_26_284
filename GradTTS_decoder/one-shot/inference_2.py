import json
import re
import torch
import shutil
import numpy as np
from model import GradTTS
from torch.utils.data import DataLoader
from data import TextMelDataset, TextMelBatchCollate
from text.symbols import symbols
from utils import intersperse
from model.utils import fix_len_compatibility
from scipy.io.wavfile import write
from scipy import integrate
from datetime import datetime
import matplotlib.pyplot as plot
import sys, os

sys.path.append('./hifi-gan/')
from env import AttrDict
from models import Generator as HiFiGAN

from omegaconf import DictConfig, OmegaConf
import hydra

def _get_basename(path: str) -> str:
    return os.path.splitext(os.path.split(path)[-1])[0]

def save_mel_spectrograms_to_file(mel_spectrograms, output_dir,basenames): # input a list of mel tensors, output one mel tensor to one file in output_dir
    for i, mel_spec in enumerate(mel_spectrograms):

        torch.save(mel_spec , f'{output_dir}/{basenames[i]}.pt')

def pt_to_pdf(pt, pdf, vmin=-12.5, vmax=0.0):
    spec = pt
    fig = plot.figure(figsize=(20, 4), tight_layout=True)
    subfig = fig.add_subplot()
    image = subfig.imshow(spec, cmap="jet", origin="lower", aspect="equal", interpolation="none", vmax=vmax,
                          vmin=vmin)
    fig.colorbar(mappable=image, orientation='vertical', ax=subfig, shrink=0.5)
    plot.savefig(pdf, format="pdf")
    plot.close()

def get_integer_part(s):
    return int(''.join(filter(str.isdigit, s)))


#####################################################################################################


@hydra.main(version_base=None, config_path='./config')
def main(cfg: DictConfig):
    
    cwd = os.getcwd()
    print(f'the current working directory is,{cwd}')

    sample_rate = cfg.data.sample_rate
    hop_length = cfg.data.hop_length
    win_length = cfg.data.win_length
    n_fft = cfg.data.n_fft
    add_blank = cfg.data.add_blank
    n_feats = cfg.data.n_feats

    gt_dir = cfg.eval.gt_dir
    cvt_dir = cfg.eval.cvt_dir
    checkpoint_dir = cfg.eval.checkpoint_dir
    epoch_interval = cfg.eval.epoch_interval
    evaluation_mode = cfg.eval.evaluation_mode
    split = cfg.eval.split
    HIFIGAN_CONFIG = cfg.eval.code_dir + 'checkpts/hifigan-config.json'
    HIFIGAN_CHECKPT = cfg.eval.code_dir + 'checkpts/hifigan.pt'

    device = torch.device(f'cuda:{cfg.training.gpu}')

    # import cmudict
    print('Logging validation/test dataset...')
    valid_dataset = TextMelDataset(split, cfg)
    
    print('get checkpts paths')
    checkpoint_files = [os.path.join(checkpoint_dir, file) for file in os.listdir(checkpoint_dir) if
                        file.endswith('.pt')]
    checkpoint_files = sorted(checkpoint_files, key=get_integer_part)
    ending_epoch = len(checkpoint_files)

    print('checkpoint_files', checkpoint_dir)
    print('Initialize GradTTS MODEL')
    generator = GradTTS(cfg).to(device)

    print('build Valid batch...')
    # idx = np.random.choice(list(range(len(test_dataset))), size=params.test_size, replace=False)

    valid_batch_text = []
    valid_batch_mel = []
    filepaths = []
    basenames = []
    i=0
    for filepath_and_text in valid_dataset.filepaths_and_text:
        # Each entry of filepaths_and_text is a [filepath, text_content] list.
        filepath, text = filepath_and_text[0], filepath_and_text[1]
        mel = valid_dataset.get_mel(filepath)
        basename = os.path.basename(filepath).split('.')[0]
        filepaths.append(filepath)
        basenames.append(basename)
        valid_batch_text.append(text)  # [{'y': mel, 'x': text}, {'y': mel, 'x': text}, {'y': mel, 'x': text}]
        valid_batch_mel.append(mel)
    if evaluation_mode == 'WAVPDFMEL':

        print('output original mel spectrogram and text')
        print('all mel spectrogram is written in a file')
        print('all text is written in a file')
        if not os.path.exists(f'{split}/{gt_dir}'):
            os.makedirs(f'{split}/{gt_dir}', exist_ok=True)
        if not os.path.exists(f'{split}/{cvt_dir}'):
            os.makedirs(f'{split}/{cvt_dir}', exist_ok=True)

        gt_text = f'{split}/{gt_dir}'+'/text.txt'
        with open(gt_text, 'w') as text_file:
            for i, item in enumerate(valid_batch_text):
                text_file.write(f"{valid_batch_text[i]}\n")

        texts = valid_batch_text
        ''' uncommented for faster inference
        print('move the.wav file from LJSpeech dataset to evaluation/test directory')
        for i, filepath in enumerate(filepaths):
            shutil.copy(filepath, f'{split}/{gt_dir}/{basenames[i]}.wav')

        save_mel_spectrograms_to_file(valid_batch_mel, f'{split}/{gt_dir}', basenames)
        
        for i, mel_spec in enumerate(valid_batch_mel):
            pt_to_pdf(mel_spec, f'{split}/{gt_dir}/mel_{basenames[i]}.pdf' , vmin=-12.5, vmax=0.0)
        '''
        print('Initializing HiFi-GAN as vocoder')
        
        with open(HIFIGAN_CONFIG) as f:
            h = AttrDict(json.load(f))
        vocoder = HiFiGAN(h)
        vocoder.load_state_dict(torch.load(HIFIGAN_CHECKPT, map_location=lambda loc, storage: loc, weights_only=False)['generator'])
        _ = vocoder.to(device).eval()
        vocoder.remove_weight_norm()

        for i in range(ending_epoch - 1, ending_epoch - 21, -1):
            y_mel = []
            #get integer parts of ith checkpoint file
            checkpoint_name = _get_basename(checkpoint_files[i])
            number_part = get_integer_part(checkpoint_name)
            index = int(number_part)
    
    #load the ith checkpoint file
            generator.load_state_dict(torch.load(f'{checkpoint_files[i]}', map_location=lambda loc, storage: loc, weights_only=False))
            _ = generator.cuda().eval()
            print('using checkpoint files', f'{checkpoint_files[i]}')
            print(f'Doing the {index}st epoch')
            if not os.path.exists(f'{split}/{cvt_dir}/Epoch_{index}'):
                os.makedirs(f'{split}/{cvt_dir}/Epoch_{index}')
            valid_dataset.sample_test_batch(len(valid_batch_text))
            now = datetime.now()
            print(f'begin {index}st epoch', now.strftime("%Y-%m-%d %H:%M:%S")) 
            with torch.no_grad():
                for i, item in enumerate(valid_dataset):
                    # convert word to phonemes:
                    
                    x = item['x'].to(torch.long).unsqueeze(0).to(device)
                    x_lengths = torch.LongTensor([x.shape[-1]]).to(device)
                    y_enc, y_dec, attn = generator.forward(x, x_lengths, n_timesteps=cfg.eval.n_timesteps, temperature=1.5,
                                                            stoc=False, length_scale=0.91)
                        
                    if torch.isnan(y_dec).any():
                        print("Warning: The tensor contains NaN")
                    if torch.isinf(y_dec).any():
                        print("Warning: The tensor contains infinite values.")
                        
                    audio = (vocoder.forward(y_dec).cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16)
                    write(f'{split}/{cvt_dir}/Epoch_{index}/{basenames[i]}.wav', 22050, audio)
                    pt_to_pdf(y_dec.cpu().squeeze(0), f'{split}/{cvt_dir}/Epoch_{index}/dec_{basenames[i]}.pdf')
                    pt_to_pdf(y_enc.cpu().squeeze(0), f'{split}/{cvt_dir}/Epoch_{index}/enc_{basenames[i]}.pdf')

            now = datetime.now()
            print(f'end {index}st epoch', now.strftime("%Y-%m-%d %H:%M:%S"))

if __name__ == '__main__':
    main()
