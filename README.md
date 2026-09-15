# SLT_26

This repository contains two decoder implementations for text-to-speech experiments. Each implementation has its own model code, experiment configurations, training scripts, and inference scripts.

## Decoder folders

| Folder | Decoder | 
| --- | --- | --- |
| [Grad_TTS_decoder](Grad_TTS_decoder/) | 
| [SGMSEpls_decoder](SGMSEpls_decoder/) | 

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

| Decoder folder | 
| --- | --- |
| `Grad_TTS_decoder` | 
| `SGMSEpls_decoder` | 

The commands below assume Conda is available and initialized in your shell. These environments are already configured on the original system; creating new environments is not required there.

### Grad-TTS decoder

Install the env specified in Grad_TTS_decoder/module_list.yaml

Then, from the repository root, enter the selected variant, for example:

```bash
cd Grad_TTS_decoder/add
```

Use this environment for all variants under `Grad_TTS_decoder`. Ensure the selected launch script activates this environment or uses its Python executable.

### SGMSE-based decoder

Install a separate env from Grad-TTS decoder specified in Grad_TTS_decoder/module_list.yaml
Then in SLT_26_284 repo, run pip install -e ./sgmse
Then, from the repository root, enter the selected variant, for example:

```bash
cd SGMSEpls_decoder/add
```

Use this environment for all variants under `SGMSEpls_decoder`. Ensure the selected launch script activates this environment or uses its Python executable. This decoder uses the bundled SGMSE implementation, including the custom `sgmse.backbones.ncsnpp_v2_wote` backbone.


## Running experiments

1. Activate the environment for the chosen decoder.
2. Enter its experiment variant directory, such as `Grad_TTS_decoder/add/` or `SGMSEpls_decoder/add/`.
3. Configure dataset paths, checkpoint paths, and output locations in the variant's configuration files and scripts.
4. Review `run.sh`, `inference.sh`, and the available metric scripts before launching the experiment. Adapt cluster-specific modules, Slurm settings, and absolute paths to your system.

Training and inference entry points are provided as `train.py` and `inference.py` within the variants. Their launch scripts provide the intended Hydra configuration arguments.

## Data and checkpoints

[Download model checkpoints from Google Drive](https://drive.google.com/drive/folders/1wwiHbAnYU7OO7Wj7gasy_ZGncf8rfpF3?usp=sharing).

# SLT_26_284
