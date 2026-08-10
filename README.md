# SGGP

This repository provides inference-only code for reproducing the 4-shot medical anomaly detection results of SGGP with three frameworks: MVFA, MadCLIP, and IQE-CLIP.

Because the SGGP paper has not yet been accepted, only the testing code is released at this stage. All training code will be released after the paper is accepted.

## Repository structure

```text
.
├── frameworks/
│   ├── mvfa/                    # MVFA inference implementation
│   ├── madclip/                 # MadCLIP inference implementation
│   └── iqeclip/                 # IQE-CLIP inference implementation
├── dataset/
│   ├── medical_few.py           # shared medical dataset loader
│   └── fewshot_seed/            # shared fixed few-shot support lists
├── checkpoints/                 # downloaded separately; not included in the code release
├── test.sh                      # evaluate one framework/method/dataset checkpoint
├── reproduce_table1.sh          # reproduce all six rows in Table 1
├── requirements.txt
└── README.md
```

The three frameworks share the dataset loader and fixed support lists under `dataset/`.

## Environment

Python 3.10 and a CUDA-capable NVIDIA GPU are recommended. The release was validated with PyTorch 2.0.1+cu118 and torchvision 0.15.2+cu118.

Install the PyTorch and torchvision builds that match your CUDA version first. Then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

MVFA uses RAPIDS cuML for pixel-level AUROC when it is available. If cuML is not installed, the MVFA evaluator automatically falls back to scikit-learn's `roc_auc_score`, so cuML is optional. For GPU-accelerated segmentation evaluation, install the cuML build that matches your CUDA version by following the [RAPIDS installation guide](https://docs.rapids.ai/install/).

The validated MVFA/MadCLIP environment uses `transformers==4.25.1`, while the validated IQE-CLIP environment uses `transformers==4.44.2` (the version listed in `requirements.txt`). Separate environments are therefore recommended for strict reproduction.

By default, all scripts use `python`. Set the corresponding interpreter variables when using separate environments:

```bash
export MVFA_PYTHON=/path/to/mvfa/python
export MADCLIP_PYTHON=/path/to/madclip/python
export IQECLIP_PYTHON=/path/to/iqeclip/python
```

## Datasets

Download and preprocess the six medical datasets by following the dataset instructions in [MVFA-AD](https://github.com/MediaBrain-SJTU/MVFA-AD). Dataset images are not included in this repository.

Arrange the processed data as follows:

```text
<DATA_ROOT>/
├── Brain_AD/{valid,test}/...
├── Liver_AD/{valid,test}/...
├── Retina_RESC_AD/{valid,test}/...
├── Retina_OCT2017_AD/{valid,test}/...
├── Chest_AD/{valid,test}/...
└── Histopathology_AD/{valid,test}/...
```

The fixed 4-shot support lists required for evaluation are already included under `dataset/fewshot_seed/`.

## Checkpoints

Download the checkpoint archive from [Baidu Netdisk](https://pan.baidu.com/s/1eiHsk9DSTgc0LoC7kHPtnQ?pwd=hewy) (password: `hewy`) and extract it into the repository root:

```bash
unzip checkpoints.zip -d .
```

After extraction, the checkpoint layout should be:

```text
checkpoints/
├── pretrained/ViT-L-14-336px.pt
├── mvfa/{baseline,sggp}/<dataset>.pth
├── madclip/{baseline,sggp}/<dataset>.pth
└── iqeclip/{baseline,sggp}/<dataset>.pth
```

Each framework contains six `baseline` checkpoints and six `sggp` checkpoints, one for each dataset.

## Evaluation

Evaluate one framework, method, and dataset with `test.sh`:

```bash
bash test.sh \
  --framework madclip \
  --method sggp \
  --dataset Brain \
  --data-root /path/to/data \
  --gpu 0
```

Supported frameworks are `mvfa`, `madclip`, and `iqeclip`; supported methods are `baseline` and `sggp`; supported datasets are `Retina_OCT2017`, `Histopathology`, `Chest`, `Brain`, `Liver`, and `Retina_RESC`.

To verify that a checkpoint can be constructed and loaded without running dataset inference, add `--check-only`:

```bash
bash test.sh \
  --framework iqeclip \
  --method baseline \
  --dataset Brain \
  --data-root /path/to/data \
  --gpu 0 \
  --check-only
```

Reproduce all six rows in Table 1 with two GPUs:

```bash
bash reproduce_table1.sh \
  --data-root /path/to/data \
  --gpus 0,1 \
  --output results/table1
```

For checkpoint-load verification only, append `--check-only`. For a single GPU, use `--gpus 0`.

A successful complete run ends with `TABLE1_REPRODUCTION_PASS`. Per-task JSON files and logs are stored under the selected output directory.

## Table 1 results

All values are percentages. `Mean AUC` is the arithmetic mean over the six displayed image-level AUC values. `Mean pAUC` is the arithmetic mean over the displayed pixel-level AUC values for Brain, Liver, and Retina_RESC; Retina_OCT2017, Histopathology, and Chest are detection-only datasets without pixel-level ground-truth masks.

| Method | OCT AUC | HIS AUC | Chest AUC | Brain AUC | Brain pAUC | Liver AUC | Liver pAUC | RESC AUC | RESC pAUC | Mean AUC | Mean pAUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MVFA | 99.57 | 83.58 | 82.46 | 90.94 | 96.89 | 86.94 | 99.57 | 95.86 | 99.16 | 89.89 | 98.54 |
| MVFA+SGGP | 99.67 | 83.84 | 83.27 | 91.74 | 97.75 | 87.13 | 99.71 | 96.29 | 99.03 | 90.32 | 98.83 |
| MadCLIP | 99.63 | 82.43 | 82.70 | 92.64 | 97.31 | 83.42 | 99.55 | 94.28 | 98.56 | 89.18 | 98.47 |
| MadCLIP+SGGP | 99.84 | 83.04 | 82.56 | 92.85 | 97.33 | 83.46 | 99.53 | 94.29 | 98.69 | 89.34 | 98.52 |
| IQE-CLIP | 98.83 | 74.01 | 79.72 | 81.27 | 97.70 | 62.74 | 99.47 | 93.71 | 98.72 | 81.71 | 98.63 |
| IQE-CLIP+SGGP | 98.91 | 77.52 | 77.76 | 85.19 | 97.77 | 63.60 | 99.53 | 94.30 | 98.75 | 82.88 | 98.68 |

## Acknowledgements

**This implementation uses code from [OpenCLIP](https://github.com/mlfoundations/open_clip), [MVFA-AD](https://github.com/MediaBrain-SJTU/MVFA-AD), and [IQE-CLIP](https://github.com/hongh0/IQE-CLIP). Please follow the licenses and citation requirements of the corresponding projects and datasets.**
