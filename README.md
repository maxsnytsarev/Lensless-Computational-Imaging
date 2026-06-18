# Lensless Computational Imaging


## About

This repository contains an implementation of ADMM-based methods for lensless image recovery presented in this [paper](https://arxiv.org/abs/2502.01102)


#### Implemented methods:
- ADMM-100 with fixed parameters
- Unrolled ADMM-20
- 8M LeADMM-5 with a pre-processor (8M stands for eight million  parameters)
- 8M LeADMM-5 with a post-processor
- 8M LeADMM-5 with both pre-processor and post-processor

All models are trained and evaluated on [DigiCam](https://huggingface.co/datasets/bezzam/DigiCam-Mirflickr-MultiMask-10K) dataset

## Installation

### Clone the repository 

```bash
git clone https://github.com/maxsnytsarev/Lensless-Computational-Imaging.git
```
```bash
cd Lensless-Computational-Imaging
```

### (If needed) Create virtual environment (e.g. venv)
```bash
python3 -m venv lensless_computation
source lensless_computation/bin/activate  
```

### Install required packages
```bash
pip install -r requirements.txt
```

## Writer
If you want to log any values to comet_ml do not forget to provide your api key

```bash
export COMET_API_KEY="..."
```

If you don't want to use writer please put `writer.run_name=null` where needed.

## Data preparation

Model expects data either from [DigiCam](https://huggingface.co/datasets/bezzam/DigiCam-Mirflickr-MultiMask-10K) or in the custom dataset format described below

## Custom Dataset
If you provide your own dataset, please make sure that it is in the following format 
```
CustomDataset
├── lensless
│   ├── ImageID1.png
│   ├── ImageID2.png
│   .
│   .
│   .
│   └── ImageIDn.png
├── masks
│   ├── ImageID1.npy
│   ├── ImageID2.npy
│   .
│   .
│   .
│   └── ImageIDn.npy
└── lensed # ground truth original image, may not exist
    ├── ImageID1.png
    ├── ImageID2.png
    .
    .
    .
    └── ImageIDn.png
```

## Training

#### Available training configurations:
- Unrolled ADMM-20 `unrolled_admm_20`
- 8M LeADMM-5 with a pre-processor `le_admm_5_pre_inv`
- 8M LeADMM-5 with a post-processor `le_admm_5_post_inv`
- 8M LeADMM-5 with both pre-processor and post-processor `le_admm_5_pre_post_inv`

To train a model run choose `model` and `writer.run_name` according to the model you want to train
```bash
python train.py \
  model=le_admm_5_pre_post_inv \
  writer.run_name=leadmm_pre_post \
  trainer.save_dir=saved
```
*Note: for training you must use logger (writer)*

Checkpoints are saved according to the config to `saved` folder. You can also change it from bash if you like

You can also easily manage your own hyperparameters from bash:
```bash
python train.py trainer.n_epochs=20
```
It is useful to use HF token - it will improve dataset loading
```bash
export COMET_API_KEY="..."
```

## Inference

#### All models are supported. Model names for inference:
- ADMM-100 `ADMM-100`
- Unrolled ADMM-20 `Unrolled_ADMM-20`
- 8M LeADMM-5 with a pre-processor `LeADMM-5-pre-inv`
- 8M LeADMM-5 with a post-processor `LeADMM-5-post-inv`
- 8M LeADMM-5 with both pre-processor and post-processor `LeADMM-5-pre-post-inv`


#### Inference on HF Dataset:
```bash
python inference.py \
  model_name=LeADMM-5-pre-post-inv \
  writer.run_name=inference \
  inferencer.save_path=inference_save
```

#### Inference on custom dataset
```bash
python inference.py \
  model_name=LeADMM-5-post-inv \
  writer.run_name=null \
  datasets=inference \
  datasets.inference.path=/path/to/dataset \
  inferencer.save_path=inference_save
```
`inference_save` is a default path to saved reconstructions, change if needed

Model checkpoints are downloaded automatically from the [Hugging Face repository](https://huggingface.co/maxsnytsarev/lensless_camera)

If you want to evaluate average reconstruction time put `inferencer.eval_time=true`. For example:
```bash
python inference.py \
  model_name=ADMM-100 \
  writer.run_name=null \
  datasets=inference \
  datasets.inference.path=/path/to/dataset \
  inferencer.save_path=inference_save \
  inferencer.eval_time=true 
```
## Compute metrics

Script that computes metrics given path to ground truth images and their reconstructed versions

Supported dataset format:
```
Dataset
├── ID0
│   ├── lensed_roi_ID0.png
│   ├── reconstructed_roi_ID0.png
├── ID1
│   ├── lensed_roi_ID1.png
│   ├── reconstructed_roi_ID1.png
└── 
│   .
│   .
│   .
├── IDn
│   ├── lensed_roi_IDn.png
│   ├── reconstructed_roi_IDn.png
└── 
```

The metrics script calculates MSE, LPIPS, PSNR and SSIM.

```bash
python get_metrics.py \
  metrics_info.metrics_path=inference_save \
  writer.run_name=metrics
```

If you want to run inference and then compute metrics than `metrics_info.metrics_path` must point to the output directory created by
`inference.py`. Ground-truth images must be available to calculate metrics.

## Demo

You can try the model yourself

Download [`DEMO.ipynb`](DEMO.ipynb) - open it in an empty Colab and follow the instructions. 
The notebook downloads datasets, runs reconstruction, displays the
results and calculates metrics.

## Project structure

```
.
├── train.py
├── inference.py
├── get_metrics.py
├── demo.py
├── DEMO.ipynb
├── lensless_helpers/
└── src/
    ├── configs/
    ├── datasets/
    ├── loss/
    ├── metrics/
    ├── model/
    ├── trainer/
    └── utils/
```

## Link to the report
[Report](https://www.comet.com/maxsnytsarev/lensless-computation/reports/890Ha8AVNSsKWSxcDyNazbceN)

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
