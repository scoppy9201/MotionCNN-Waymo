# MotionCNN Waymo: project implementation

This repository is a project-specific implementation and extension of the
MotionCNN motion-prediction codebase. The current work focuses on getting the
pipeline running with Python 3.10, processing Waymo Motion TFRecords, training a
ResNet-34 model, evaluating trajectory errors, and exploring predictions in an
interactive dashboard.

This repository does not claim authorship of the original MotionCNN research,
paper, source code, or Waymo challenge results. Upstream attribution and license
terms are recorded in [LICENSE](LICENSE).

## Current implementation

- **Input:** Waymo Motion Dataset v1.0.0, `uncompressed/tf_example` format.
- **Preprocessing:** creates a 512 × 512 raster with 27 channels and agent-level
  NPZ files containing 11 history steps and up to 80 future steps.
- **Model:** ResNet-34 predicts 3 trajectory modes, each with 80 points and
  Gaussian uncertainty components. The default config initializes from random
  weights (`pretrained: false`).
- **Training:** PyTorch; automatically selects CUDA when available, otherwise
  runs on CPU. The checked-in config is currently set to one epoch for a local
  smoke run. Increase `training.num_epochs` in `configs/basic.yaml` for a longer
  experiment.
- **Evaluation and viewer:** ADE/FDE diagnostics and a Streamlit dashboard for
  inspecting scenes, histories, ground truth, predicted modes, and confidence.

The local one-shard development data used during setup produced 2,640 training
agent samples and 1,427 validation agent samples. These are only local sample
counts, not full-dataset or benchmark results.

## Environment

Use Python 3.10 on Windows. TensorFlow 2.18 is used for CPU TFRecord
preprocessing; PyTorch handles model training and can use CUDA when available.
From PowerShell in the repository directory:

```powershell
py -3.10 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

In VS Code, run **Python: Select Interpreter** and choose
`.venv\Scripts\python.exe` so Pylance uses the environment where these packages
were installed.

Check the environment:

```powershell
python -c "import torch, tensorflow as tf, timm, cv2, streamlit, plotly, reportlab; print('PyTorch:', torch.__version__); print('TensorFlow:', tf.__version__); print('Streamlit:', streamlit.__version__); print('Plotly:', plotly.__version__); print('CUDA:', torch.cuda.is_available())"
```

## Dataset setup

Accept Waymo's dataset terms, then download the Motion Dataset v1.0.0 TFRecord
files. The following Google Cloud CLI commands download one shard per split:

```cmd
gcloud storage cp gs://waymo_open_dataset_motion_v_1_0_0/uncompressed/tf_example/training/training_tfexample.tfrecord-00000-of-01000 data\raw\training\
gcloud storage cp gs://waymo_open_dataset_motion_v_1_0_0/uncompressed/tf_example/validation/validation_tfexample.tfrecord-00000-of-00150 data\raw\validation\
```

Place files under `data/raw/training` and `data/raw/validation`. Dataset files
are not included in this repository. Waymo dataset access is governed by the
[Waymo Open Dataset terms](https://waymo.com/open/terms/).

## Preprocess

Run from PowerShell with the project environment active:

```powershell
python prerender.py --data-path data/raw/training --output-path data/processed/training --config configs/basic.yaml --n-jobs 1
python prerender.py --data-path data/raw/validation --output-path data/processed/validation --config configs/basic.yaml --n-jobs 1
```

Prerendering can take time. Start with a small number of shards and one worker
to limit memory use.

## Train

```powershell
python train.py --train-data-path data/processed/training --val-data-path data/processed/validation --checkpoints-path checkpoints --config configs/basic.yaml --device cpu
```

Use `--device cuda` on a CUDA-enabled PyTorch installation. CPU training is
slower. Checkpoints are written to `checkpoints/basic` when training completes
and at configured evaluation intervals.

## Evaluate a checkpoint

```powershell
$checkpoint = Get-ChildItem checkpoints\basic\*.pth | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName
python evaluate.py --data-path data/processed/validation --checkpoint "$checkpoint" --config configs/basic.yaml --output-dir evaluation --device cpu
```

This writes `evaluation/metrics.json` and `evaluation/trajectory.svg`. Metrics
include minADE, minFDE, and the ADE/FDE for the highest-confidence mode. They
are diagnostics over the local validation samples and are not the official
Waymo challenge evaluation.

## Interactive dashboard

Start the local viewer after a checkpoint exists:

```powershell
python -m streamlit run app.py
```

To let people on the same local network view it, run:

```powershell
python -m streamlit run app.py --server.address 0.0.0.0
```

Share the **Network URL** printed by Streamlit with people on that network.
Public internet access requires deployment to a reachable host and a separate
review of dataset licensing and hosting limits.

## Project presentation

The updated project overview is in
[img/waymo_motion_prediction_2021_3rd_place_solution_presentation.pdf](img/waymo_motion_prediction_2021_3rd_place_solution_presentation.pdf).
The prior upstream presentation is retained as
[img/upstream_motioncnn_2021_presentation.pdf](img/upstream_motioncnn_2021_presentation.pdf)
for source-history and attribution.

## Source and research acknowledgment

This repository contains adapted work from the upstream
[MotionCNN Waymo repository](https://github.com/stepankonev/MotionCNN-Waymo-Open-Motion-Dataset).
The related paper is:

> Stepan Konev, Kirill Brodt, and Artsiom Sanakoyeu, “MotionCNN: A Strong
> Baseline for Motion Prediction in Autonomous Driving,” 2022.
> [arXiv:2206.02163](https://arxiv.org/abs/2206.02163).

The current preprocessing compatibility changes, CPU training path, evaluation
script, and interactive dashboard are project-specific additions. The
upstream copyright notice and required acknowledgment remain in [LICENSE](LICENSE).
