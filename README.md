# Explainable AI for Autonomous Driving Weather Robustness

Research-grade pipeline for evaluating how autonomous-driving perception models behave under adverse weather, with a particular focus on explainability, robustness, and explanation stability. The project compares a detection model, Faster R-CNN, against an interpretable classification model, ProtoPNet, on BDD100K-derived control and stress splits.

The core question is:

> Do object-recognition models keep making reliable, explainable decisions when clear-weather driving scenes are replaced by lower-visibility or adverse-weather scenes?

## Highlights

- Builds control and stress weather subsets from BDD100K-style annotations.
- Trains and evaluates Faster R-CNN for object detection.
- Trains and evaluates ProtoPNet for prototype-based classification.
- Uses Grad-CAM, prototype activations, and Captum Integrated Gradients for interpretability.
- Measures weather robustness with control-vs-stress performance deltas.
- Runs explainability stress tests including drop tests, causal re-evaluation, weather ablation, and explanation drift.
- Includes result visualizations and supporting research notes under `Results/` and `Papers/`.

## Repository Layout

```text
.
+-- Notebooks/
|   +-- 1_dataprep_newds1.ipynb
|   +-- 1_newds1_fast_r_cnn_training.ipynb
|   +-- 2_dataprep_newds2.ipynb
|   +-- 2_newds2_fast_r_cnn_training.ipynb
|   +-- evaluate_fastrcnn.ipynb
|   +-- evaluate_protopnet.ipynb
|   +-- final-pipeline-explainable-ai.ipynb
+-- Papers/
|   +-- Autonomous Driving.pdf
|   +-- BDD100k.pdf
|   +-- Captum.pdf
|   +-- Causal Re-evaluation (via Captum).pdf
|   +-- Counterfactual Weather Ablation (MPRNet De-raining).pdf
|   +-- Grad-CAM.pdf
|   +-- High-Resolution Auditing (Integrated Gradients).pdf
|   +-- Quantitative Fidelity Testing Drop-Test.pdf
+-- Results/
|   +-- Metrics/
|   +-- Visualisations/
+-- report.pdf
```

## Methodology

### Dataset Strategy

The notebooks derive smaller research datasets from BDD100K-style image and JSON annotation pairs. Two dataset definitions are explored:

- `newds1`: control is clear weather with good visibility; stress is everything else.
- `newds2`: control is clear weather with good visibility; stress is explicitly non-clear weather and non-good visibility.

The final pipeline expects the following dataset layout:

```text
/kaggle/working/data/
+-- control/
|   +-- images/
|   |   +-- train/
|   |   +-- val/
|   |   +-- test/
|   +-- labels/
|       +-- train/
|       +-- val/
|       +-- test/
+-- stress/
    +-- images/
    |   +-- test/
    +-- labels/
        +-- test/
```

The final notebook reports these split sizes:

| Split | Images |
| --- | ---: |
| Control train | 1,973 |
| Control validation | 561 |
| Control test | 448 |
| Stress test | 1,552 |

The object classes used by the final pipeline are:

- `background`
- `car`
- `traffic light`
- `traffic sign`

### Models

#### Faster R-CNN

The detection baseline uses `torchvision.models.detection.fasterrcnn_resnet50_fpn` with a custom prediction head for the project classes. It is evaluated using detection metrics such as mAP, AP per class, precision, and recall.

#### ProtoPNet

The interpretable model uses a ResNet-50 feature extractor with learnable prototype vectors. The implementation trains in warmup, joint, prototype-push, and last-layer stages, then evaluates accuracy and class-level precision/recall/F1.

### Explainability Methods

The final pipeline includes:

- Grad-CAM over Faster R-CNN backbone activations.
- ProtoPNet prototype activation maps.
- Captum Integrated Gradients for both model families.
- Top-saliency masking drop tests.
- Weather ablation through MPRNet-style deraining.
- Causal re-evaluation by measuring attention shift under perturbation.
- Explanation drift using SSIM and cosine similarity.

## Key Results

### Faster R-CNN Detection Performance

| Condition | mAP@0.50:0.95 | mAP@0.50 | mAP@0.75 | AP car | AP traffic light | AP traffic sign |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Control | 0.2632 | 0.5475 | 0.2289 | 0.3842 | 0.1645 | 0.2409 |
| Stress | 0.2492 | 0.5301 | 0.2084 | 0.3807 | 0.1456 | 0.2212 |

### ProtoPNet Classification Performance

| Condition | Accuracy | Weighted Precision | Weighted F1 |
| --- | ---: | ---: | ---: |
| Control | 0.5022 | 0.5035 | 0.4248 |
| Stress | 0.4304 | 0.4466 | 0.3661 |

The observed ProtoPNet accuracy drop from control to stress is approximately `0.072`.

### Weather Ablation With MPRNet Deraining

| Model | Metric | Original Stress | MPRNet Derained Stress | Change |
| --- | --- | ---: | ---: | ---: |
| Faster R-CNN | mAP@0.50:0.95 | 0.2492 | 0.2466 | -0.0026 |
| ProtoPNet | Accuracy | 0.4304 | 0.4130 | -0.0174 |

In this run, deraining did not improve downstream model performance. This is useful evidence: image restoration does not automatically translate into better perception-model robustness.

### Explanation Drift

| Model | Mean SSIM | Mean Cosine Similarity | Drift Score |
| --- | ---: | ---: | ---: |
| Faster R-CNN | 0.8372 | 0.0536 | 0.9464 |
| ProtoPNet | 0.5429 | 0.5007 | 0.4993 |

The drift measurements suggest that performance alone is not enough to characterize robustness. Explanation behavior can change substantially even when metric deltas look modest.

## Getting Started

### Recommended Environment

The notebooks were developed for a Kaggle-style GPU environment and have been run with:

- Python 3.12
- NVIDIA Tesla T4 GPU
- PyTorch and Torchvision
- BDD100K-style image/JSON data under `/kaggle/working/data`

### Install Dependencies

From a notebook cell:

```python
!pip install -q torch torchvision torchmetrics pycocotools grad-cam captum scikit-image seaborn scikit-learn gdown
```

For the MPRNet weather-ablation section, the final pipeline also clones the official MPRNet repository and downloads the deraining checkpoint:

```python
!git clone -q https://github.com/swz30/MPRNet.git /kaggle/working/MPRNet
```

The notebook expects the MPRNet deraining checkpoint at:

```text
/kaggle/working/model_deraining.pth
```

### Dataset

The raw BDD100K dataset is not committed to this repository. The final notebook includes an optional `gdown` helper for downloading a prepared dataset archive when available:

```python
import gdown

file_id = "1x3l0D95_FtNiUOrN4ErTLkVxmukaYhrh"
gdown.download(f"https://drive.google.com/uc?id={file_id}", "new_dataset.zip", quiet=False)
```

After extraction, verify that the directory structure matches the expected `/kaggle/working/data` layout above.

### Checkpoints

The final notebook looks for these local checkpoint files:

```text
model_a_fasterrcnn.pth
model_b_protopnet.pth
```

If they exist, the notebook loads them. If they are absent, the notebook trains the corresponding model and saves a checkpoint.

## How to Reproduce

1. Open a GPU-backed notebook environment.
2. Install the Python dependencies.
3. Prepare or download the BDD100K-derived dataset under `/kaggle/working/data`.
4. Run `Notebooks/final-pipeline-explainable-ai.ipynb` from top to bottom.
5. Review generated metrics, Grad-CAM outputs, prototype activations, integrated-gradient plots, and weather-ablation tables.
6. Compare generated figures with the archived outputs under `Results/`.

For a more stepwise workflow, use the notebooks in this order:

```text
Notebooks/1_dataprep_newds1.ipynb
Notebooks/1_newds1_fast_r_cnn_training.ipynb
Notebooks/2_dataprep_newds2.ipynb
Notebooks/2_newds2_fast_r_cnn_training.ipynb
Notebooks/evaluate_fastrcnn.ipynb
Notebooks/evaluate_protopnet.ipynb
Notebooks/final-pipeline-explainable-ai.ipynb
```

## Results Artifacts

The repository includes exported visualizations such as:

- Faster R-CNN and ProtoPNet accuracy comparisons.
- Drop-test curves.
- Weather-ablation results.
- Causal re-evaluation charts.
- Explanation-drift metrics.
- Integrated Gradients examples.
- ProtoPNet confusion matrix.

See:

```text
Results/Visualisations/
Results/Metrics/Metrics/
```

## Engineering Notes

- The final pipeline is notebook-first rather than package-first. This is appropriate for research exploration, but productionization would benefit from moving dataset loaders, model builders, metrics, and explainability utilities into importable Python modules.
- Dataset paths are currently Kaggle-specific. For local execution, parameterize `DATASET_ROOT` and checkpoint paths.
- The prepared dataset and model checkpoints are intentionally not stored in Git because they are large artifacts.
- Reported metrics are tied to the notebook run captured in the current artifacts. Re-running with different seeds, checkpoint states, or dataset filtering may shift the numbers.
- The object-class scope is intentionally narrow. Extending to the full BDD100K label space will require class remapping, metric updates, and additional evaluation coverage.

## Limitations

- The pipeline evaluates a curated subset of BDD100K rather than the complete dataset.
- ProtoPNet is used as a classification-style interpretable baseline, while Faster R-CNN is a detector; comparisons should be interpreted by task-specific metrics rather than as a single unified score.
- Deraining is evaluated as a counterfactual intervention, not as a guaranteed preprocessing improvement.
- Explainability metrics are useful diagnostics, but they do not prove causal model reasoning on their own.

## Future Work

- Convert the notebook pipeline into a reusable Python package with CLI entry points.
- Add configuration files for dataset paths, model hyperparameters, and experiment tracking.
- Store metrics as structured CSV or JSON artifacts in addition to PNG plots.
- Add deterministic experiment manifests for each reported run.
- Expand the stress taxonomy into rain, fog, snow, night, and low-visibility subgroups.
- Evaluate additional detectors and modern vision-language models.
- Add CI checks for notebook execution, linting, and lightweight metric smoke tests.

## Project Status

This repository is best understood as a research prototype and experiment archive. The notebooks contain the executable methodology, `Results/` contains supporting visual evidence, `Papers/` contains background and method references, and `report.pdf` contains the compiled project report.
