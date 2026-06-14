# SFD-CAM Reproducibility Notes

## 1. Task Definition

SFD-CAM performs **3D-volume-informed 2D multi-structure OCTA segmentation**.

- Input: a 3D OCTA volume.
- Output: a 2D segmentation map.
- Foreground classes: capillary, artery, vein, and foveal avascular zone (FAZ).

## 2. Data Splits and Seeds

The experiments use OCTA-500.

| Subset | IDs | Samples | Validation per fold | Split file |
|---|---:|---:|---:|---|
| OCTA_3mm | 10301-10500 | 200 | 40 | `hypes_yaml/fold-1.json` |
| OCTA_6mm | 10001-10300 | 300 | 60 | `hypes_yaml/fold-2.json` |

Each split file stores the exact train/validation case IDs for all five folds. Within each fold, train and validation IDs are mutually exclusive. OCT, OCTA, projection maps, and labels are paired by case ID.

The SFD-CAM release configuration fixes the seed to:

```yaml
seed: 3407
```

Use this seed for initialization, fold-related reproducibility, augmentation sampling, and training unless a different seed is explicitly documented in a new experiment.

## 3. Main SFD-CAM Configuration

The main SFD-CAM configuration is `hypes_yaml/SFDFormer.yaml`.

| Item | Value |
|---|---|
| Device | CUDA |
| AMP | enabled（Transformer is not） |
| Input type | 3D |
| Modality | OCTA |
| Classes | background, capillary, artery, vein, FAZ |
| Dataset class | `OCTA500Dataset` |
| Data root | `OCTA-500` |
| Data size | `[640, 400, 400]` |
| Block size | `[160, 100, 100]` |
| Fold number | 5 |
| Ignore index | 255 |
| Train expand rate | 6 |
| Batch size | 4 |
| Epochs | 100 |
| Evaluation frequency | 1 |
| Save frequency | 80 |
| Resume training | enabled |
| Augmentor | `OCTA500` |
| Crop size | `[100, 100]` |
| Kernel size | 3 |
| Deep supervision | disabled |
| Loss | CrossEntropyLoss + Dice Loss |
| Optimizer | NAdam |
| Initial learning rate | 0.0005 |
| NAdam epsilon | 1e-10 |
| Weight decay | 1e-4 |
| Scheduler | multistep |
| Scheduler gamma | 0.1 |
| Scheduler milestones | `[10, 50, 80]` |
| Early stopping | enabled |
| Early-stopping delta | 0 |

Class weights are assigned in `tools/train.py` according to the number of classes. For five-class OCTA-500 training, the loss weight is:

```python
[1.0, 2.0, 2.0, 2.0, 2.0]
```

### Architecture Summary for the Reported SFD-CAM Setting

| Item | Reported setting |
|---|---|
| Model variant | SFD-CAM with S-size 2D segmentation backbone |
| Input channels | 1 OCTA channel |
| Output channels | 5 classes: background, capillary, artery, vein, FAZ |
| 3D block size | `[160, 100, 100]` |
| CCPM channels | 32 |
| CCPM projection branches | U branch pooling `(8, 10, 4)`; D branch pooling `(8, 5, 8)` |
| CCPM projection kernels | 3D convolution kernels 5 and 7; the 7-kernel convolution uses dilation 3 |
| 2D backbone channels | 32, 64, 128, 256, 512 |
| Backbone block counts | `[2, 2, 2, 2, 2, 2, 2, 2, 2]` |
| Expansion ratio | 2 for the S-size backbone |
| Segmentation kernel size | 3 |
| SFSD placement | skip-connection feature fusion at 32, 64, 128, and 256 channels |
| SFSD patch size | 8 |
| DSCM placement | bottleneck feature level |
| DSCM directional modeling | four directional channel groups with two directional refinement passes |

## 4. Baseline Configuration Tree

For the final public release, OCTA-500 configuration files should be provided for the major comparison methods reported in the manuscript:

| Subset | Config directory |
|---|---|
| OCTA_3mm baselines | `hypes_yaml/OCTA-500/3mm/` |
| OCTA_6mm baselines | `hypes_yaml/OCTA-500/6mm/` |
| Shared OCTA-500 examples | `hypes_yaml/OCTA-500/` |

## 5. Preprocessing and 3D-to-2D Handling

The dataset loader reads OCTA volumes and 2D labels by case ID. For the SFD-CAM task:

1. OCTA volumetric data are used as the model input.
2. 2D segmentation labels are paired by the same case ID.
3. No additional registration is applied because OCTA-500 is already aligned.
4. Training samples are randomly cropped from the available volume according to the configured block/crop sizes.
5. The learnable 3D-to-2D handling is part of the network forward path rather than a manual pre-projection step.

For the reported 3D OCTA setting, the loader samples a valid spatial crop by choosing `left` and `long_start` uniformly inside the available image range. The full depth dimension of the configured block is preserved, and only the two enface spatial dimensions are cropped. Sampling is not class-balanced in the reported setting. The reported 3D OCTA path does not use a foreground-threshold rejection step; this threshold is present only in other augmentor presets and is not active for the SFD-CAM OCTA-500 setting.

The manual preprocessing and learnable projection are separated as follows:

| Stage | Operation |
|---|---|
| Manual preprocessing | case-ID pairing, volume resizing/organization, random spatial crop, tensor conversion |
| Network projection | CCPM converts 3D OCTA features into 2D feature maps inside the model |
| Output | 2D segmentation map for background, capillary, artery, vein, and FAZ |

## 6. Augmentation Parameters

The reported 3D OCTA setting uses the OCTA-500 augmentor. Its geometric transform is replayed across all depth slices so that a single random transform is applied consistently to the complete 3D volume and its paired 2D label.

| Augmentation item | Value / range | Probability | Notes |
|---|---|---:|---|
| Random spatial crop | `[160, 100, 100]` block, preserving the depth dimension | always during training | crop origin sampled uniformly within valid bounds |
| Affine rotation | `[-15 degrees, 15 degrees]` | 0.2 | N/A |
| Affine scale | `(1, 1)` | included in affine transform | N/A                                               |
| Affine translation | `(0, 0)` | included in affine transform | N/A |
| Affine shear | `(0, 0)` | included in affine transform | N/A |
| Test-time augmentation | not used | 0 | N/A |

## 7. Training Procedure

Install the dependencies listed in the software environment section or in the environment file provided with the final public repository.

Train SFD-CAM:

```bash
python tools/train.py --hypes_yaml hypes_yaml/SFDFormer.yaml --model_dir logs/SFDCAM
```

The selected fold is controlled by:

```yaml
train_params:
  train_fold_list: 0
```

Set the target fold explicitly in the YAML file or generate fold-specific run directories as needed. For reporting cross-validation results, train and evaluate all five folds using the same configuration except for the fold selector.

## 8. Inference, Post-Processing, and Metrics

Run inference/evaluation:

```bash
python tools/inference.py --model_dir logs/SFDCAM
```

Inference uses MONAI sliding-window inference. The reported SFD-CAM setting uses:

| Item | Value |
|---|---|
| ROI size | `(160, 128, 128)` |
| Overlap | 0.25 |
| Sliding-window batch size | 1 |
| Padding | MONAI default padding behavior |
| Aggregation | MONAI default overlapping-window aggregation |

The reported post-processing removes connected components smaller than 5 pixels, using the same threshold across compared methods. Metric aggregation is implemented in:

```text
metric/calculator.py
tools/inference.py
```

Reported values in Tables 2 and 3 are mean values with standard deviations across five validation folds. mIoU and mDice are averaged over capillary, artery, vein, and FAZ, excluding background.



## 9. Software Environment

| Package | Version / note |
|---|---|
| Python | 3.10 |
| PyTorch | ≥2.00 |
| MONAI | 1.4.0 |
| scikit-learn | 1.6.1 |
| NumPy | 1.22.0 |
| OpenCV | 4.11.0.86 |
| Pillow | required |
| h5py / h5pickle | required for OCTA-500 data loading |
| TensorBoard | 2.19.0 |

The final public repository should include an environment file containing the complete dependency list.

## 10. Hardware Environment

The reported experiments were run on an Ubuntu server equipped with four NVIDIA GeForce RTX 5090 GPUs. The CUDA version was 13.0.

