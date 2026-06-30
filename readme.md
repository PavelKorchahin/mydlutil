# mydlutil

`mydlutil` is a lightweight deep-learning utility package built around PyTorch. It focuses on single-label supervised learning and provides a packaged configuration and training framework.
It mainly provides configuration-driven dataset loading, pretrained-model loading, training, evaluation, visualization, image utilities, and several reusable loss and helper classes.

The fastest way to use the package is to configure three files and then run one training script:

1. `config.py` — define project-level root directories.
2. `dataset_config.py` — register datasets in `DATASET_CONFIG`.
3. `model_config.py` — register pretrained models in `MODEL_CONFIG`.

> **Source of truth**
>
> This README explains the main configuration items needed to start quickly. For every optional or advanced item, use the in-code comments and docstrings as the authoritative reference:
>
> - Project paths and acceleration: `config.py`
> - Dataset configuration fields: `load_data.py`, class `DatasetConfig`
> - Train/test dataset aliases: `load_data.py`, function `generate_datadset`
> - DataLoader options and return values: `load_data.py`, function `load_data`
> - Model registry fields: `model_config.py` and `load_model.py`, functions `generate_pretrained_model`, `download_pretrained_model`, and `load_pretrained_model`
> - Training parameters: `train.py`, functions `train_epoch` and `train`
> - Evaluation parameters and metrics: `metric.py`, classes `ClassificationEvaluate` and `ShowEvaluateResult`

---

# Package Directory Structure

```text
mydlutil/
├── __init__.py          # Public package imports
├── _function.py         # Dataset/model callbacks, downloaders, and loader helpers
├── accumulator.py       # Numeric accumulator used during training
├── charts.py            # Line-chart utility
├── config.py            # Project, dataset, pretrained-model, and result-model roots
├── dataset_config.py    # DATASET_CONFIG registry and dataset path helper
├── diceloss.py          # DiceLoss and DiceCELoss
├── img.py               # Image opening and visualization helpers
├── load_data.py         # Dataset configuration, dataset classes, downloads, and DataLoaders
├── load_model.py        # Pretrained-model downloading, generation, and loading
├── metric.py            # Classification/segmentation metrics and visual evaluation
├── model_config.py      # MODEL_CONFIG registry and result-model save/load helpers
├── test.py              # Debug printing and text-file output helpers
└── train.py             # One-epoch and full-training workflows
```
---
# Dependencies
```text
huggingface_hub>=1.15.0
kagglehub>=1.0.2
matplotlib>=3.11.0
numpy>=2.5.0
opencv_contrib_python>=4.13.0.92
opencv_python>=4.12.0.88
opencv_python_headless>=4.13.0.92
pandas>=3.0.3
Pillow>=12.2.0
ptflops>=0.7.5
scikit_learn>=1.9.0
seaborn>=0.13.2
tifffile>=2025.10.16
torch>=2.11.0+cu128
torchvision>=0.26.0+cu128
tqdm>=4.67.3
transformers>=5.8.1
typing_extensions>=4.15.0
```

Install the versions appropriate for your Python, CUDA, and PyTorch environment. GPU training additionally requires a working CUDA-enabled PyTorch installation when using a CUDA device.

---
# How to Start

## 0.Note
- All the custom functions should be written in the `_function` module (referred to as `fn` in the example below) or outer, one should not use lambda functions or inner fuctions unless he /she 
  ensure that it can be serialized by pytorch.
- It can't handle data with multi-label very well right now.

## 1. Configure the project roots

Edit [config.py](config.py) before loading any dataset or model.

```python
from pathlib import Path
import platform

SYSTEM = platform.system()

ROOT = Path("D:/projects/my_research")
DATASET_DIR = Path("D:/data/datasets")
PRETRAINED_MODEL_DIR = Path("D:/models/pretrained")
RESULT_MODEL_DIR = Path("D:/models/results")

NEED_ACCELERATE_SYSTEM = ("linux", "unix")
```


### Meaning of every project configuration item

| Item                     | Meaning                                                                                                                                                                                                                                                                              |
|--------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `SYSTEM`                 | Operating-system name returned by `platform.system()`, such as `Windows`, `Linux`, or `Darwin`. It is normally detected automatically and should not need manual editing.                                                                                                            |
| `ROOT`                   | Root directory of the current research project. Relative paths passed to `get_local_path()` are resolved under this directory. Use it for logs, figures, reports, and other project outputs that are not dataset files or model files.                                               |
| `DATASET_DIR`            | Root directory containing all datasets. Every relative dataset path in `DATASET_CONFIG` is resolved relative to this directory.                                                                                                                                                      |
| `PRETRAINED_MODEL_DIR`   | Root directory containing downloaded or manually prepared pretrained models. Every relative `model_dir` in `MODEL_CONFIG` is resolved relative to this directory.                                                                                                                    |
| `RESULT_MODEL_DIR`       | Root directory for trained result models or saved training artifacts. Relative paths passed through `save_model()`, `load_model()`, or the `model_save` argument of `train()` are resolved relative to this directory.                                                               |
| `NEED_ACCELERATE_SYSTEM` | Operating-system names on which `mydlutil` enables its accelerated loading/training branch. The current code uses extra DataLoader workers, pinned memory, prefetching, automatic mixed precision, and higher matrix-multiplication precision when `is_accelerate()` returns `True`. |
| `get_local_path(path)`   | Returns an absolute `Path`. An absolute input is kept as-is; a relative input is resolved under `ROOT`.                                                                                                                                                                              |
| `is_accelerate()`        | Returns whether the current operating system is listed in `NEED_ACCELERATE_SYSTEM`.                                                                                                                                                                                                  |

Absolute paths are accepted directly. For relative paths, use the root associated with the relevant configuration item: dataset paths use `DATASET_DIR`, pretrained-model paths use `PRETRAINED_MODEL_DIR`, result-model paths use `RESULT_MODEL_DIR`, and general project outputs can be converted through `get_local_path()` under `ROOT`. The `log_save` and `loss_curve_save` arguments are used directly by `train()`, so pass an absolute path or call `get_local_path()` first.

Examples:

```python
from mydlutil.config import get_local_path
from mydlutil.dataset_config import get_dataset_path
from mydlutil.model_config import get_pretrained_model_path

log_file = get_local_path("research/results/train.log")
dataset_folder = get_dataset_path("CustomSegmentation/images/train")
pretrained_folder = get_pretrained_model_path("SegFormer/segformer-b1-ade")
```

---

## 2. Configure a dataset

Datasets are registered in [dataset_config.py](dataset_config.py) through the `DATASET_CONFIG` dictionary.

- The top-level key is the dataset name passed to `load_data()`. see [load_data.py](load_data.py)
- The top-level value is a dictionary of dataset configuration fields.
- All relative paths inside a dataset entry are relative to `config.DATASET_DIR`.
- The complete meaning of advanced fields is defined by the comments of `load_data.DatasetConfig`. see [load_data.py](load_data.py)
- Fields that separately define training and testing sources are handled by `load_data.generate_datadset`.

### Example extracted from the package configuration

```python
import mydlutil._function as fn 
DATASET_CONFIG = {
    "ade20ksegmentation": {
        "name": "ADE20KSegmentationDataset",
        "get_train_data": "ADE20K/ADEChallengeData2016/images/training",
        "get_train_label": "ADE20K/ADEChallengeData2016/annotations/training",
        "get_test_data": "ADE20K/ADEChallengeData2016/images/validation",
        "get_test_label": "ADE20K/ADEChallengeData2016/annotations/validation",
        "original_size": 512,
        "num_class": 151,
        "pixelwise": True,
        "class_label": ("wall", "building", "sky"),
        "channels": 3,
        "get_index": fn.get_ade_index,
    }
}
```

The `class_label` tuple above is intentionally shortened for documentation. In the actual configuration, provide the complete ordered class list.

### Recommended custom segmentation dataset entry

Assume the following directory layout under `DATASET_DIR`:

```text
CustomSegmentation/
├── images/
│   ├── train/
│   └── val/
└── masks/
    ├── train/
    └── val/
```

Add this entry to `DATASET_CONFIG`:

```python
DATASET_CONFIG = {
    # Existing datasets...

    "custom_segmentation": {
        "name": "CustomSegmentationDataset",
        "get_train_data": "CustomSegmentation/images/train",
        "get_train_label": "CustomSegmentation/masks/train",
        "get_test_data": "CustomSegmentation/images/val",
        "get_test_label": "CustomSegmentation/masks/val",
        "original_size": 512,
        "num_class": 4,
        "pixelwise": True,
        "class_label": ["background", "class_a", "class_b", "class_c"],
        "channels": 3,
        "target_suffixes": (".jpg", ".jpeg", ".png"),
    },
}
```

Then load it by its registry key:

```python
from mydlutil.load_data import load_data

train_iter, test_iter = load_data(
    "custom_segmentation",
    batch_size=8,
    target_size=256,
    normalize=True,
    only_get_dataloader=True,
)
```

### Dataset dictionary fields

The following fields are accepted by `DatasetConfig`, `generate_datadset`, or both.see [load_data.py](load_data.py)

#### Identity and data sources

| Field             | Meaning                                                                                                                                                                                                           |
|-------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `name`            | Human-readable dataset name. It is also used as the default destination name when automatic downloading is requested.                                                                                             |
| `get_train_data`  | Training feature source. Usually a directory, but it may also be a list or another source form supported by `DatasetConfig.get_data`. Relative paths are resolved under `DATASET_DIR`.                            |
| `get_test_data`   | Testing or validation feature source. Relative paths are resolved under `DATASET_DIR`.                                                                                                                            |
| `get_train_label` | Training label source. For segmentation this is usually a mask directory; for ordinary classification it may be a text file, list, callable, or another supported source.                                         |
| `get_test_label`  | Testing or validation label source.                                                                                                                                                                               |
| `get_data`        | Shared feature source used for both training and testing when both sources are the same. It is an abbreviation accepted by `generate_datadset`. Do not define it together with conflicting split-specific fields. |
| `get_label`       | Shared label source used for both training and testing when both sources are the same.                                                                                                                            |
| `get_train`       | Abbreviation that assigns the same source to both training features and training labels. Use only when that structure is intentional.                                                                             |
| `get_test`        | Abbreviation that assigns the same source to both testing features and testing labels.                                                                                                                            |
| `original_size`   | Original image size metadata and the initial size used by the generated dataset class. It follows the same size form as `DatasetConfig.target_size`.                                                              |
| `pixelwise`       | Set to `True` when each target is an image or mask aligned with the feature. Set to `False` for ordinary sample-level labels.                                                                                     |
| `num_class`       | Number of classes. Class IDs should start at `0` and be contiguous.                                                                                                                                               |
| `class_label`     | Ordered class names. The index of each item is treated as its class ID.                                                                                                                                           |

#### File discovery and split rules

| Field                 | Meaning                                                                                                                                       |
|-----------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| `data_pattern`        | Shared feature-file filter for both splits. It may be a callable mapping a filename stem to `bool`, or a text file containing selected names. |
| `label_pattern`       | Shared label-file filter for both splits.                                                                                                     |
| `train_data_pattern`  | Feature-file filter used only for training data.                                                                                              |
| `test_data_pattern`   | Feature-file filter used only for testing data.                                                                                               |
| `train_label_pattern` | Label-file filter used only for training labels.                                                                                              |
| `test_label_pattern`  | Label-file filter used only for testing labels.                                                                                               |
| `train_pattern`       | Abbreviation that applies the same split rule to training features and training labels.                                                       |
| `test_pattern`        | Abbreviation that applies the same split rule to testing features and testing labels.                                                         |
| `target_suffixes`     | Allowed file suffixes when scanning directories. Default: `(".jpg", ".jpeg", ".png")`. A single string is also accepted.                      |

#### Label mapping and preprocessing

| Field             | Meaning                                                                                                                                                                                                           |
|-------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `label_to_idx`    | Mapping between class names, class IDs, or pixel tuples and class IDs/names. It is commonly used to convert color masks into contiguous class indices.                                                            |
| `preprocess`      | Whether to run persistent, file-level preprocessing. Because this may modify source files, back up the raw dataset first and normally enable it only once.                                                        |
| `preprocess_func` | Callable that receives a `DatasetConfig` instance and performs persistent preprocessing. The package helper `mydlutil._function.map_pixel` is an example.                                                         |
| `get_index`       | Final callback applied before a sample is returned. It receives `(result, config)` and may modify the feature/label output. The default helper validates pixelwise labels and maps invalid class values to `255`. |

#### File readers and tensor layout

| Field        | Meaning                                                                                                                                                                                                   |
|--------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `open_data`  | Callable used to open or convert a feature source into a PIL image, NumPy array, or tensor. The default path-based image reader is `PIL.Image.open`. Use readers such as `tifffile.imread` for TIFF data. |
| `open_label` | Callable used to open or convert label files. Do not set it for ordinary string labels read from a text file.                                                                                             |
| `channels`   | Selects or aligns feature channels. An integer or a list of channel indices is accepted.                                                                                                                  |
| `label_dims` | Required target dimensionality before batching. The loader inserts leading dimensions until this dimensionality is reached.                                                                               |

#### Runtime dataset options

These fields are usually supplied to `load_data()` instead of being permanently fixed in `DATASET_CONFIG`.

| Field         | Meaning                                                                                                                                                                                                                                                                                               |
|---------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mode`        | Dataset mode: `"train"`, `"test"`, or `"img"`. Training mode can augment and normalize data. Test mode does neither. Image mode uses the test source but may apply training-style normalization/processing. Normally let `load_data()` set this automatically.                                        |
| `target_size` | Output size. An integer means a square size; a tuple means `(height, width)`.                                                                                                                                                                                                                         |
| `dset_size`   | Maximum number of base samples to use. If omitted or larger than the available data, all samples are used.                                                                                                                                                                                            |
| `device`      | Device used while loading/processing dataset samples. The source comments recommend keeping dataset loading on CPU and using the `device` argument of `train()` for GPU training.                                                                                                                     |
| `resize_mode` | `"pad"` preserves aspect ratio and pads to the requested size; `"nopad"` directly resizes features with bilinear interpolation and pixelwise labels with nearest-neighbor interpolation.                                                                                                              |
| `normalize`   | Enables feature normalization in `"train"` or `"img"` mode. It is applied only to three-channel features.                                                                                                                                                                                             |
| `mean`        | Per-channel normalization mean. Default: ImageNet mean `(0.485, 0.456, 0.406)`.                                                                                                                                                                                                                       |
| `std`         | Per-channel normalization standard deviation. Default: ImageNet standard deviation `(0.229, 0.224, 0.225)`.                                                                                                                                                                                           |
| `transform`   | Data-augmentation list. A callable transforms only the feature. A `(feature_transform, label_transform)` pair transforms both and is recommended for segmentation. Add `nn.Identity()` or `(nn.Identity(), nn.Identity())` when the original unaugmented sample must remain in the augmented dataset. |

#### Download metadata

| Field           | Meaning                                                                                                                                   |
|-----------------|-------------------------------------------------------------------------------------------------------------------------------------------|
| `handle`        | Remote dataset identifier understood by `download_func`. When configured data cannot be found, the loader may ask whether to download it. |
| `download_func` | Callable with parameters `handle` and `out_dir`. The default helper uses Kaggle Hub.                                                      |

> **Important:** use the canonical field names defined by `DatasetConfig` and `generate_datadset`. For example, use `num_class`, `target_suffixes`, `handle`, and `preprocess_func`. Advanced or custom fields passed through `**kwargs` are retained as attributes, but they are not automatically used unless another part of the code reads them.

The bundled registry contains several legacy or non-canonical names. Their intended meanings and current equivalents are:

| Existing key      | Intended meaning                  | Canonical key used by the loader   |
|-------------------|-----------------------------------|------------------------------------|
| `download`        | Remote dataset identifier         | `handle`                           |
| `download_mode`   | Download backend selection        | Configure `download_func` directly |
| `preprocess_mode` | Persistent preprocessing callback | `preprocess_func`                  |
| `target_suffixe`  | Allowed file suffix               | `target_suffixes`                  |
| `num_classes`     | Number of classes                 | `num_class`                        |

Use the canonical names in new dataset entries so that the current loader reads them directly.

---

## 3. Configure a pretrained model

Models are registered in [model_config.py](model_config.py) through the nested `MODEL_CONFIG` dictionary.

- The first-level key is the model name passed to `load_pretrained_model(name, ...)`. see [load_model.py](load_model.py)
- The second-level key is the version passed to `load_pretrained_model(..., version=...)`. see [load_model.py](load_model.py)
- Relative `model_dir` values are resolved under `config.PRETRAINED_MODEL_DIR`.
- All parameters required by the selected loader should be placed in the version dictionary.

### Example extracted from the package configuration

```python
import mydlutil._function as fn
import transformers
import torch.nn.functional as F
MODEL_CONFIG = {
    "segformer": {
        "b1_ade": {
            "model_dir": "SegFormer/segformer-b1-ade",
            "load_func": fn.load_pretrained_model_func_by_huggingface,
            "model_cls": transformers.SegformerForSemanticSegmentation,
            "out": lambda out: F.interpolate(
                out.logits,
                scale_factor=4,
                mode="bilinear",
                align_corners=False,
            ),
            "ignore_mismatched_sizes": True,
            "handle": "nvidia/segformer-b1-finetuned-ade-512-512",
            "download_func": fn.download_pretrained_model_func_by_huggingface,
        }
    }
}
```

Load this model with:

```python
from mydlutil import load_pretrained_model

net = load_pretrained_model(
    "segformer",
    version="b1_ade",
    num_labels=150,
)
```

The call-time keyword `num_labels=150` is merged into the selected version configuration and forwarded to the model loader.

### Model dictionary fields

| Field           | Meaning                                                                                                                                                                                                                      |
|-----------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `model_dir`     | Local pretrained-model directory. An absolute path is used directly; a relative path is resolved under `PRETRAINED_MODEL_DIR`.                                                                                               |
| `model_cls`     | Model class, normally a subclass of `torch.nn.Module`. For a Hugging Face model, this is usually a `transformers` model class with `from_pretrained()`.                                                                      |
| `load_func`     | Callable that accepts `model_dir`, `model_cls`, and loader-specific keyword arguments, then returns the loaded model. The default helper calls `model_cls.from_pretrained(model_dir, **kwargs)`.                             |
| `out`           | Callable or `nn.Module` used to convert the raw model output into the tensor format required by the loss and evaluator. `generate_pretrained_model()` wraps the pretrained model and this output adapter in `nn.Sequential`. |
| `handle`        | Remote model identifier used when the local model directory is missing or empty.                                                                                                                                             |
| `download_func` | Callable accepting `handle` and `out_dir`. The default Hugging Face helper downloads a repository snapshot.                                                                                                                  |
| Additional keys | Loader-specific keyword arguments forwarded to `load_func`, such as `ignore_mismatched_sizes` or `num_labels`.                                                                                                               |

For complete behavior, see:

- `model_config.py`, comments immediately above `MODEL_CONFIG`
- `load_model.generate_pretrained_model`
- `load_model.download_pretrained_model`
- `load_model.load_pretrained_model`

When a configured pretrained-model directory does not exist or is empty, the current loader asks whether it should download the model. After downloading, it exits and asks you to verify the configuration and run the training script again.

---

## 4. Define augmentation correctly

see parameters `transform` of `DatasetConfig` class in [load_data.py](load_data.py)

For pixelwise prediction, apply geometrically identical geometric operations to the feature and label. Feature-only appearance transforms such as `ColorJitter` should be supplied as a single callable so that the mask is not modified. Use bilinear interpolation for images and nearest-neighbor interpolation for masks.

```python
from torch import nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode

transforms_list = [
    (nn.Identity(), nn.Identity()),
    transforms.ColorJitter(),
    (
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
    ),
    (
        transforms.RandomRotation(
            45,
            interpolation=InterpolationMode.BILINEAR,
        ),
        transforms.RandomRotation(
            45,
            interpolation=InterpolationMode.NEAREST,
        ),
    ),
]
```

The identity pair keeps the original sample in the augmented dataset. Without it, the transformed variants replace the original variant.

---

## 5. Load the configured dataset

The common training form is:

```python
from mydlutil.load_data import load_data
transforms_list = [...]  # data augmentation (see the example in #4)
train_iter, test_iter = load_data(
    "ade20ksegmentation",
    batch_size=4,
    target_size=256,
    normalize=True,
    transform=transforms_list,
    only_get_dataloader=True,
)
```

### Important `load_data()` parameters

| Parameter             | Meaning                                                                                                              |
|-----------------------|----------------------------------------------------------------------------------------------------------------------|
| `dataset`             | Dataset registry key, configuration dictionary, `DatasetConfig` instance, or generated dataset class.                |
| `batch_size`          | Number of samples per batch.                                                                                         |
| `drop_last`           | Drops an incomplete final batch when `True`.                                                                         |
| `only_get_dataset`    | Returns only the generated dataset class. It cannot be enabled together with `only_get_dataloader`.                  |
| `only_get_dataloader` | Returns only `(train_iter, test_iter)`. This is the simplest option for training scripts.                            |
| `train_map`           | Final callable applied to the instantiated training dataset.                                                         |
| `test_map`            | Final callable applied to the instantiated testing dataset.                                                          |
| `dset_map`            | Callable applied to the generated dataset class before train/test instances are created.                             |
| `test_mode`           | `"test"` for unnormalized, non-augmented test data; `"img"` for test-source data that may use image-mode processing. |
| `**kwargs`            | Runtime `DatasetConfig` parameters such as `target_size`, `dset_size`, `resize_mode`, `normalize`, and `transform`.  |

### Return forms
Default
```python
from mydlutil.load_data import load_data

Dset, (train_iter, test_iter) = load_data("dataset_name")

```
DataLoaders only
```python
from mydlutil.load_data import load_data
train_iter, test_iter = load_data(
    "dataset_name",
    only_get_dataloader=True,
)
```
Generated dataset class only
```python
from mydlutil.load_data import load_data
Dset = load_data(
    "dataset_name",
    only_get_dataset=True,
)
```

---

## 6. Configure loss, optimizer, scheduler, and training

see [train.py](train.py)

Example:

```python
from torch import optim
from mydlutil import DiceCELoss

loss = DiceCELoss(ignore_index=255)
net = ... # model
head_params = [
    parameter
    for name, parameter in net.named_parameters()
    if "decode_head.classifier" in name
]

backbone_params = [
    parameter
    for name, parameter in net.named_parameters()
    if "decode_head.classifier" not in name
]

trainer = optim.AdamW(
    [
        {"params": backbone_params, "lr": 3e-5},
        {"params": head_params, "lr": 3e-4},
    ],
    weight_decay=0.01,
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    trainer,
    T_max=30,
)
```

### Important `train()` parameters in  [train.py](train.py)

| Parameter                  | Meaning                                                                                                                                                                                                 |
|----------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `net`                      | Model to train.                                                                                                                                                                                         |
| `num_class`                | Number of output classes used by evaluation.                                                                                                                                                            |
| `train_iter`               | Training DataLoader.                                                                                                                                                                                    |
| `test_iter`                | Testing or validation DataLoader.                                                                                                                                                                       |
| `loss`                     | Loss module receiving `(model_output, target)`.                                                                                                                                                         |
| `num_epochs`               | Maximum training epochs.                                                                                                                                                                                |
| `trainers`                 | One optimizer or a sequence of optimizers for staged training.                                                                                                                                          |
| `trainers_epoch_threshold` | Epochs at which each optimizer becomes active. It must start with `0` and match the number of optimizers. An integer `n` means `(0, n)` for two optimizers.                                             |
| `device`                   | Training device, such as `"cuda:0"` or `"cpu"`.                                                                                                                                                         |
| `lr_scheduler`             | Optional learning-rate scheduler.                                                                                                                                                                       |
| `lr_scheduler_type`        | `"epoch"` updates the scheduler after each epoch; `"batch"` updates it after each batch.                                                                                                                |
| `pixelwise`                | Set to `True` for segmentation or other pixelwise tasks.                                                                                                                                                |
| `loss_curve_save`          | Figure path for the training-loss and metric curves. The path is used directly, so use an absolute path or `get_local_path(...)`.                                                                       |
| `metrics`                  | Overall evaluation metrics calculated after each epoch. Common choices: `"accuracy"`, `"moa"`, `"miou"`, `"fwiou"`, `"f1"`, and `"weighted_f1"`. Aliases documented in `train.train` are also accepted. |
| `model_save`               | Model-result path. A relative value is resolved under `RESULT_MODEL_DIR`.                                                                                                                               |
| `log_save`                 | Training-log path used directly by `train()`. Prefer an absolute path or `get_local_path(...)`.                                                                                                         |
| `log_prompt`               | Header written before one training run in the log file. Use it to record the experiment name and important hyperparameters.                                                                             |
| `save_query`               | When `True`, asks for confirmation before saving configured outputs.                                                                                                                                    |
| `early_stopping_metric`    | Metric to monitor, or `"loss"`. It must be present in `metrics` unless it is `"loss"`.                                                                                                                  |
| `early_stopping_times`     | Number of consecutive non-improving epochs allowed before stopping.                                                                                                                                     |
| `early_stopping_mode`      | `"max"` for metrics such as mIoU/accuracy; `"min"` for loss.                                                                                                                                            |
| `early_stopping_min_delta` | Minimum improvement required to reset the early-stopping counter.                                                                                                                                       |

The function returns:

```python
from mydlutil.train import train
curve_axes, actual_epoch_count, best_epoch_number = train(...)
```

> **Saving behavior:** the current implementation tracks the best state dictionary through the early-stopping branch and passes it to `save_model()`. When using `model_save`, configure a valid `early_stopping_metric` so that a best checkpoint is selected.

---

## 7. Evaluate the trained model

see [metric.py](metric.py)

For a quick visual report:

```python
from mydlutil.metric import ShowEvaluateResult
net = ...  # model
num_class =...  # number of classes
test_iter =...  # testing DataLoader
evaluator = ShowEvaluateResult(
    net=net,
    num_class=num_class,
    test_iter=test_iter,
    pixelwise=True,
    device="cuda:0",
)

figure = evaluator.show_metric_figure(klist=[None, 0])
figure.show()
```

`klist=[None, 0]` means:

- `None`: show overall confusion matrix and overall metrics.
- `0`: show the class-specific confusion matrix and metrics for class `0`.

Other useful methods include:

```python
evaluator = ...  # above
evaluator.print_cm()
evaluator.show_confusion_matrix()
evaluator.show_all_metrics()
evaluator.show_result_img(start_idx=0, num=10)
image_figure, metric_figure = evaluator.show_all_figure()
```

For numeric access, instantiate `ClassificationEvaluate` directly and call methods such as `accuracy()`, `mean_iou()`, `fw_iou()`, `f1()`, `all_metric()`, or `get_metrics()`.

---

## 8. Complete end-to-end training example

The following example loads ADE20K, loads a configured SegFormer model, fine-tunes it with different learning rates for the backbone and classifier head, applies early stopping based on mIoU, saves logs under the project root, saves the best model result under the result-model root, and visualizes evaluation metrics.

```python
from torch import nn, optim
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import matplotlib.pyplot as plt
from mydlutil import DiceCELoss
from mydlutil import load_pretrained_model
from mydlutil.config import get_local_path
from mydlutil.load_data import load_data
from mydlutil.metric import ShowEvaluateResult
from mydlutil.train import train


NUM_CLASSES = 150
BATCH_SIZE = 4
NUM_EPOCHS = 30
DEVICE = "cuda:0"


if __name__ == "__main__":
    augmentation = [
        (nn.Identity(), nn.Identity()),
        transforms.ColorJitter(),
        (
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
        ),
        (
            transforms.RandomRotation(
                45,
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.RandomRotation(
                45,
                interpolation=InterpolationMode.NEAREST,
            ),
        ),
    ]

    train_iter, test_iter = load_data(
        "ade20ksegmentation",
        batch_size=BATCH_SIZE,
        target_size=256,
        normalize=True,
        transform=augmentation,
        only_get_dataloader=True,
    )

    net = load_pretrained_model(
        "segformer",
        version="b1_ade",
        num_labels=NUM_CLASSES,
    )

    head_params = [
        parameter
        for name, parameter in net.named_parameters()
        if "decode_head.classifier" in name
    ]

    backbone_params = [
        parameter
        for name, parameter in net.named_parameters()
        if "decode_head.classifier" not in name
    ]

    loss = DiceCELoss(ignore_index=255)

    trainer = optim.AdamW(
        [
            {"params": backbone_params, "lr": 3e-5},
            {"params": head_params, "lr": 3e-4},
        ],
        weight_decay=0.01,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        trainer,
        T_max=NUM_EPOCHS,
    )

    train(
        net=net,
        num_class=NUM_CLASSES,
        train_iter=train_iter,
        test_iter=test_iter,
        loss=loss,
        num_epochs=NUM_EPOCHS,
        trainers=trainer,
        device=DEVICE,
        pixelwise=True,
        lr_scheduler=scheduler,
        early_stopping_metric="miou",
        early_stopping_times=7,
        early_stopping_mode="max",
        log_save=get_local_path("research/results/ade20k.txt"),
        log_prompt="ADE20K with base augmentation",
        model_save="segformer/ade20k_base.pth",
    )

    evaluator = ShowEvaluateResult(
        net=net,
        num_class=NUM_CLASSES,
        test_iter=test_iter,
        pixelwise=True,
        device=DEVICE,
    )

    figure = evaluator.show_metric_figure(klist=[None, 0])
    plt.show()
```

### What each path in the example means

```python
from mydlutil.config import get_local_path
log_save=get_local_path("research/results/ade20k.txt")
```

The relative path is resolved under `config.ROOT`.

```python
model_save="segformer/ade20k_base.pth"
```

The relative path is resolved under `config.RESULT_MODEL_DIR`.

The dataset paths in the `ade20ksegmentation` entry are resolved under `config.DATASET_DIR`, and the SegFormer `model_dir` is resolved under `config.PRETRAINED_MODEL_DIR`.

---

## 9. Quick-start checklist

Before running training, verify the following:

- `ROOT`, `DATASET_DIR`, `PRETRAINED_MODEL_DIR`, and `RESULT_MODEL_DIR` exist or can be created.
- The dataset registry key passed to `load_data()` exists in `DATASET_CONFIG`.
- Dataset paths are written relative to `DATASET_DIR`, unless absolute paths are intentionally used.
- Feature and label files are correctly paired after alphabetical sorting or split-file filtering.
- `num_class`, `class_label`, mask IDs, model output channels, and `num_labels` agree. For the bundled ADE20K callback, labels are shifted to class IDs `0` through `149`, with `255` used as the ignore value; the example therefore uses `150` output classes.
- Segmentation labels use nearest-neighbor interpolation during geometric augmentation.
- Ignore labels such as `255` are handled consistently by the dataset callback and loss.
- The model registry key and version passed to `load_pretrained_model()` exist in `MODEL_CONFIG`.
- `model_dir` is relative to `PRETRAINED_MODEL_DIR`, unless it is absolute.
- `model_save` is relative to `RESULT_MODEL_DIR`, unless it is absolute.
- `log_save` and `loss_curve_save` are absolute or converted with `get_local_path()`.
- `early_stopping_mode` is `"max"` for accuracy/mIoU and `"min"` for loss.
- A valid `early_stopping_metric` is set when saving the best training result with the current implementation.


# Module Reference

## [config.py](config.py)

### Main items

- `SYSTEM`
- `ROOT`
- `DATASET_DIR`
- `PRETRAINED_MODEL_DIR`
- `RESULT_MODEL_DIR`
- `NEED_ACCELERATE_SYSTEM`
- `get_local_path(path)`
- `is_accelerate()`

Use this module to define machine-specific roots and project-output locations.

## [dataset_config.py](dataset_config.py)

### Main items

- `DATASET_CONFIG`
- `get_dataset_path(path)`

`get_dataset_path()` resolves relative paths under `DATASET_DIR`.

## [model_config.py](model_config.py)

### Main items

- `MODEL_CONFIG`
- `get_pretrained_model_path(path)`
- `save_model(model, path)`
- `load_model(path)`

`save_model()` and `load_model()` resolve relative paths under `RESULT_MODEL_DIR`.

## [load_data.py](load_data.py)

### Main classes

| Class            | Purpose                                                                                                                                                    |
|------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `DatasetConfig`  | Stores the complete configuration of one dataset mode. Its docstring defines the supported dataset fields.                                                 |
| `GenericDataset` | Base PyTorch dataset that performs tensor conversion, resizing, normalization, augmentation, channel handling, label mapping, and final sample processing. |
| `ImageDataset`   | Resolves image/label sources, scans directories, opens files, validates pairing, and optionally offers downloading.                                        |
| `Dset`           | Generated dataset implementation used for final train/test instances.                                                                                      |
| `ZipDataset`     | Combines one selected output position from several datasets, useful for multi-source or multi-label workflows.                                             |

### Main functions

| Function                              | Purpose                                                                                                                                                      |
|---------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `generate_datadset(...)`              | Generates a configured `Dset` class. The function name is spelled `generate_datadset` in the current source and should be imported with that exact spelling. |
| `get_dataset(nameorconfig, **kwargs)` | Resolves a registry key, dictionary, or `DatasetConfig` into a generated dataset class.                                                                      |
| `load_data(...)`                      | Creates training/testing dataset instances and DataLoaders.                                                                                                  |
| `download_dataset(...)`               | Downloads one or more datasets through a configurable downloader.                                                                                            |
| `data_show(...)`                      | Displays selected dataset samples and labels.                                                                                                                |

## [load_model.py](load_model.py)

### Main classes and functions

| Item                                             | Purpose                                                                           |
|--------------------------------------------------|-----------------------------------------------------------------------------------|
| `FunctionModel`                                  | Wraps an ordinary callable as an `nn.Module`, mainly for model-output adaptation. |
| `download_pretrained_model(...)`                 | Downloads one or more pretrained models into `PRETRAINED_MODEL_DIR`.              |
| `generate_pretrained_model(...)`                 | Loads a pretrained model and appends the configured output adapter.               |
| `load_pretrained_model(name, version, **kwargs)` | Loads a registered model/version from `MODEL_CONFIG`.                             |

## [train.py](train.py)

### Main functions

| Function           | Purpose                                                                                                                                        |
|--------------------|------------------------------------------------------------------------------------------------------------------------------------------------|
| `train_epoch(...)` | Runs one training epoch and returns average loss. It supports epoch-level or batch-level scheduler stepping.                                   |
| `train(...)`       | Runs the full training loop, evaluates after every epoch, supports staged optimizers, logging, curve saving, model saving, and early stopping. |

## [metric.py](metric.py)

### Main classes

| Class                    | Purpose                                                                                                                                                                                           |
|--------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `ClassificationEvaluate` | Computes confusion matrices and classification/segmentation metrics. Supported metrics include accuracy, precision, recall, specificity, F1, IoU, mean IoU, frequency-weighted IoU, ROC, and AUC. |
| `ShowEvaluateResult`     | Visualizes confusion matrices, metric bars, ROC curves, predictions, and combined evaluation figures.                                                                                             |
| `EvaluateModules`        | Provides model-analysis utilities, including effective receptive-field visualization, metric comparison across datasets, and FLOPs/parameter reporting.                                           |

## [diceloss.py](diceloss.py)

### Main classes

| Class        | Purpose                                                                                       |
|--------------|-----------------------------------------------------------------------------------------------|
| `DiceLoss`   | Multiclass Dice loss with optional ignored target index.                                      |
| `DiceCELoss` | Sum of Dice loss and cross-entropy loss with optional class weights and ignored target index. |

Example:

```python
from mydlutil import DiceCELoss

loss = DiceCELoss(ignore_index=255)
```

## [img.py](img.py)

### Main functions

| Function        | Purpose                                                                                      |
|-----------------|----------------------------------------------------------------------------------------------|
| `open_img(...)` | Opens ordinary images or TIFF files, optionally resizes them, and returns a tensor or array. |
| `img_show(...)` | Displays one or more PIL images, NumPy arrays, or tensors in a Matplotlib figure.            |

## [charts.py](charts.py)

### Main function

- `plot_in_one_chart(...)` — draws one or more lines on one Matplotlib axis and optionally saves the figure.

## [accumulator.py](accumulator.py)

### Main class

- `Accumulator(n)` — maintains `n` running numeric totals. It supports `add`, `clear`, indexing, length queries, and resizing.

## [_function.py](_function.py)

This module stores reusable callbacks and backend helpers. It is the recommended location for custom named functions that must be referenced from configuration dictionaries.

Included helpers cover:

- Dataset filters and final sample callbacks
- ADE20K/GF label handling
- Pixel-value-to-class preprocessing
- Kaggle dataset/model downloading
- Hugging Face model downloading and loading
- State-dictionary-based local model loading
- Identity and always-true callbacks

Prefer a top-level named function here instead of a lambda or nested function when the callback must be reused or serialized through configuration.

## [test.py](test.py)

### Main functions

- `debug(...)` — structured debug printing with optional program termination.
- `printf(file, content, mode="a+")` — writes one string or a list of strings to a text file.



## [init](__init__.py)

The package initializer re-exports utilities from several modules, including losses, metrics, chart/image helpers, model configuration, model loading, and debugging helpers. Data loading and training are normally imported explicitly:

```python
from mydlutil.load_data import load_data
from mydlutil.train import train
```

---

# Common Usage Patterns

## Load a registered dataset and inspect samples

```python
from mydlutil.load_data import load_data, data_show

Dset, (train_iter, test_iter) = load_data(
    "custom_segmentation",
    batch_size=4,
    target_size=256,
)

train_dataset = Dset(mode="train", target_size=256)
figure = data_show(train_dataset, start_idx=0, num=6)
figure.show()
```

## Load a model without using the registry

```python
import transformers

from mydlutil.load_model import generate_pretrained_model

net = generate_pretrained_model(
    model_dir="SegFormer/segformer-b1-ade",
    model_cls=transformers.SegformerForSemanticSegmentation,
    num_labels=150,
)
```

## Save and load a result artifact

```python
from mydlutil.model_config import save_model, load_model
net = ...  # trained model
save_model(net, "experiments/model.pth")
loaded_object = load_model("experiments/model.pth")
```

Both relative paths are resolved under `RESULT_MODEL_DIR`.

## Use multiple optimizers in stages

```python
from mydlutil.train import train
net = ...  # model
num_class =...  # number of classes
train_iter =...  # training DataLoader
test_iter =...  # testing DataLoader
loss =...  # loss function
optimizer_stage_1, optimizer_stage_2 =...  # multiple optimizers
train(
    net=net,
    num_class=num_class,
    train_iter=train_iter,
    test_iter=test_iter,
    loss=loss,
    num_epochs=30,
    trainers=(optimizer_stage_1, optimizer_stage_2),
    trainers_epoch_threshold=(0, 10),
    device="cuda:0",
)
```

The first optimizer is used for epochs `0` through `9`; the second starts at epoch `10`.

---

# Configuration Reference Map

Use this table whenever a configuration field is not fully explained in this README.

| Topic                                                      | Authoritative code location                                                                                                    |
|------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| Project roots and project-relative paths                   | [config.py](config.py): module constants and `get_local_path`                                                                  |
| Dataset-relative paths                                     | [dataset_config.py](dataset_config.py): `get_dataset_path`                                                                     |
| Every core dataset field                                   | [load_data.py](load_data.py): `DatasetConfig` class docstring and constructor                                                  |
| Train/test source aliases in a dataset entry               | [load_data.py](load_data.py): `generate_datadset`                                                                              |
| DataLoader parameters and return forms                     | [load_data.py](load_data.py): `load_data`                                                                                      |
| Dataset scanning, pairing, and automatic download behavior | [load_data.py](load_data.py): `ImageDataset`                                                                                   |
| Pretrained-model-relative paths                            | [model_config.py](model_config.py): `get_pretrained_model_path`                                                                |
| Result-model-relative paths                                | [model_config.py](model_config.py): `save_model` and `load_model`                                                              |
| Every model registry field                                 | [model_config.py](model_config.py): comments above `MODEL_CONFIG`; [load_model.py](load_model.py): `generate_pretrained_model` |
| Pretrained-model download fields                           | [load_model.py](load_model.py): `download_pretrained_model`                                                                    |
| Registry-based model loading                               | [load_model.py](load_model.py): `load_pretrained_model`                                                                        |
| Training and early stopping                                | [train.py](train.py): `train_epoch` and `train`                                                                                |
| Numeric metrics                                            | [metric.py](metric.py): `ClassificationEvaluate`                                                                               |
| Evaluation figures and prediction visualization            | [metric.py](metric.py): `ShowEvaluateResult`                                                                                   |
| Reusable callbacks and custom loader helpers               | [_function.py](_function.py)                                                                                                   |

---
## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE.txt) file for details.
