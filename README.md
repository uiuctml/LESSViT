<div align="center">

<h1>LESSViT: Robust Hyperspectral Representation Learning under Spectral Configuration Shift</h1>

[![arXiv](https://img.shields.io/badge/arXiv-2503.12843-red?logo=arxiv)](https://arxiv.org/abs/2605.18541)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://uiuctml.github.io/LESSViT/)
[![GitHub](https://img.shields.io/badge/GitHub-LESSViT-green?logo=github)](https://github.com/uiuctml/LESSViT)

</div>

This is the official repository for the paper
"_LESSViT: Robust Hyperspectral Representation Learning under Spectral Configuration Shift_".

Authors:
[Haozhe Si](https://ehzoahis.github.io/),
Yuxuan Wan,
Yuqing Wang,
[Minh Do](https://minhdo.ece.illinois.edu/),
[Han Zhao](https://hanzhaoml.github.io/).

## Release scope

`main` contains configuration generation, all model implementations and checkpoint
loaders, and dataset construction/loading. The complete research workspace,
including training/evaluation launchers, sweeps, notebooks, figures, and project
website, is preserved on `dev` at `0c611f6`. To run the original end-to-end
training/evaluation workflows, use `dev`. This minimal branch does not include a
training runner or claim to reproduce published scores by itself.

## Installation

Use Python 3.11 in a fresh environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
```

`requirements.txt` records the reference dependency versions. Install PyTorch and
Torchvision wheels appropriate for your hardware. Hugging Face Hub is pinned
below 1.0 for Transformers 4.53.3; GFMBench requires `datasets<4` because its
loaders use dataset scripts. DINOv3's optional 2:4 sparse execution additionally
requires a compatible GPU installation of `xformers`; ordinary model loading
does not use it. Preserve upstream license notices in the bundled baseline code.

## Configuration generation

Commands run from the repository root, or anywhere after installation:

```bash
python -m GeospatialFM.configs --help
python -m GeospatialFM.configs model --name lessvit --output configs/lessvit.json
python -m GeospatialFM.configs sensor --name sentinel2_like --output configs/sentinel2.json
python -m GeospatialFM.configs pretrain --output configs/pretrain.json -- \
  --dataset_name enmap --data_dir /path/to/SpectralEarth --run_name example
python -m GeospatialFM.configs finetune --output configs/finetune.json -- \
  --dataset_name enmap_cdl --task_type segmentation \
  --data_dir /path/to/datasets --run_name example
```

Model configurations use the original constructor defaults, not a claim of
paper-specific hyperparameters. Supply `--overrides overrides.json` for model
constructor values. The pretrain/finetune modes preserve the original argument
schemas and accept their options after `--` (including `--help`). Existing output
files are never overwritten. Sensor wavelengths and FWHMs are in nanometers;
`sensors.yaml`, `sensors.py`, and `data_process/srf.py` retain the original sensor
construction and resampling algorithms.

## Models and checkpoint loading

All seven registered families are retained: `lessvit`, `specvit`, `dinov3`,
`dofa`, `spatsigma`, `channelvit`, and `hyperfree`. SpatSigma has classification
and segmentation encoders. SatMAE, LESSViT MAE, SpecViT MAE, downstream heads,
and their supporting modules are also retained.

```python
from GeospatialFM.models.registry import ENCODER_CONFIGS, ENCODER_MODELS

config = ENCODER_CONFIGS["channelvit"].from_json_file("configs/channelvit.json")
encoder = ENCODER_MODELS["channelvit"](config)
encoder.load_pretrained_weights("/path/to/channelvit/checkpoint_directory")
```

Generate `configs/channelvit.json` with the model command above first. Baseline
weight loaders accept a **directory** containing the original model's checkpoint
format; see `GeospatialFM/models/wrappers/` for filename and key conventions.
SpatSigma expects both `spat*.pth` and `spec*.pth`. Inputs and wavelength handling
are model-specific; consult each wrapper's `forward_encoder` signature.

For a downstream LESSViT model:

```python
from GeospatialFM.models.downstream_models import LESSWithProjection, LESSWithProjectionConfig

config = LESSWithProjectionConfig(num_labels=14, model_name="lessvit", task_type="classification")
model = LESSWithProjection(config)
model.load_pretrained_encoder("/path/to/lessvit/checkpoint_directory")
```

LESSViT's pretrained encoder loader reads `*.safetensors` containing `encoder.*`
keys and retains the legacy checkpoint key remapping. For a complete downstream
checkpoint, use `LESSWithProjection.from_pretrained(path)` or
`LESSWithUPerNet.from_pretrained(path)`. Construct downstream models from the
original argument namespace with `GeospatialFM.models.loading.get_task_model`.
Checkpoint architecture settings must match. Pretrained weights are not bundled.

## Dataset construction and loading

All original dataset implementations, metadata, split handling, transforms,
collators, quadrant tiling, and SRF statistics code are retained.

```python
from functools import partial
from torch.utils.data import DataLoader
from GeospatialFM.datasets.enmap import SpectralEarthDataset
from GeospatialFM.data_process import unimodal_collate_fn

# root/<patch_id>/*.tif; each source GeoTIFF has 202 channels.
dataset = SpectralEarthDataset("/path/to/SpectralEarth", subset=120)
loader = DataLoader(dataset, batch_size=2, collate_fn=partial(unimodal_collate_fn, modal="optical"))
batch = next(iter(loader))
```

This example loads raw values. Use `pretrain_transform` with matching subset
statistics and `get_enmap_transform` for normalization and downstream transforms.
The EnMAP downstream registry includes CDL, CORINE, EuroCrops, BD Foret, BNETD,
TreeMap, NLCD, DESIS-CDL, and EO-1-CDL:

```python
from types import SimpleNamespace
from GeospatialFM.datasets.enmap import get_enmap_downstream_dataset

args = SimpleNamespace(dataset_name="enmap_cdl", data_dir="/path/to/datasets")
splits = get_enmap_downstream_dataset(args, train_transform=None, eval_transform=None, gen_task="id")
sample = splits["test"][0]
```

For CDL, use `root/enmap_cdl/enmap/<sample>.tif`,
`root/enmap_cdl/cdl/<sample>.tif`, and
`root/splits/enmap_cdl/{train,val,test}.txt` listing relative sample filenames.
Other datasets retain their own layouts documented in their constructors.
`id`, `ood_a`, `ood_complement`, and `ood_full` retain the original spectral
selection semantics. SRF tasks additionally require the matching resampling
transform and target statistics; setting `gen_task` alone does not apply SRF
resampling. SSL4EO is available through `GeospatialFM.datasets.ssl4eo`;
GFMBench through `GeospatialFM.datasets.GFMBench.utils` (requires external data
access and execution of the upstream dataset scripts).

Datasets and checkpoints are not bundled. The previous machine-specific
`results` symlink has been removed; its external target was not modified.

## Validation

`tests/test_core.py` checks module imports, all registered model and sensor
configuration exports, original argument export, SRF normalization, a small
LESSViT checkpoint round trip, and synthetic GeoTIFF loading/collation at all
four supported pretraining channel counts. These are offline CPU checks, not
full training runs or validation of external pretrained baseline weights.
See [CLEANUP.md](CLEANUP.md) for the preservation and cleanup record.

## Citation

If you find our work helpful, please cite our paper:
```bibtex
@misc{si2026lessvitrobusthyperspectralrepresentation,
      title={LESSViT: Robust Hyperspectral Representation Learning under Spectral Configuration Shift}, 
      author={Haozhe Si and Yuxuan Wan and Yuqing Wang and Minh Do and Han Zhao},
      year={2026},
      eprint={2605.18541},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2605.18541}, 
}
```

## Contact

[Haozhe Si](mailto:haozhes3@illinois.edu), [Han Zhao](mailto:hanzhao@illinois.edu)
