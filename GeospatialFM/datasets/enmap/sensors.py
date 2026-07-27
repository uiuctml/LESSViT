import collections.abc
import json
import logging
import os
import random

import numpy as np
import yaml

from GeospatialFM.data_process.srf import nonzero_support_counts, resample_cube

from .enmap import S2C_WV, SpectralEarthDataset

logger = logging.getLogger(__name__)

_SENSORS_YAML = os.path.join(os.path.dirname(__file__), "sensors.yaml")
_DEFAULT_CACHE_DIR = os.path.join("results", "srf_stats")

# Safety margin for `enmap_identity`'s FWHM, as a fraction of the nearest-neighbor band
# spacing. At this ratio sigma = (0.4*d)/2.355 = 0.17d, so the nearest neighbor (distance d)
# gets Gaussian weight exp(-0.5*(d/sigma)**2) = exp(-17.3) ~= 3e-8, comfortably below
# build_srf_matrix's 1e-3 zero threshold -- i.e. every row collapses to exactly its own band,
# giving a literal identity matrix rather than merely a "dominant diagonal" one.
_IDENTITY_FWHM_SAFETY = 0.4


def _enmap_identity_fwhm():
    """Per-band FWHM (nm) for EnMAP's own centres, tight enough that build_srf_matrix(S2C_WV,
    S2C_WV, fwhm) is an exact identity (every row's only nonzero entry is its own band).

    True per-band EnMAP FWHM isn't available in this repo, so this derives a narrow FWHM from
    S2C_WV's own sampling density instead of real calibration data -- documented approximation,
    used only for the `enmap_identity` sanity check. Uses the smaller of the two neighboring
    gaps (not mean/median) so bands adjacent to a wide spectral gap (the ~1283->1530nm and
    ~1760->1939nm EnMAP water-vapour gaps) don't get an inflated FWHM from bridging across it.
    """
    wv = np.asarray(S2C_WV, dtype=np.float64)
    diffs = np.diff(wv)
    fwhm = np.empty_like(wv)
    for i in range(len(wv)):
        neighbors = []
        if i > 0:
            neighbors.append(diffs[i - 1])
        if i < len(wv) - 1:
            neighbors.append(diffs[i])
        fwhm[i] = _IDENTITY_FWHM_SAFETY * min(neighbors)
    return fwhm


def _drop_unbuildable_bands(sensor_label, lam_tgt, fwhm_tgt, band_names=None):
    """Drop nominal target bands with no EnMAP source support (EnMAP's water-vapour gaps, or a
    nominal upper/lower edge sitting past EnMAP's actual coverage) -- logged clearly rather than
    silently produced as NaNs, and rather than letting build_srf_matrix's hard zero-row error
    abort the whole sensor. `band_names`, if given, lets the log name the dropped band(s)
    (e.g. Sentinel-2's "B10") instead of only their wavelength.
    """
    lam_tgt = np.asarray(lam_tgt, dtype=np.float64)
    fwhm_tgt = np.asarray(fwhm_tgt, dtype=np.float64)
    counts = nonzero_support_counts(S2C_WV, lam_tgt, fwhm_tgt)
    valid = counts > 0
    if not valid.all():
        if band_names is not None:
            dropped_desc = [f"{band_names[j]} ({lam_tgt[j]:.1f}nm)" for j in np.where(~valid)[0]]
        else:
            dropped_desc = [f"{v:.1f}nm" for v in lam_tgt[~valid]]
        logger.warning(
            "%s: dropping %d/%d nominal band(s) with no EnMAP source support (EnMAP "
            "water-vapour gaps ~1283-1530nm/~1760-1939nm, and/or beyond EnMAP's %.2fnm max "
            "wavelength): %s",
            sensor_label, (~valid).sum(), len(lam_tgt), max(S2C_WV), ", ".join(dropped_desc),
        )
    return lam_tgt[valid].tolist(), fwhm_tgt[valid].tolist()


def _build_prisma_like(prisma_cfg):
    lam_lo, lam_hi = prisma_cfg["wavelength_range_nm"]
    spacing = prisma_cfg["spacing_nm"]
    fwhm = float(prisma_cfg["fwhm_nm"])
    lam_tgt = np.arange(lam_lo, lam_hi + 1e-6, spacing)
    fwhm_tgt = np.full_like(lam_tgt, fwhm)
    return _drop_unbuildable_bands("prisma_like", lam_tgt, fwhm_tgt)


def _build_sentinel2_like(s2_bands):
    lam_tgt = [float(b["centre_nm"]) for b in s2_bands]
    fwhm_tgt = [float(b["fwhm_nm"]) for b in s2_bands]
    names = [b.get("name", f"{c:.0f}nm") for b, c in zip(s2_bands, lam_tgt)]
    return _drop_unbuildable_bands("sentinel2_like", lam_tgt, fwhm_tgt, band_names=names)


def _build_eo1h_like(cfg):
    # Deferred import: eo1h_cdl.py isn't otherwise a dependency of sensors.py, and importing it
    # eagerly at module level would risk a circular import via enmap/__init__.py's own import
    # order (enmap_cdl.py, which imports sensors.py, is imported before eo1h_cdl.py there).
    # Safe here because _build_registry() only runs lazily, on first actual SENSOR_CONFIGS access
    # (see _LazySensorRegistry) -- by then every enmap dataset module is already imported.
    from .eo1h_cdl import EO1_WV
    lam_tgt = [float(w) for w in EO1_WV]
    fwhm_tgt = [float(cfg["fwhm_nm"])] * len(lam_tgt)
    return _drop_unbuildable_bands("eo1h_like", lam_tgt, fwhm_tgt)


def _build_desis_like(cfg):
    from .desis_cdl import DESIS_WV
    lam_tgt = [float(w) for w in DESIS_WV]
    fwhm_tgt = [float(cfg["fwhm_nm"])] * len(lam_tgt)
    return _drop_unbuildable_bands("desis_like", lam_tgt, fwhm_tgt)


def _build_registry():
    with open(_SENSORS_YAML, "r") as f:
        raw = yaml.safe_load(f)

    registry = {
        "enmap_identity": {
            "name": "enmap_identity",
            "lam_tgt": [float(w) for w in S2C_WV],
            "fwhm_tgt": _enmap_identity_fwhm().tolist(),
        }
    }

    prisma_lam_tgt, prisma_fwhm_tgt = _build_prisma_like(raw["prisma_like"])
    registry["prisma_like"] = {
        "name": "prisma_like",
        "lam_tgt": prisma_lam_tgt,
        "fwhm_tgt": prisma_fwhm_tgt,
    }

    s2_lam_tgt, s2_fwhm_tgt = _build_sentinel2_like(raw["sentinel2_like"]["bands"])
    registry["sentinel2_like"] = {
        "name": "sentinel2_like",
        "lam_tgt": s2_lam_tgt,
        "fwhm_tgt": s2_fwhm_tgt,
    }

    eo1h_lam_tgt, eo1h_fwhm_tgt = _build_eo1h_like(raw["eo1h_like"])
    registry["eo1h_like"] = {
        "name": "eo1h_like",
        "lam_tgt": eo1h_lam_tgt,
        "fwhm_tgt": eo1h_fwhm_tgt,
    }

    desis_lam_tgt, desis_fwhm_tgt = _build_desis_like(raw["desis_like"])
    registry["desis_like"] = {
        "name": "desis_like",
        "lam_tgt": desis_lam_tgt,
        "fwhm_tgt": desis_fwhm_tgt,
    }

    return registry


class _LazySensorRegistry(collections.abc.Mapping):
    """Defers _build_registry() (and its band-dropping log warnings) until the first actual
    lookup, rather than running it at import time.

    Every EnMAP downstream dataset class does `from .sensors import SENSOR_CONFIGS` at module
    level (see e.g. enmap_bdforet.py), and `GeospatialFM/datasets/enmap/__init__.py` eagerly
    imports all of those dataset classes -- so anything that imports GeospatialFM.datasets.enmap
    at all (including pretraining's scripts/train.py, which never touches gen_task or sensor
    configs) would otherwise trigger this module's full registry construction and logging on
    every import, unconditionally.
    """

    def __init__(self, builder):
        self._builder = builder
        self._registry = None

    def _ensure_built(self):
        if self._registry is None:
            self._registry = self._builder()
        return self._registry

    def __getitem__(self, key):
        return self._ensure_built()[key]

    def __iter__(self):
        return iter(self._ensure_built())

    def __len__(self):
        return len(self._ensure_built())

    def __contains__(self, key):
        return key in self._ensure_built()

    def __repr__(self):
        return repr(self._ensure_built())


SENSOR_CONFIGS = _LazySensorRegistry(_build_registry)


def list_sensor_names():
    """All registered SRF sensor names -- these are also valid `--gen_task` values."""
    return list(SENSOR_CONFIGS.keys())


def load_sensor_config(name_or_path):
    """Look up a sensor by name in SENSOR_CONFIGS, or load a standalone YAML/JSON file.

    Returns a dict with `name`, `lam_tgt` (nm), `fwhm_tgt` (nm).
    """
    if name_or_path in SENSOR_CONFIGS:
        return SENSOR_CONFIGS[name_or_path]

    with open(name_or_path, "r") as f:
        if str(name_or_path).endswith(".json"):
            cfg = json.load(f)
        else:
            cfg = yaml.safe_load(f)
    assert "lam_tgt" in cfg and "fwhm_tgt" in cfg, (
        f"Sensor config {name_or_path} must define lam_tgt and fwhm_tgt"
    )
    cfg.setdefault("name", os.path.splitext(os.path.basename(str(name_or_path)))[0])
    return cfg


def compute_target_stats(
    sensor_name, W, mu_src, lam_tgt=None, data_root=None, cache_dir=None,
    n_patches=2000, force=False, seed=0,
):
    """Return (mean, std) for a sensor's resampled bands.

    `mean` is always the closed-form `W @ mu_src` (resampling is linear on raw values, so this
    is exact). `std` is looked up from `{cache_dir}/{sensor_name}.json`; if the cache is missing
    (or `force=True`), it is computed -- never silently skipped -- by resampling `n_patches`
    random raw patches from `SpectralEarthDataset(root=data_root)` through `W` and accumulating
    per-band statistics over every pixel of every sampled patch.
    """
    cache_dir = cache_dir or _DEFAULT_CACHE_DIR
    cache_path = os.path.join(cache_dir, f"{sensor_name}.json")
    mu_src = np.asarray(mu_src, dtype=np.float64)
    mean = W @ mu_src

    if not force and os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            cached = json.load(f)
        std = np.asarray(cached["std"], dtype=np.float64)
        assert std.shape == mean.shape, (
            f"Cached std at {cache_path} has {std.shape[0]} bands, expected {mean.shape[0]} "
            f"for sensor '{sensor_name}' -- delete the stale cache file and recompute."
        )
        return mean, std

    if data_root is None:
        raise FileNotFoundError(
            f"No cached SRF stats at {cache_path} for sensor '{sensor_name}', and no data_root "
            "was given to compute them. Pass the EnMAP pretraining data_root so std can be "
            "computed, or precompute the cache separately."
        )

    logger.info(
        "Computing empirical std for sensor '%s' over up to %d patches from %s",
        sensor_name, n_patches, data_root,
    )
    dataset = SpectralEarthDataset(root=data_root)
    n_available = len(dataset)
    if n_available == 0:
        raise FileNotFoundError(
            f"SpectralEarthDataset at {data_root} has no patches to compute SRF stats from."
        )

    rng = random.Random(seed)
    indices = rng.sample(range(n_available), min(n_patches, n_available))

    C_tgt = W.shape[0]
    count = 0
    running_sum = np.zeros(C_tgt, dtype=np.float64)
    running_sumsq = np.zeros(C_tgt, dtype=np.float64)
    for idx in indices:
        raw = dataset[idx]["optical"]  # (202, H, W), raw values
        resampled = resample_cube(raw, W).numpy().astype(np.float64)  # (C_tgt, H, W)
        pixels = resampled.reshape(C_tgt, -1)
        running_sum += pixels.sum(axis=1)
        running_sumsq += (pixels ** 2).sum(axis=1)
        count += pixels.shape[1]

    empirical_mean = running_sum / count
    var = running_sumsq / count - empirical_mean ** 2
    std = np.sqrt(np.clip(var, a_min=0.0, a_max=None))

    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(
            {
                "sensor_name": sensor_name,
                "lam_tgt": lam_tgt,
                "mean": mean.tolist(),
                "std": std.tolist(),
                "n_patches": len(indices),
                "n_pixels": int(count),
                "data_root": data_root,
            },
            f,
        )
    logger.info("Cached SRF std for sensor '%s' to %s", sensor_name, cache_path)

    return mean, std
