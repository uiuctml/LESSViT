import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _raw_response_matrix(lam_src, lam_tgt, fwhm_tgt, zero_threshold=1e-3):
    """Thresholded (not row-normalized) Gaussian response matrix, shape (C_tgt, len(lam_src))."""
    lam_src = np.asarray(lam_src, dtype=np.float64)
    lam_tgt = np.asarray(lam_tgt, dtype=np.float64)
    fwhm_tgt = np.broadcast_to(np.asarray(fwhm_tgt, dtype=np.float64), lam_tgt.shape)

    sigma = fwhm_tgt / 2.355
    diff = lam_src[None, :] - lam_tgt[:, None]  # (C_tgt, C_src)
    W = np.exp(-0.5 * (diff / sigma[:, None]) ** 2)
    W[W < zero_threshold] = 0.0
    return W


def nonzero_support_counts(lam_src, lam_tgt, fwhm_tgt, zero_threshold=1e-3):
    """Number of source bands with nonzero (thresholded) weight for each target band."""
    W = _raw_response_matrix(lam_src, lam_tgt, fwhm_tgt, zero_threshold)
    return (W > 0).sum(axis=1)


def build_srf_matrix(lam_src, lam_tgt, fwhm_tgt, zero_threshold=1e-3, min_nonzero_warn=3):
    """Gaussian spectral response function matrix resampling `lam_src` bands onto `lam_tgt` bands.

    W[j, i] = exp(-0.5 * ((lam_src[i] - lam_tgt[j]) / sigma[j])**2), sigma = fwhm_tgt / 2.355.
    Entries below `zero_threshold` are zeroed before row-normalizing, so each row sums to 1.

    Args:
        lam_src: (C_src,) source band centre wavelengths, nm.
        lam_tgt: (C_tgt,) target band centre wavelengths, nm.
        fwhm_tgt: scalar or (C_tgt,) target band FWHMs, nm.

    Returns:
        W: (C_tgt, C_src) row-normalized response matrix.

    Raises:
        ValueError: if any target band has no source support (row sums to ~0 before
            normalization) -- e.g. a target centre falling in a source wavelength gap.
    """
    lam_tgt = np.asarray(lam_tgt, dtype=np.float64)
    W = _raw_response_matrix(lam_src, lam_tgt, fwhm_tgt, zero_threshold)

    row_sums = W.sum(axis=1)
    zero_rows = np.where(row_sums <= 0)[0]
    if len(zero_rows) > 0:
        offending = ", ".join(f"{lam_tgt[j]:.2f}nm" for j in zero_rows)
        raise ValueError(
            f"{len(zero_rows)} target band(s) have no source support (row sum ~0 before "
            f"normalization): {offending}. These likely fall in a source wavelength gap "
            "(e.g. EnMAP's water-vapour absorption gaps around ~1350-1450nm / ~1800-1950nm) "
            "-- check lam_tgt/fwhm_tgt for this sensor."
        )

    nonzero_counts = (W > 0).sum(axis=1)
    undersampled = np.where(nonzero_counts < min_nonzero_warn)[0]
    for j in undersampled:
        logger.warning(
            "SRF target band at %.2fnm has only %d nonzero source band(s) (< %d) -- "
            "quadrature may be too coarse to trust.",
            lam_tgt[j], nonzero_counts[j], min_nonzero_warn,
        )
    logger.debug("Nonzero source bands per target band: %s", nonzero_counts.tolist())

    W = W / row_sums[:, None]
    return W


def resample_cube(x_raw, W):
    """Resample a raw (C_src, H, W) cube to (C_tgt, H, W): x_out = einsum('cj,jhw->chw', W, x_raw)."""
    if not isinstance(x_raw, torch.Tensor):
        x_raw = torch.as_tensor(x_raw)
    if isinstance(W, torch.Tensor):
        W = W.to(dtype=x_raw.dtype, device=x_raw.device)
    else:
        W = torch.as_tensor(np.asarray(W), dtype=x_raw.dtype, device=x_raw.device)
    return torch.einsum("cj,jhw->chw", W, x_raw)


def summarize_srf(lam_tgt, W, sensor_name=None):
    """Print n_bands, min/median/max nonzero-source-bands-per-target-band, and wavelength range."""
    lam_tgt = np.asarray(lam_tgt, dtype=np.float64)
    nonzero_counts = (np.asarray(W) > 0).sum(axis=1)
    label = f" ({sensor_name})" if sensor_name else ""
    stats = {
        "n_bands": len(lam_tgt),
        "wavelength_min": float(lam_tgt.min()),
        "wavelength_max": float(lam_tgt.max()),
        "nonzero_min": int(nonzero_counts.min()),
        "nonzero_median": int(np.median(nonzero_counts)),
        "nonzero_max": int(nonzero_counts.max()),
    }
    print(
        f"SRF sensor{label}: n_bands={stats['n_bands']}, "
        f"wavelength range=[{stats['wavelength_min']:.1f}, {stats['wavelength_max']:.1f}]nm, "
        f"nonzero source bands/target band: min={stats['nonzero_min']}, "
        f"median={stats['nonzero_median']}, max={stats['nonzero_max']}"
    )
    return stats
