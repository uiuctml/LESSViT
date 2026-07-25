import time

import torch
from transformers import TrainerCallback


class ThroughputMemoryCallback(TrainerCallback):
    """
    Adds peak_mem_gb and tokens/sec to the existing per-step wandb/log_history
    output (train loss, wall clock, etc. already flow through MAETrainer/wandb
    unmodified). tokens_per_step is a fixed estimate (batch_size * num_processes *
    num_spatial_tokens * an assumed channel count), not read from the actual
    per-step HCS-sampled channel count -- HF Trainer callbacks aren't given the
    batch `inputs`, and `SpatialSpectralMAEViT.main_input_name` being a list
    already breaks the trainer's own built-in token counter (see
    MAETrainer.floating_point_ops's docstring for the same issue), so an exact
    per-step count isn't available without touching MAETrainer.compute_loss
    itself. Good enough for a throughput comparison across arms, which all see
    the same channel-count distribution.
    """

    def __init__(self, tokens_per_step: int):
        self.tokens_per_step = tokens_per_step
        self._step_start_time = None

    def on_step_begin(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._step_start_time = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or self._step_start_time is None:
            return
        step_time = time.time() - self._step_start_time
        if torch.cuda.is_available():
            logs["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
        if step_time > 0:
            logs["tokens_per_sec"] = self.tokens_per_step / step_time


def estimate_tokens_per_step(per_device_batch_size: int, num_processes: int, crop_size: int,
                              patch_size: int, base_channels: int, channel_dropout) -> int:
    """
    Rough, fixed per-step token estimate for throughput logging (see
    ThroughputMemoryCallback). Uses the midpoint of `channel_dropout`'s keep-ratio
    range as the assumed channel count under HCS, rather than the actual per-step
    random count -- consistent across all three arms, which is all that matters
    for comparing their relative throughput.
    """
    num_spatial_tokens = (crop_size // patch_size) ** 2 + 1  # +1 for the spatial CLS token
    if channel_dropout is not None:
        lo, hi = sorted(channel_dropout)
        keep_ratio = 1 - (lo + hi) / 2
        channels = max(1, int(base_channels * keep_ratio))
    else:
        channels = base_channels
    return per_device_batch_size * num_processes * num_spatial_tokens * channels
