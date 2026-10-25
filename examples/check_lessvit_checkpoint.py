"""Validate a LESSViT MAE encoder on synthetic downstream tasks, offline."""
import argparse
import json
from pathlib import Path
import re

import torch
from safetensors import safe_open

from GeospatialFM.models.downstream_models import (
    LESSWithProjection, LESSWithProjectionConfig,
    LESSWithUPerNet, LESSWithUPerNetConfig,
)


def check(checkpoint_dir, config_path, task, num_labels=14):
    config_dict = json.loads(config_path.read_text())
    # Preserve checkpoint architecture, but disable pretraining augmentation.
    config_dict.update(model_name='lessvit', task_type=task, num_labels=num_labels,
                       channel_dropout=None, return_dict=True)
    config_dict.pop('model_type', None)
    config_dict.pop('architectures', None)
    if task == 'segmentation':
        config_dict['image_size'] = 32
        model = LESSWithUPerNet(LESSWithUPerNetConfig(**config_dict))
    else:
        model = LESSWithProjection(LESSWithProjectionConfig(**config_dict))
    model.load_pretrained_encoder(str(checkpoint_dir))

    # Verify every parameter was actually copied, including legacy key remapping.
    state = model.state_dict()
    loaded = set()
    remapped = 0
    checkpoint = sorted(checkpoint_dir.glob('*.safetensors'))[-1]
    legacy = re.compile(r'^(encoder\.blocks\.\d+\.)(low_dim_pool\.|channel_norm\.|spatial_norm\.|attn\.)')
    with safe_open(checkpoint, framework='pt', device='cpu') as source:
        for key in source.keys():
            if not key.startswith('encoder.') or key == 'encoder.perception_field_mask':
                continue
            target = key
            if target not in state:
                match = legacy.match(key)
                if match:
                    target = match.group(1) + 'attn.' + match.group(2) + key[match.end():]
                    remapped += 1
            torch.testing.assert_close(state[target], source.get_tensor(key), rtol=0, atol=0)
            loaded.add(target)
    expected = {'encoder.' + key for key, _ in model.encoder.named_parameters()}
    missing = expected - loaded
    if missing:
        raise RuntimeError(f'Encoder parameters not loaded: {sorted(missing)}')

    # Head-only gradient check: pretrained encoder frozen, task head trainable.
    model.encoder.requires_grad_(False)
    model.eval()
    from GeospatialFM.datasets.enmap.enmap import S2C_WV, SELECTED_CHANNEL_IDX_B_10
    wavelengths = torch.tensor([[S2C_WV[i] for i in SELECTED_CHANNEL_IDX_B_10]])
    wavelengths = (wavelengths - min(S2C_WV)) / (max(S2C_WV) - min(S2C_WV))
    inputs = torch.randn(1, 10, 32, 32)
    logits = model(optical=inputs, optical_channel_wv=wavelengths, spatial_resolution=30)['logits']
    expected_shape = (1, num_labels, 32, 32) if task == 'segmentation' else (1, num_labels)
    assert tuple(logits.shape) == expected_shape, (logits.shape, expected_shape)
    assert torch.isfinite(logits).all()
    if task == 'multilabel':
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits))
    else:
        labels = torch.zeros((1, 32, 32) if task == 'segmentation' else (1,), dtype=torch.long)
        loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()
    head_grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert head_grads and all(g is not None and torch.isfinite(g).all() for g in head_grads)
    return dict(task=task, encoder_parameters=len(expected), verified_tensors=len(loaded),
                remapped_tensors=remapped, logits_shape=list(logits.shape), loss=float(loss.detach()),
                head_backward='passed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True, help='Original pretraining config.json')
    parser.add_argument('--task', choices=['classification', 'multilabel', 'segmentation', 'all'], default='all')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    tasks = ['classification', 'multilabel', 'segmentation'] if args.task == 'all' else [args.task]
    for task in tasks:
        print(json.dumps(check(args.checkpoint_dir, args.config, task)), flush=True)


if __name__ == '__main__':
    main()
