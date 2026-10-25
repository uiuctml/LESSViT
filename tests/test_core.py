"""Offline release checks: no published weights or datasets are required."""
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import rasterio
import torch
from rasterio.transform import from_origin


class CoreTests(unittest.TestCase):
    def test_imports(self):
        root = Path(__file__).resolve().parents[1]
        for path in sorted((root / 'GeospatialFM').rglob('*.py')):
            if path.name != '__main__.py':
                importlib.import_module('.'.join(path.relative_to(root).with_suffix('').parts))

    def test_configuration_exports(self):
        from GeospatialFM.configs.__main__ import main
        from GeospatialFM.models.registry import ENCODER_CONFIGS
        from GeospatialFM.datasets.enmap.sensors import list_sensor_names, load_sensor_config
        from GeospatialFM.datasets.enmap.enmap import S2C_WV
        from GeospatialFM.data_process.srf import build_srf_matrix, resample_cube
        with tempfile.TemporaryDirectory() as tmp:
            for kind, names in [('model', ENCODER_CONFIGS), ('sensor', list_sensor_names())]:
                for name in names:
                    target = Path(tmp) / f'{kind}-{name}.json'
                    with patch('sys.argv', ['configs', kind, '--name', name, '--output', str(target)]):
                        main()
                    payload = json.loads(target.read_text())
                    if kind == 'model':
                        cfg = ENCODER_CONFIGS[name].from_dict(payload)
                        self.assertEqual(cfg.model_type, payload['model_type'])
                    else:
                        self.assertEqual(payload, load_sensor_config(name))
                        W = build_srf_matrix(S2C_WV, payload['lam_tgt'], payload['fwhm_tgt'], min_nonzero_warn=0)
                        np.testing.assert_allclose(W.sum(axis=1), 1)
                        cube = resample_cube(torch.ones(202, 4, 4), W)
                        torch.testing.assert_close(cube, torch.ones(len(payload['lam_tgt']), 4, 4))
            for kind, extra in [('pretrain', []), ('finetune', ['--task_type', 'classification'])]:
                target = Path(tmp) / f'{kind}.json'
                argv = ['configs', kind, '--output', str(target), '--', '--dataset_name', 'enmap',
                        '--data_dir', '/example/data', '--run_name', 'smoke'] + extra
                with patch('sys.argv', argv):
                    main()
                self.assertEqual(json.loads(target.read_text())['run_name'], 'smoke')
                with patch('sys.argv', argv), self.assertRaises(FileExistsError):
                    main()

    def test_model_checkpoint_roundtrip(self):
        from GeospatialFM.models.downstream_models import LESSWithProjection, LESSWithProjectionConfig
        config = LESSWithProjectionConfig(embed_dim=48, depth=1, num_heads=3,
                                          channel_embed_dims_per_head=4, num_labels=2,
                                          model_name='lessvit', task_type='classification')
        model = LESSWithProjection(config)
        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp)
            restored = LESSWithProjection.from_pretrained(tmp)
            restored.load_pretrained_encoder(tmp)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, restored.state_dict()[key])

    def test_geotiff_dataset(self):
        from GeospatialFM.datasets.enmap import SpectralEarthDataset
        from GeospatialFM.data_process.collate_func import unimodal_collate_fn
        from torch.utils.data import DataLoader
        with tempfile.TemporaryDirectory() as tmp:
            patch_dir = Path(tmp) / 'patch_0'
            patch_dir.mkdir()
            with rasterio.open(patch_dir / 'sample.tif', 'w', driver='GTiff', height=16, width=16,
                               count=202, dtype='float32', transform=from_origin(100, 100, 30, 30)) as dst:
                dst.write(np.ones((202, 16, 16), dtype=np.float32))
            for subset, channels in [(None, 202), (120, 120), (60, 60), (10, 10)]:
                dataset = SpectralEarthDataset(tmp, subset=subset)
                self.assertEqual(len(dataset), 1)
                sample = dataset[0]
                self.assertEqual(tuple(sample['optical'].shape), (channels, 16, 16))
                self.assertEqual(len(sample['optical_channel_wv']), channels)
                batch = next(iter(DataLoader(dataset, batch_size=1, collate_fn=unimodal_collate_fn)))
                self.assertTrue(batch)


if __name__ == '__main__':
    unittest.main()
