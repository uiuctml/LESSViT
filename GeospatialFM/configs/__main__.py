"""Generate JSON without starting training or downloading weights/data."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=['model', 'sensor', 'pretrain', 'finetune'])
    parser.add_argument('--name', help='Registry model or sensor name')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--overrides', type=Path, help='JSON model constructor overrides')
    args, remaining = parser.parse_known_args()
    if remaining[:1] == ['--']:
        remaining = remaining[1:]
    if args.kind in ('model', 'sensor') and remaining:
        parser.error('Unexpected arguments: ' + ' '.join(remaining))
    if args.overrides and args.kind != 'model':
        parser.error('--overrides is only supported for model configs')
    if args.kind in ('model', 'sensor') and not args.name:
        parser.error('--name is required for model and sensor configs')
    if args.kind == 'model':
        from GeospatialFM.models.registry import ENCODER_CONFIGS
        if args.name not in ENCODER_CONFIGS:
            parser.error('Available models: ' + ', '.join(ENCODER_CONFIGS))
        overrides = json.loads(args.overrides.read_text()) if args.overrides else {}
        result = ENCODER_CONFIGS[args.name](**overrides).to_dict()
    elif args.kind == 'sensor':
        from GeospatialFM.datasets.enmap.sensors import load_sensor_config
        result = load_sensor_config(args.name)
    elif args.kind == 'pretrain':
        from GeospatialFM.scripts.args import parse_args
        result = vars(parse_args(remaining))
    else:
        from GeospatialFM.finetune.args import parse_args
        result = vars(parse_args(remaining))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Never overwrite an existing experiment configuration silently.
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
