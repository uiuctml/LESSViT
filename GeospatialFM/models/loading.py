"""Construct downstream models using the original experiment arguments."""
from GeospatialFM.models.downstream_models import LESSWithProjectionConfig, LESSWithUPerNetConfig, LESSWithProjection, LESSWithUPerNet

def get_task_model(args, num_classes=None, image_size=None):
    if args.task_type == "classification" or args.task_type == "multilabel":
        assert num_classes is not None
        config = LESSWithProjectionConfig(num_labels=num_classes, **vars(args))
        model = LESSWithProjection(config)
    elif args.task_type == "segmentation":
        assert num_classes is not None and image_size is not None
        config = LESSWithUPerNetConfig(num_labels=num_classes, image_size=image_size, **vars(args))
        model = LESSWithUPerNet(config)
    else:
        raise NotImplementedError
    return model

