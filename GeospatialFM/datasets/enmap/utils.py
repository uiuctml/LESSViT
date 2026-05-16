from .enmap import SpectralEarthDataset
from .enmap_bdforet import EnMAPBDForetDataset
from .enmap_corine import EnMAPCorineDataset
from .enmap_bnetd import EnMAPBNETDDataset
from .enmap_eurocrops import EnMAPEurocropsDataset
from .enmap_treemap import EnMAPTreemapDataset
from .enmap_cdl import EnMAPCDLDataset
from .enmap_nlcd import EnMAPNLCDataset
from .desis_cdl import DESISCDLDataset
from .eo1h_cdl import EO1CDLDataset

ENMAP_DATASET = {
    "enmap_bdforet": EnMAPBDForetDataset,
    "enmap_corine": EnMAPCorineDataset,
    "enmap_bnetd": EnMAPBNETDDataset,
    "enmap_eurocrops": EnMAPEurocropsDataset,
    "enmap_treemap": EnMAPTreemapDataset,
    "enmap_cdl": EnMAPCDLDataset,
    "enmap_nlcd": EnMAPNLCDataset,
    "desis_cdl": DESISCDLDataset,
    "eo1_cdl": EO1CDLDataset,
}

def get_enmap_metadata():
    return SpectralEarthDataset.metadata

def get_enmap_downstream_metadata(dataset_name, dataset_version=None):
    return ENMAP_DATASET[dataset_name].metadata

def get_enmap_downstream_dataset(args, train_transform, eval_transform, gen_task=None):
    dataset_dict = {}
    splits = ["train", "val", "test"]
    for split in splits:
        transform = train_transform if split == "train" else eval_transform
        dataset_dict[split] = ENMAP_DATASET[args.dataset_name](
            root=args.data_dir,
            split=split,
            transform=transform,
            gen_task=gen_task
        )

    return dataset_dict