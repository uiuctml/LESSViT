import os
import random
from typing import Optional

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset
from torchgeo.datasets.cdl import CDL

from .enmap import S2C_MEAN, S2C_STD, S2C_WV, SELECTED_CHANNEL_IDX_B, SELECTED_CHANNEL_IDX_A

DESIS_WV = [402.00, 404.55, 407.10, 409.65, 412.20, 414.75, 417.30, 419.85, 422.40, 424.95, 427.50, 430.05, 432.60, 435.15, 437.70, 440.25, 442.80, 445.35, 447.90, 450.45, 453.00, 455.55, 458.10, 460.65, 463.20, 465.75, 468.30, 470.85, 473.40, 475.95, 478.50, 481.05, 483.60, 486.15, 488.70, 491.25, 493.80, 496.35, 498.90, 501.45, 504.00, 506.55, 509.10, 511.65, 514.20, 516.75, 519.30, 521.85, 524.40, 526.95, 529.50, 532.05, 534.60, 537.15, 539.70, 542.25, 544.80, 547.35, 549.90, 552.45, 555.00, 557.55, 560.10, 562.65, 565.20, 567.75, 570.30, 572.85, 575.40, 577.95, 580.50, 583.05, 585.60, 588.15, 590.70, 593.25, 595.80, 598.35, 600.90, 603.45, 606.00, 608.55, 611.10, 613.65, 616.20, 618.75, 621.30, 623.85, 626.40, 628.95, 631.50, 634.05, 636.60, 639.15, 641.70, 644.25, 646.80, 649.35, 651.90, 654.45, 657.00, 659.55, 662.10, 664.65, 667.20, 669.75, 672.30, 674.85, 677.40, 679.95, 682.50, 685.05, 687.60, 690.15, 692.70, 695.25, 697.80, 700.35, 702.90, 705.45, 708.00, 710.55, 713.10, 715.65, 718.20, 720.75, 723.30, 725.85, 728.40, 730.95, 733.50, 736.05, 738.60, 741.15, 743.70, 746.25, 748.80, 751.35, 753.90, 756.45, 759.00, 761.55, 764.10, 766.65, 769.20, 771.75, 774.30, 776.85, 779.40, 781.95, 784.50, 787.05, 789.60, 792.15, 794.70, 797.25, 799.80, 802.35, 804.90, 807.45, 810.00, 812.55, 815.10, 817.65, 820.20, 822.75, 825.30, 827.85, 830.40, 832.95, 835.50, 838.05, 840.60, 843.15, 845.70, 848.25, 850.80, 853.35, 855.90, 858.45, 861.00, 863.55, 866.10, 868.65, 871.20, 873.75, 876.30, 878.85, 881.40, 883.95, 886.50, 889.05, 891.60, 894.15, 896.70, 899.25, 901.80, 904.35, 906.90, 909.45, 912.00, 914.55, 917.10, 919.65, 922.20, 924.75, 927.30, 929.85, 932.40, 934.95, 937.50, 940.05, 942.60, 945.15, 947.70, 950.25, 952.80, 955.35, 957.90, 960.45, 963.00, 965.55, 968.10, 970.65, 973.20, 975.75, 978.30, 980.85, 983.40, 985.95, 988.50, 991.05, 993.60, 996.15, 998.70]

# Recomputed 2026-07-27 from the full desis_cdl/desis corpus on disk (1000 tiles,
# /datasets/geospatial/desis_cdl/desis/*/*.tif), streaming per-channel sum/sumsq -- the
# previous constants here differed from this by up to ~9%/7.6% (mean/std) per channel,
# apparently computed from a different sample. (eo1h_cdl.py's EO1_MEAN/EO1_STD were
# checked the same way and matched the full 550-tile corpus exactly -- untouched.)
DESIS_MEAN = [647.211, 1120.68, 1172.17, 1055.81, 971.862, 893.899, 827.453, 760.466, 732.841, 722.206, 694.549, 696.206, 711.728, 691.874, 673.356, 670.92, 655.416, 660.415, 658.428, 643.835, 651.079, 651.703, 653.954, 658.162, 654.956, 660.818, 662.774, 673.496, 675.699, 685.849, 687.211, 686.969, 679.573, 700.661, 700.74, 697.95, 704.731, 705.907, 713.506, 722.262, 727.723, 737.175, 752.587, 767.473, 785.22, 814.324, 846.036, 868.334, 882.564, 913.134, 921.736, 942.512, 960.372, 977.501, 985.739, 1000.55, 1009.69, 1023.11, 1037.01, 1049.25, 1055.31, 1061.59, 1064.89, 1060.99, 1063.94, 1054.47, 1055.53, 1045.2, 1046.21, 1046.96, 1049.84, 1045.15, 1039.85, 1011.76, 1008.7, 1017.16, 1031.17, 1048.64, 1064.96, 1070.6, 1074.78, 1078, 1078.59, 1077.76, 1083, 1079.44, 1084.17, 1086.68, 1085.28, 1092.99, 1100.11, 1105.72, 1106.5, 1105.13, 1105.76, 1098.05, 1073.12, 1082.64, 1086.2, 1090.61, 1107.09, 1107.43, 1110.42, 1112.03, 1111.57, 1115.75, 1122.33, 1130.25, 1133.14, 1141.22, 1157.18, 1161.82, 1182.84, 1219.48, 1215.43, 1276.94, 1392.92, 1455.91, 1540.01, 1620.66, 1698.95, 1786.65, 1890.35, 1974.96, 2086.18, 2162.24, 2274.8, 2372.29, 2481.38, 2583.51, 2646.56, 2738.62, 2814.67, 2883.93, 2926.67, 2984.67, 3029.89, 3071.11, 3124.32, 3100.6, 3083.93, 3039.1, 3058.4, 3178.89, 3251.73, 3247.64, 3254.38, 3265.32, 3278.3, 3278.14, 3263.53, 3224.33, 3216.34, 3258.83, 3239.05, 3353.05, 3362.22, 3377.21, 3377.32, 3384.96, 3399.14, 3410.82, 3429.42, 3423.48, 3414.75, 3457.66, 3425.74, 3455.5, 3450.21, 3458.51, 3459.09, 3472.24, 3471.49, 3414.52, 3434.58, 3455.2, 3447.6, 3526.22, 3473.24, 3494.47, 3497.4, 3527.12, 3558.51, 3524.04, 3522.76, 3528.31, 3514.72, 3524.98, 3528.22, 3521.28, 3560.3, 3552.78, 3560.61, 3596.03, 3580.49, 3613.86, 3611.19, 3581.46, 3620.22, 3590.54, 3623.76, 3624.67, 3631.93, 3621.26, 3634.08, 3619.82, 3673.4, 3700.08, 3746.4, 3734.08, 3642.71, 3643.08, 3645.4, 3623.49, 3598.01, 3595.94, 3561.01, 3533.94, 3543.4, 3497.98, 3483.92, 3470.21, 3434.42, 3442.75, 3450.4, 3438.87, 3450.11, 3451.44, 3461.1, 3466.94, 3480.65, 3520.92, 3541.81, 3549.95, 3502.39]

DESIS_STD = [2275.65, 1492.57, 1042.6, 917.007, 857.989, 822.97, 796.353, 774.391, 746.808, 723.847, 698.828, 694.161, 669.952, 660.572, 652.664, 650.346, 642.656, 640.412, 633.753, 626.698, 626.284, 619.792, 614.298, 611.853, 606.286, 604.527, 600.621, 601.091, 597.501, 597.805, 596.646, 595.885, 592.391, 600.88, 594.96, 593.715, 596.733, 595.159, 596.742, 596.937, 594.504, 592.167, 591.415, 588.461, 586.824, 586.124, 580.717, 577.487, 572.998, 573.812, 566.168, 566.046, 565.156, 565.921, 564.983, 566.735, 566.146, 568.379, 569.668, 572.759, 574.777, 580.214, 584.875, 588.027, 596.629, 601.658, 612.387, 616.741, 625.817, 632.982, 638.774, 640.992, 645.68, 640.74, 642.134, 646.355, 651.869, 659.695, 667.429, 671.502, 678.695, 685.364, 689.487, 693.85, 700.396, 701.388, 708.736, 712.083, 711.347, 715.825, 718.529, 721.701, 724.385, 728.522, 733.619, 735.036, 731.003, 738.636, 743.372, 751.535, 760.377, 763.208, 771.689, 779.115, 782.433, 787.782, 791.264, 794.652, 797.05, 796.844, 801.126, 798.715, 796.4, 782.693, 752.852, 737.615, 740.474, 718.609, 697.224, 676.428, 656.982, 637.102, 625.489, 626.966, 642.726, 624.789, 646.733, 677.149, 716.686, 755.47, 781.205, 827.028, 870.914, 908.135, 934.248, 964.816, 992.156, 1014.94, 1029.63, 986.794, 1025.53, 1055.51, 1086.84, 1088.71, 1090.41, 1081.51, 1080.9, 1084.81, 1084.68, 1079.46, 1071.81, 1066.42, 1067.58, 1073.21, 1063.55, 1086.66, 1089.08, 1095.44, 1087.42, 1082.23, 1079.16, 1082.62, 1094.34, 1093.25, 1077.01, 1100.01, 1078.37, 1088.32, 1089.19, 1091.15, 1089.27, 1090.22, 1083.21, 1067.08, 1067.17, 1075.95, 1071.86, 1100.48, 1073.73, 1076.85, 1072.99, 1083.35, 1095.34, 1081.32, 1074.44, 1072.48, 1066.59, 1069.73, 1070.86, 1066.09, 1074.41, 1064.95, 1060.78, 1067.57, 1058.06, 1069, 1070.56, 1055.58, 1063.13, 1046.46, 1054.43, 1053.63, 1054.42, 1051.01, 1052.93, 1035.64, 1036.97, 1042.12, 1193.63, 1122.66, 1031.63, 1002.89, 1007.99, 994.522, 971.473, 963.864, 944.187, 927.846, 922.07, 909.269, 903.377, 896.827, 883.584, 877.001, 880.363, 875.529, 875.982, 875.056, 873.627, 869.047, 870.972, 880.348, 886.578, 888.847, 904.752]


class DESISCDLDataset(Dataset):
    """PyTorch dataset for DESIS-CDL samples."""

    classes = [0, 1, 2, 3, 5, 42, 43, 49, 54, 56, 68, 69, 75, 76, 204]
    ignore_index = len(classes) - 1
    num_classes = len(classes) - 1  # excluding ignore_index 

    spatial_resolution = 30
    metadata = {
        "s2c": {
            "bands": None,
            "channel_wv": DESIS_WV,
            "mean": DESIS_MEAN,
            "std": DESIS_STD,
        },
        "s1": {
            "bands": None,
            "channel_wv": None,
            "mean": None,
            "std": None,
        },
        "num_classes": num_classes,
        "ignore_index": ignore_index,
    }

    image_root = "desis"
    mask_root = "cdl"

    def __init__(self, root: str, split: str, transform, gen_task: Optional[str] = None) -> None:
        """
        Args:
            root: Root directory containing the dataset.
            split: Optional split subdirectory inside ``root``.
            transform: Optional transform to be applied on a sample.
        """
        self.root = os.path.join(root, "desis_cdl")
        self.split_file = os.path.join(root, "splits", "desis_cdl", f"{split}.txt")
        self.split = split
        self.transform = transform
        self.gen_task = gen_task
        if not os.path.isdir(self.root):
            raise FileNotFoundError(f"Dataset directory not found: {self.root}")
        
        self.ordinal_map = torch.zeros(max(CDL.cmap.keys()) + 1, dtype=torch.long) + len(self.classes) - 1
        self.ordinal_cmap = torch.zeros((len(self.classes), 4), dtype=torch.uint8)
        self.classes.remove(0)  
        self.classes.append(0)
        for v, k in enumerate(self.classes):
            self.ordinal_map[k] = v
            self.ordinal_cmap[v] = torch.tensor(CDL.cmap[k])

        if os.path.exists(self.split_file):
            self.sample_collection = self.read_split_file()
        else:
            raise ValueError(f"Split file not found: {self.split_file}")

        # print(f"ignore_index: {self.ignore_index}")

    def read_split_file(self):
        with open(self.split_file, "r") as f:
            sample_ids = [x.strip() for x in f.readlines()]
        sample_collection = [
            (
                os.path.join(self.root, self.image_root, sample_id),
                os.path.join(self.root, self.mask_root, sample_id)
            )
            for sample_id in sample_ids
        ]
        return sample_collection

    def __getitem__(self, index: int) -> dict[str, object]:
        img_path, mask_path = self.sample_collection[index]
        with rasterio.open(img_path) as src:
            optical = torch.from_numpy(src.read()).float()
        
        with rasterio.open(mask_path) as src:
            mask = torch.from_numpy(src.read()).long().squeeze(0)  # shape: (H, W)
            mask = self.ordinal_map[mask]  # remap to ordinal labels

        if self.transform is not None:
            optical, _, mask, spatial_resolution = self.transform(
                optical=optical,
                radar=None,
                label=mask,
                spatial_resolution=self.spatial_resolution
            )
        
        optical_channel_wv = self.metadata["s2c"]["channel_wv"]
        if self.gen_task is not None:
            if self.split == "train":
                optical = optical[SELECTED_CHANNEL_IDX_B, :, :]
                optical_channel_wv = [optical_channel_wv[i] for i in SELECTED_CHANNEL_IDX_B]
            else:
                # val and test set
                if self.gen_task == "id":
                    optical = optical[SELECTED_CHANNEL_IDX_B, :, :]
                    optical_channel_wv = [optical_channel_wv[i] for i in SELECTED_CHANNEL_IDX_B]
                elif self.gen_task == "ood_a":
                    optical = optical[SELECTED_CHANNEL_IDX_A, :, :]
                    optical_channel_wv = [optical_channel_wv[i] for i in SELECTED_CHANNEL_IDX_A]
                elif self.gen_task == "ood_full":
                    pass
                elif self.gen_task == "ood_complement":
                    optical = optical[[i for i in range(optical.shape[0]) if i not in SELECTED_CHANNEL_IDX_B], :, :]
                    optical_channel_wv = [optical_channel_wv[i] for i in range(202) if i not in SELECTED_CHANNEL_IDX_B]
                else:
                    raise ValueError(f"Invalid gen_task: {self.gen_task}")


        return {
            "optical": optical,
            "radar": None,
            "optical_channel_wv": optical_channel_wv,
            "radar_channel_wv": None,
            "spatial_resolution": spatial_resolution,
            "label": mask,
        }

    def __len__(self) -> int:
        return len(self.sample_collection)
