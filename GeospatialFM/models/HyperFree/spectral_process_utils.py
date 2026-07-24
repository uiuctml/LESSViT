"""
Ported from https://github.com/Jingtao-Li-CVer/HyperFree (HyperFree/utils/spectral_process_utils.py)

The gdal-based `read_img`/`write_img` I/O helpers from the original file are intentionally
omitted here: they are only used by the upstream demo scripts, not by the image encoder, and
dropping them avoids adding a GDAL dependency to this repo.
"""
import random

import numpy as np

# Preset wavelengths of key bands
spectral_wavelength = [400, 412.5, 429.5, 443, 455, 467.5, 473.375, 481.25, 488.25,
                       500, 520, 531, 536, 545, 550.5, 561.25, 564.75, 565.5, 575, 580,
                       596, 605, 610, 612, 626, 627.5, 630, 635, 640, 645, 650, 655, 656,
                       660, 664.5, 665, 667, 671.25, 677.5, 686, 700, 705, 710, 716, 725,
                       730, 740, 748.5, 760, 764.25, 776, 783, 790, 808, 820, 825, 830,
                       835.3125, 842, 850, 858.5, 865, 866, 869.5, 880, 896, 905, 910, 926,
                       938, 945, 950, 959, 1240, 1375, 1575, 1575.5, 1610, 1640, 1650, 2050.25,
                       2130, 2195, 2217.5, 2500]

# Preset wavelength indices for weight dictionary from 400-2500
weight_bank_wavelength = np.arange(400, 2510, 10).tolist()

# Wavelengths for training AVIRIS hyperspectral data
input_wavelengths_hy = [404.6129, 414.2946, 423.9808, 433.6713, 443.3662, 453.0655,
        462.7692, 472.4773, 482.1898, 491.9066, 501.6279, 511.3535, 521.0836, 530.818, 540.5568, 550.3,
        560.0477, 569.7996, 579.556, 589.3168, 599.0819, 608.8515, 618.6254, 628.4037, 638.1865, 647.9736,
        657.7651, 667.561, 655.2923, 665.0994, 674.9012, 684.6979, 694.4894, 704.2756, 714.0566, 723.8325,
        733.6031, 743.3685, 753.1287, 762.8837, 772.6335, 782.3781, 792.1174, 801.8516, 811.5805, 821.3043,
        831.0228, 840.7361, 850.4442, 860.1471, 869.8448, 879.5372, 889.2245, 898.9066, 908.5834, 918.2551,
        927.9214, 937.5827, 947.2387, 956.8895, 966.5351, 976.1755, 985.8106, 995.4406, 1005.065, 1014.685,
        1024.299, 1033.908, 1043.512, 1053.111, 1062.704, 1072.293, 1081.876, 1091.454, 1101.026, 1110.594,
        1120.156, 1129.713, 1139.265, 1148.811, 1158.353, 1167.889, 1177.42, 1186.946, 1196.466, 1205.982,
        1215.492, 1224.997, 1234.496, 1243.991, 1253.48, 1262.964, 1253.373, 1263.346, 1273.318, 1283.291,
        1293.262, 1303.234, 1313.206, 1323.177, 1333.148, 1343.119, 1353.089, 1363.06, 1373.03, 1383.0,
        1392.969, 1402.939, 1412.908, 1422.877, 1432.845, 1442.814, 1452.782, 1462.75, 1472.718, 1482.685,
        1492.652, 1502.619, 1512.586, 1522.552, 1532.518, 1542.484, 1552.45, 1562.416, 1572.381, 1582.346,
        1592.311, 1602.275, 1612.24, 1622.204, 1632.167, 1642.131, 1652.094, 1662.057, 1672.02, 1681.983,
        1691.945, 1701.907, 1711.869, 1721.831, 1731.792, 1741.753, 1751.714, 1761.675, 1771.635, 1781.596,
        1791.556, 1801.515, 1811.475, 1821.434, 1831.393, 1841.352, 1851.31, 1861.269, 1871.227, 1880.184,
        1874.164, 1884.225, 1894.285, 1904.342, 1914.396, 1924.448, 1934.499, 1944.546, 1954.592, 1964.635,
        1974.675, 1984.714, 1994.75, 2004.784, 2014.815, 2024.845, 2034.872, 2044.896, 2054.919, 2064.939,
        2074.956, 2084.972, 2094.985, 2104.996, 2115.004, 2125.01, 2135.014, 2145.016, 2155.015, 2165.012,
        2175.007, 2184.999, 2194.989, 2204.977, 2214.962, 2224.945, 2234.926, 2244.905, 2254.881, 2264.854,
        2274.826, 2284.795, 2294.762, 2304.727, 2314.689, 2324.649, 2334.607, 2344.562, 2354.516, 2364.467,
        2374.415, 2384.361, 2394.305, 2404.247, 2414.186, 2424.123, 2434.058, 2443.99, 2453.92, 2463.848,
        2473.773, 2483.696, 2493.617, 2503.536]


# Wavelengths for training multispectral data
input_wavelengths_mu = [
   425, 480, 545, 605, 660, 725, 835, 950
]


def generate_random_indices(N, T):
    indices = []
    for _ in range(T):
        index = random.randint(0, N)
        indices.append(index)
    return indices


def find_corresponding_indices(input_wavelengths, target_wavelengths, dis):
    corresponding_indices = []
    unmatched_indices = []
    matched_indices = []
    for target_index, target_wavelength in enumerate(target_wavelengths):
        found_corresponding = False
        for input_index, input_wavelength in enumerate(input_wavelengths):
            if abs(target_wavelength - input_wavelength) <= dis:
                corresponding_indices.append(input_index)
                found_corresponding = True
                matched_indices.append(target_index)
                break
        if not found_corresponding:
            unmatched_indices.append(target_index)
    return corresponding_indices, unmatched_indices, matched_indices
