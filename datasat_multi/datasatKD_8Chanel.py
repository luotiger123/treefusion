import os
import re
import torch
import numpy as np
import rasterio
import cv2
from torch.utils.data import Dataset, DataLoader


import os
import torch
import numpy as np
import rasterio
import cv2
from torch.utils.data import Dataset


class MultiModalTreeCHMFullDataset(Dataset):
    """
    Full multimodal dataset for TreeFusion v1.0.

    Input:
        RGB-NIR imagery       : 4 channels
        SAR VV backscatter    : 1 channel
        SAR VH backscatter    : 1 channel
        nDSM                  : 1 channel
        Rough CHM             : 1 channel

        Total input shape:
            [8, H, W]

    Output:
        Very-high-resolution canopy height map:
            [1, H, W]

    File naming assumption:
        All modalities use the same filename key.

        optical_dir   : {key}.tif or {key}.tiff
        sar_vv_dir    : {key}.tif or {key}.tiff
        sar_vh_dir    : {key}.tif or {key}.tiff
        ndsm_dir      : {key}.tif or {key}.tiff
        roughchm_dir  : {key}.tif or {key}.tiff
        vhr_chm_dir   : {key}.tif or {key}.tiff

    Notes:
        This class keeps the original preprocessing logic used for training.
        Do not change the normalization rules unless the model is retrained.
    """

    def __init__(self,
                 optical_dir,
                 sar_vv_dir,
                 sar_vh_dir,
                 ndsm_dir,
                 roughchm_dir,
                 vhr_chm_dir,
                 target_size=256,
                 exts=(".tif", ".tiff")):
        self.optical_dir = optical_dir
        self.sar_vv_dir = sar_vv_dir
        self.sar_vh_dir = sar_vh_dir
        self.ndsm_dir = ndsm_dir
        self.roughchm_dir = roughchm_dir
        self.vhr_chm_dir = vhr_chm_dir
        self.target_size = target_size
        self.exts = exts

        # Optical normalization parameters used by the DINOv3-based encoder.
        self.rgb_mean = np.array([0.430, 0.411, 0.296], dtype=np.float32)
        self.rgb_std  = np.array([0.213, 0.156, 0.143], dtype=np.float32)

        # Collect valid samples by matching the same filename key
        # across all modalities.
        self.samples = self._collect_samples()
        print(f"[INFO] Matched {len(self.samples)} samples "
              f"(RGB-NIR + SAR + nDSM + Rough CHM + VHR CHM).")

    # --------------------------------------------------------
    # Sample matching
    # --------------------------------------------------------
    def _collect_samples(self):
        samples = []

        def list_keys(folder):
            keys = []
            for fn in os.listdir(folder):
                if fn.lower().endswith(self.exts):
                    keys.append(os.path.splitext(fn)[0])
            return set(keys)

        # Use the intersection of all modality keys to ensure that
        # each sample has complete multimodal inputs and target CHM.
        keys_optical = list_keys(self.optical_dir)
        keys_sarvv   = list_keys(self.sar_vv_dir)
        keys_sarvh   = list_keys(self.sar_vh_dir)
        keys_ndsm    = list_keys(self.ndsm_dir)
        keys_rough   = list_keys(self.roughchm_dir)
        keys_target  = list_keys(self.vhr_chm_dir)

        keys_all = keys_optical & keys_sarvv & keys_sarvh & keys_ndsm & keys_rough & keys_target
        keys_all = sorted(keys_all)

        # Select the existing file path for each key.
        # Both .tif and .tiff are supported.
        def pick_path(folder, key):
            p1 = os.path.join(folder, f"{key}.tif")
            if os.path.exists(p1):
                return p1
            p2 = os.path.join(folder, f"{key}.tiff")
            if os.path.exists(p2):
                return p2
            return None

        for key in keys_all:
            optical_fp = pick_path(self.optical_dir, key)
            sarvv_fp   = pick_path(self.sar_vv_dir, key)
            sarvh_fp   = pick_path(self.sar_vh_dir, key)
            ndsm_fp    = pick_path(self.ndsm_dir, key)
            rough_fp   = pick_path(self.roughchm_dir, key)
            target_fp  = pick_path(self.vhr_chm_dir, key)

            if None in [optical_fp, sarvv_fp, sarvh_fp, ndsm_fp, rough_fp, target_fp]:
                continue

            samples.append({
                "key": key,
                "optical": optical_fp,
                "sarvv": sarvv_fp,
                "sarvh": sarvh_fp,
                "ndsm": ndsm_fp,
                "roughchm": rough_fp,
                "target": target_fp
            })

        return samples

    # --------------------------------------------------------
    # Utility functions
    # --------------------------------------------------------
    def _read_tif(self, path):
        with rasterio.open(path) as src:
            arr = src.read()  # [C, H, W]
        return arr.astype(np.float32)

    def _resize(self, arr):
        c, h, w = arr.shape
        if (h, w) == (self.target_size, self.target_size):
            return arr
        out = np.zeros((c, self.target_size, self.target_size), dtype=np.float32)
        for i in range(c):
            out[i] = cv2.resize(arr[i], (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
        return out

    # --------------------------------------------------------
    # Modality-specific normalization
    # --------------------------------------------------------
    def _normalize_optical(self, optical):
        optical = np.clip(optical / 255.0, 0, 1)
        for i in range(3):
            optical[i] = (optical[i] - self.rgb_mean[i]) / self.rgb_std[i]
        optical[3] = np.clip(optical[3], 0, 1)
        return optical

    def _normalize_sar(self, sar):
        sar = np.clip(sar, -30, 7)
        sar = (sar + 30) / 37.0
        return sar

    def _normalize_ndsm(self, ndsm):
        ndsm = np.clip(ndsm, 0, 50)
        ndsm = ndsm / 50
        return ndsm.astype(np.float32)

    def _normalize_roughchm(self, rough):
        rough = np.clip(rough, 0, 50)
        rough = rough / 50
        return rough.astype(np.float32)

    def _normalize_chm_target(self, chm):
        chm = chm / 100.0
        chm = np.clip(chm, 0, 50)
        chm = chm / 50.0
        return chm.astype(np.float32)

    # --------------------------------------------------------
    # Load one sample
    # --------------------------------------------------------
    def __getitem__(self, idx):
        item = self.samples[idx]

        optical = self._read_tif(item["optical"])[:4]
        optical = self._resize(optical)
        optical = self._normalize_optical(optical)

        sarvv = self._read_tif(item["sarvv"])[0:1]
        sarvh = self._read_tif(item["sarvh"])[0:1]
        sar = np.concatenate([sarvv, sarvh], axis=0)
        sar = self._resize(sar)
        sar = self._normalize_sar(sar)

        ndsm = self._read_tif(item["ndsm"])[0:1]
        ndsm = self._resize(ndsm)
        ndsm = self._normalize_ndsm(ndsm)

        rough = self._read_tif(item["roughchm"])[0:1]
        rough = self._resize(rough)
        rough = self._normalize_roughchm(rough)

        x = np.concatenate([optical, sar, ndsm, rough], axis=0)
        x = torch.from_numpy(x).float()

        y = self._read_tif(item["target"])[0:1]
        y = self._resize(y)
        y = self._normalize_chm_target(y)
        y = torch.from_numpy(y).float()

        return {"x": x, "y": y, "key": item["key"]}

    def __len__(self):
        return len(self.samples)

