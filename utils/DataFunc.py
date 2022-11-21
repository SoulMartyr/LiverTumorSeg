import os
import random

import SimpleITK as sitk
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


def random_crop(tensor: torch.Tensor, crop_slices):
    assert len(tensor) != 4, "Get Tensor Is Not 4D"
    total_slices = tensor.shape[1]
    if total_slices < crop_slices:
        start = 0
    else:
        start = random.randint(0, total_slices - crop_slices)
    end = start + crop_slices
    if end > total_slices:
        end = total_slices

    return tensor[:, start:end]


def normalize_array(array: np.ndarray):
    arr_min = np.min(array)
    arr_max = np.max(array)
    arr_add_mid = np.mean([arr_min, arr_max])
    arr_sub_mid = np.mean([-arr_min, arr_max])
    return (array - arr_add_mid) / arr_sub_mid


def normalize_tensor(tensor):
    t_min = tensor.min()
    t_max = tensor.max()
    t_add_mid = torch.mean(torch.Tensor([t_min, t_max]))
    t_sub_mid = torch.mean(torch.Tensor([-t_min, t_max]))
    return (tensor - t_add_mid).true_divide(t_sub_mid)


class TiLSDataSet(Dataset):
    def __init__(self, data_path: str, index_list: list, crop_slices: int = 48, num_classes: int = 2):
        super(TiLSDataSet, self).__init__()
        self.data_path = data_path
        self.index_list = index_list
        self.crop_slices = crop_slices
        self.num_classes = num_classes

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, index):
        ct = sitk.ReadImage(os.path.join(self.data_path, "ct", self.index_list[index]), sitk.sitkInt16)
        seg = sitk.ReadImage(os.path.join(self.data_path, "seg", self.index_list[index]), sitk.sitkInt8)

        ct_array = sitk.GetArrayFromImage(ct)
        seg_array = sitk.GetArrayFromImage(seg)
        ct_array = np.expand_dims(ct_array, axis=0)
        seg_array = np.expand_dims(seg_array, axis=0)

        ct_array = normalize_array(ct_array)

        assert self.num_classes == 1 or self.num_classes == 2, "Num Classes should be 1 or 2"

        if self.num_classes == 1:
            seg_array[seg_array > 0] = 1
        else:
            seg_array = seg_array.repeat(2, axis=0)
            seg_array[0, seg_array[0] == 2] = 1
            seg_array[1, seg_array[1] == 1] = 0
            seg_array[1, seg_array[1] == 2] = 1

        ct_tensor = torch.FloatTensor(ct_array)
        seg_tensor = torch.LongTensor(seg_array)

        ct_tensor = random_crop(ct_tensor, self.crop_slices)
        seg_tensor = random_crop(seg_tensor, self.crop_slices)

        return {"ct": ct_tensor, "seg": seg_tensor}


if __name__ == "__main__":
    index_df = pd.read_csv("../data/index.csv", index_col=0)
    print(index_df)
    train_index = index_df.loc["train", "index"].strip().split(" ")

    train_dataset = TiLSDataSet(data_path="../data/preprocessed_data/train", index_list=train_index, crop_slices=2,
                                num_classes=2)

    dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    print("len", len(dataloader))

    info = next(iter(dataloader))
    print(info['ct'].shape, info['seg'].shape)
    print(info['ct'][0].min(), info['ct'][0].max())
    a = [[1, 2], [3, 4]]
    print(a[0:1, :])
