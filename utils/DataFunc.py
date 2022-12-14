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
    def __init__(self, data_path: str, index_list: list, crop_slices: int = 48, num_classes: int = 2,
                 is_normalize: bool = False, is_flip: bool = True):
        super(TiLSDataSet, self).__init__()
        self.data_path = data_path
        self.index_list = index_list
        self.crop_slices = crop_slices
        self.num_classes = num_classes
        self.normalize = is_normalize
        self.flip = is_flip

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, index):
        ct = sitk.ReadImage(os.path.join(self.data_path, "ct", self.index_list[index]), sitk.sitkInt16)
        seg = sitk.ReadImage(os.path.join(self.data_path, "seg", self.index_list[index]), sitk.sitkInt8)

        ct_array = sitk.GetArrayFromImage(ct)
        seg_array = sitk.GetArrayFromImage(seg)

        if self.normalize:
            ct_array = normalize_array(ct_array)
        if self.flip:
            flip_num = np.random.randint(0, 8)
            if flip_num == 1:
                ct_array = np.flipud(ct_array)
                seg_array = np.flipud(seg_array)
            elif flip_num == 2:
                ct_array = np.fliplr(ct_array)
                seg_array = np.fliplr(seg_array)
            elif flip_num == 3:
                ct_array = np.rot90(ct_array, k=1, axes=(1, 2))
                seg_array = np.rot90(seg_array, k=1, axes=(1, 2))
            elif flip_num == 4:
                ct_array = np.rot90(ct_array, k=3, axes=(1, 2))
                seg_array = np.rot90(seg_array, k=3, axes=(1, 2))
            elif flip_num == 5:
                ct_array = np.fliplr(ct_array)
                seg_array = np.fliplr(seg_array)
                ct_array = np.rot90(ct_array, k=1, axes=(1, 2))
                seg_array = np.rot90(seg_array, k=1, axes=(1, 2))
            elif flip_num == 6:
                ct_array = np.fliplr(ct_array)
                seg_array = np.fliplr(seg_array)
                ct_array = np.rot90(ct_array, k=3, axes=(1, 2))
                seg_array = np.rot90(seg_array, k=3, axes=(1, 2))
            elif flip_num == 7:
                ct_array = np.flipud(ct_array)
                seg_array = np.flipud(seg_array)
                ct_array = np.fliplr(ct_array)
                seg_array = np.fliplr(seg_array)

        ct_array = ct_array.copy()
        seg_array = seg_array.copy()

        ct_array = np.expand_dims(ct_array, axis=0)

        assert self.num_classes == 2 or self.num_classes == 3, "Num Classes should be 2 or 3"

        multi_seg_array = np.zeros_like(seg_array)
        multi_seg_array = np.expand_dims(multi_seg_array, axis=0).repeat(self.num_classes, axis=0)

        multi_seg_array[0, seg_array == 0] = 1
        multi_seg_array[1, seg_array == 1] = 1
        if self.num_classes == 2:
            multi_seg_array[1, seg_array == 2] = 1
        else:
            multi_seg_array[2, seg_array == 2] = 1

        ct_tensor = torch.FloatTensor(ct_array)
        seg_tensor = torch.LongTensor(multi_seg_array)

        ct_tensor = random_crop(ct_tensor, self.crop_slices)
        seg_tensor = random_crop(seg_tensor, self.crop_slices)

        return {"ct": ct_tensor, "seg": seg_tensor}


if __name__ == "__main__":
    index_df = pd.read_csv("../data/index.csv", index_col=0)
    print(index_df)
    train_index = index_df.loc["train", "index"].strip().split(" ")

    train_dataset = TiLSDataSet(data_path="../data/preprocessed_data/train", index_list=train_index, crop_slices=2,
                                num_classes=3)

    dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    print("len", len(dataloader))

    info = next(iter(dataloader))
    print(info['ct'].shape, info['seg'].shape)
    print(info['ct'][0].min(), info['ct'][0].max())
