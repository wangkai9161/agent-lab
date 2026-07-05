import os
import platform

import scipy.io
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class SynDataset(Dataset):
    def __init__(self, data, data_label, use_patch=False, patch_parts=4):
        super(SynDataset, self).__init__()
        self.data = data
        self.data_label = data_label
        self.use_patch = use_patch
        self.patch_parts = max(1, int(patch_parts))

        if self.use_patch:
            self.patch_height = (self.data.size(1) + self.patch_parts - 1) // self.patch_parts
        else:
            self.patch_height = self.data.size(1)

    def __getitem__(self, index):
        if not self.use_patch:
            input_data = self.data[index:(index + 1), :, :]
            label = self.data_label[index:(index + 1), :, :]
            return input_data, label

        sample_index = index // self.patch_parts
        patch_index = index % self.patch_parts
        start = patch_index * self.patch_height
        end = start + self.patch_height

        input_data = self.data[sample_index:(sample_index + 1), start:end, :]
        label = self.data_label[sample_index:(sample_index + 1), start:end, :]

        pad_h = self.patch_height - input_data.size(1)
        if pad_h > 0:
            input_data = F.pad(input_data, (0, 0, 0, pad_h), mode="replicate")
            label = F.pad(label, (0, 0, 0, pad_h), mode="replicate")

        return input_data, label

    def __len__(self):
        if self.use_patch:
            return self.data.size(0) * self.patch_parts
        return self.data.size(0)


def _project_root():
    current_dir = os.path.split(os.path.abspath(__file__))[0]
    current_dir = current_dir.replace("\\", "/")
    return current_dir.rsplit("/", 1)[0]


def _data_path(args, data_name):
    if os.path.isabs(args.data_dir):
        return os.path.join(args.data_dir, data_name)
    return os.path.join(_project_root(), args.data_dir, data_name)


def train_data_loader(args):
    if platform.system() == "Windows":
        num_workers = 0
    else:
        num_workers = 4
    kwopt = {"num_workers": num_workers, "pin_memory": True}

    filepath = _data_path(args, args.train_data_name)
    filepath_label = _data_path(args, args.label_data_name)

    loaded_data = scipy.io.loadmat(filepath)
    data_train = torch.from_numpy(loaded_data["data"]).to(torch.float32)
    data_train = data_train[args.train_big:args.train_end, :, :]

    loaded_data = scipy.io.loadmat(filepath_label)
    data_label = torch.from_numpy(loaded_data["data"]).to(torch.float32)
    data_label = data_label[args.train_big:args.train_end, :, :]

    dataset = SynDataset(
        data_train,
        data_label,
        use_patch=args.use_patch,
        patch_parts=args.patch_parts,
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        **kwopt,
    )

    return train_loader
