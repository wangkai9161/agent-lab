import os
import platform

import scipy.io
import torch

from .train_data_process import SynDataset


def _project_root():
    current_dir = os.path.split(os.path.abspath(__file__))[0]
    current_dir = current_dir.replace("\\", "/")
    return current_dir.rsplit("/", 1)[0]


def _data_path(args, data_name):
    if os.path.isabs(args.data_dir):
        return os.path.join(args.data_dir, data_name)
    return os.path.join(_project_root(), args.data_dir, data_name)


def valid_data_loader(args):
    if platform.system() == "Windows":
        num_workers = 0
    else:
        num_workers = 4
    kwopt = {"num_workers": num_workers, "pin_memory": True}

    filepath = _data_path(args, args.valid_data_name)
    filepath_label = _data_path(args, args.label_data_name)

    loaded_data = scipy.io.loadmat(filepath)
    data_valid = torch.from_numpy(loaded_data["data"]).to(torch.float32)
    data_valid = data_valid[args.vild_big:args.vild_end, :, :]

    loaded_data = scipy.io.loadmat(filepath_label)
    data_label = torch.from_numpy(loaded_data["data"]).to(torch.float32)
    data_label = data_label[args.vild_big:args.vild_end, :, :]

    dataset = SynDataset(
        data_valid,
        data_label,
        use_patch=args.use_patch,
        patch_parts=args.patch_parts,
    )

    valid_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        **kwopt,
    )

    return valid_loader
