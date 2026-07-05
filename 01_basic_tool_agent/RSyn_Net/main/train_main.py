import argparse
import copy
import json
import os
import random
import sys
import warnings
from datetime import datetime
from pathlib import Path
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy.io

warnings.filterwarnings("ignore")


CURRENT_DIR = os.path.split(os.path.abspath(__file__))[0]


def find_project_root(start_dir):
    cur = os.path.abspath(start_dir)
    candidates = [
        cur,
        os.path.dirname(cur),
        os.path.dirname(os.path.dirname(cur)),
        os.getcwd(),
    ]

    for candidate in candidates:
        if (
            os.path.isdir(os.path.join(candidate, "models"))
            and os.path.isdir(os.path.join(candidate, "utils"))
        ):
            return candidate.replace("\\", "/")

    raise RuntimeError(
        "Cannot find RSyn_Net root. This script must be under RSyn_Net or "
        "a subdirectory containing ../models and ../utils."
    )


PROJECT_ROOT = find_project_root(CURRENT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from models import available_models, create_model, resolve_model_name
from utils import trainer, validater
from utils.scheduler import GradualWarmupScheduler
from utils.train_data_process import train_data_loader
from utils.valid_data_process import valid_data_loader


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("yes", "true", "t", "1", "y"):
        return True
    if value in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def make_abs_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def get_mat_data_shape(args, data_name):
    mat_path = os.path.join(make_abs_path(args.data_dir), data_name)
    if not os.path.isfile(mat_path):
        raise FileNotFoundError(mat_path)

    for var_name, shape, _dtype in scipy.io.whosmat(mat_path):
        if var_name == "data":
            return shape

    raise KeyError(f"{mat_path} does not contain MATLAB variable 'data'.")


def configure_data_split(args):
    train_shape = get_mat_data_shape(args, args.train_data_name)
    valid_shape = get_mat_data_shape(args, args.valid_data_name)
    label_shape = get_mat_data_shape(args, args.label_data_name)

    if len(train_shape) != 3:
        raise ValueError(f"Expected 3D train data, got {train_shape}.")
    if train_shape[0] != valid_shape[0] or train_shape[0] != label_shape[0]:
        raise ValueError(
            "Input and label sample counts must match: "
            f"train={train_shape}, valid={valid_shape}, label={label_shape}."
        )

    sample_count = int(train_shape[0])
    args.sample_count = sample_count

    if args.split_mode == "index":
        args.test_big = args.test_big if args.test_big >= 0 else args.vild_end
        args.test_end = args.test_end if args.test_end >= 0 else sample_count
        return

    ratio_sum = args.train_ratio + args.valid_ratio + args.test_ratio
    if ratio_sum <= 0:
        raise ValueError("train_ratio + valid_ratio + test_ratio must be > 0.")

    train_ratio = args.train_ratio / ratio_sum
    valid_ratio = args.valid_ratio / ratio_sum

    train_count = int(round(sample_count * train_ratio))
    valid_count = int(round(sample_count * valid_ratio))

    if train_count <= 0:
        raise ValueError("Percent split produced an empty train set.")
    if valid_count <= 0:
        raise ValueError("Percent split produced an empty valid set.")
    if train_count + valid_count >= sample_count:
        raise ValueError("Percent split produced an empty test set.")

    args.train_big = 0
    args.train_end = train_count
    args.vild_big = train_count
    args.vild_end = train_count + valid_count
    args.test_big = args.vild_end
    args.test_end = sample_count


def set_seed(seed):
    if seed is None or seed < 0:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sanitize_name(name):
    safe_chars = []
    for char in name:
        if char.isalnum() or char in ("-", "_", "."):
            safe_chars.append(char)
        else:
            safe_chars.append("_")
    return "".join(safe_chars).strip("_")


def build_run_name(args):
    if args.run_name:
        return sanitize_name(args.run_name)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_stem = Path(args.train_data_name).stem
    label_stem = Path(args.label_data_name).stem
    parts = [
        args.model,
        f"{train_stem}_to_{label_stem}",
        f"ep{args.epochs}",
        timestamp,
    ]
    return sanitize_name("_".join(parts))


def setup_device(gpu_list):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("Using device:", device)
    print("CUDA_VISIBLE_DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"])
    print("Requested physical GPU list:", gpu_list)
    print("Project root:", PROJECT_ROOT)

    if torch.cuda.is_available():
        print("Visible CUDA device count:", torch.cuda.device_count())
        print("GPU name:", torch.cuda.get_device_name(0))
        torch.cuda.empty_cache()

    print("=" * 80)
    return device


def build_model(model_name, device, allow_dataparallel=True):
    model = create_model(model_name)
    if allow_dataparallel and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"DataParallel enabled on {torch.cuda.device_count()} visible GPUs.")
        model = nn.DataParallel(model)
    return model.to(device)


def checkpoint_state(model, optimizer, epoch, args, valid_loss=None, valid_snr=None):
    state = {
        "epoch": int(epoch),
        "net": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model": args.model,
        "run_name": args.run_name,
        "train_data_name": args.train_data_name,
        "valid_data_name": args.valid_data_name,
        "label_data_name": args.label_data_name,
    }
    if valid_loss is not None:
        state["valid_loss"] = float(valid_loss)
    if valid_snr is not None:
        state["valid_snr"] = float(valid_snr)
    return state


def remove_module_prefix(state_dict):
    clean_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            clean_state_dict[key[7:]] = value
        else:
            clean_state_dict[key] = value
    return clean_state_dict


def avg_snr(img, imgn):
    norm_img_sq = torch.norm(img, p="fro", dim=[2, 3]) ** 2
    norm_diff_sq = torch.norm(imgn - img, p="fro", dim=[2, 3]) ** 2
    snr = 10 * torch.log10((norm_img_sq + 1e-12) / (norm_diff_sq + 1e-12))
    return torch.mean(snr)


def save_config(args, run_dir):
    config_path = os.path.join(run_dir, "config.json")
    serializable = copy.deepcopy(vars(args))
    serializable["project_root"] = PROJECT_ROOT
    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(serializable, file, indent=2, ensure_ascii=False)
    return config_path


def load_resume_checkpoint(args, model, optimizer, device, scheduler):
    if not args.resume_checkpoint:
        scheduler.step()
        return 1

    checkpoint_path = make_abs_path(args.resume_checkpoint)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)

    print("Resume training from:", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["net"] if "net" in checkpoint else checkpoint
    model.load_state_dict(state_dict)

    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    for _ in range(start_epoch):
        scheduler.step()
    return start_epoch


def load_data_pair(args, input_name, label_name):
    input_path = os.path.join(make_abs_path(args.data_dir), input_name)
    label_path = os.path.join(make_abs_path(args.data_dir), label_name)

    if not os.path.isfile(input_path):
        raise FileNotFoundError(input_path)
    if not os.path.isfile(label_path):
        raise FileNotFoundError(label_path)

    input_mat = scipy.io.loadmat(input_path)
    label_mat = scipy.io.loadmat(label_path)

    if "data" not in input_mat:
        raise KeyError(f"{input_path} does not contain variable 'data'.")
    if "data" not in label_mat:
        raise KeyError(f"{label_path} does not contain variable 'data'.")

    input_data = torch.from_numpy(input_mat["data"]).to(torch.float32)
    label_data = torch.from_numpy(label_mat["data"]).to(torch.float32)
    return input_data, label_data


def predict_sample_by_patches(model, img_input_norm, args):
    if not args.use_patch:
        return model(img_input_norm)

    height = img_input_norm.shape[-2]
    patch_parts = max(1, int(args.patch_parts))
    patch_height = (height + patch_parts - 1) // patch_parts
    outputs = []

    for patch_index in range(patch_parts):
        start = patch_index * patch_height
        end = start + patch_height
        patch = img_input_norm[:, :, start:end, :]
        real_height = patch.shape[-2]

        pad_h = patch_height - real_height
        if pad_h > 0:
            patch = F.pad(patch, (0, 0, 0, pad_h), mode="replicate")

        patch_output = model(patch)
        outputs.append(patch_output[:, :, :real_height, :])

    return torch.cat(outputs, dim=-2)[:, :, :height, :]


def run_test_after_training(args, run_dir, device, checkpoint_path):
    test_dir = os.path.join(run_dir, "test")
    os.makedirs(test_dir, exist_ok=True)

    model = create_model(args.model).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["net"] if "net" in checkpoint else checkpoint
    model.load_state_dict(remove_module_prefix(state_dict), strict=True)

    data_test, data_label = load_data_pair(
        args,
        args.test_data_name,
        args.label_data_name,
    )
    data_test = data_test[args.test_big:args.test_end, :, :]
    data_label = data_label[args.test_big:args.test_end, :, :]

    if data_test.shape[0] == 0:
        raise ValueError(f"Empty test set: [{args.test_big}, {args.test_end}).")

    sample_count = data_test.shape[0]
    snr_all = np.zeros(sample_count, dtype=np.float32)
    mse_all = np.zeros(sample_count, dtype=np.float32)
    time_all = np.zeros(sample_count, dtype=np.float32)
    sample_indices = np.arange(args.test_big, args.test_end, dtype=np.int32)
    predictions = []

    print("=" * 80)
    print("Start testing.")
    print("Checkpoint:", checkpoint_path)
    print("Test input:", args.test_data_name)
    print("Test range:", f"[{args.test_big}, {args.test_end})")
    print("Test shape:", tuple(data_test.shape))
    print("=" * 80)

    model.eval()
    with torch.no_grad():
        for index in range(sample_count):
            img_input = data_test[index, :, :].unsqueeze(0).unsqueeze(0).to(device)
            img_label = data_label[index, :, :].unsqueeze(0).unsqueeze(0).to(device)

            max_vals = img_input.abs().amax(dim=(1, 2, 3), keepdim=True)
            img_input_norm = img_input / (max_vals + 1e-8)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time()
            img_output_norm = predict_sample_by_patches(model, img_input_norm, args)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time()

            img_label_norm = img_label / (max_vals + 1e-8)
            img_mse = F.mse_loss(img_label_norm, img_output_norm)
            img_output = img_output_norm * max_vals
            img_snr = avg_snr(img_label, img_output)

            snr_all[index] = img_snr.item()
            mse_all[index] = img_mse.item()
            time_all[index] = end - start
            predictions.append(img_output.detach().cpu())

            print(
                "[%02d/%02d] sample=%04d time=%.4f s, MSE=%.8f, SNR=%.4f dB"
                % (
                    index + 1,
                    sample_count,
                    int(sample_indices[index]),
                    time_all[index],
                    mse_all[index],
                    snr_all[index],
                )
            )

            del img_input, img_label, img_input_norm, img_label_norm
            del img_output_norm, img_output, img_mse, img_snr
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    prediction_np = torch.cat(predictions, dim=0).numpy()[:, 0, :, :]
    noisy_np = data_test.numpy()
    clean_np = data_label.numpy()

    mean_snr = float(np.mean(snr_all))
    mean_mse = float(np.mean(mse_all))
    mean_time = float(np.mean(time_all))

    test_results_mat = os.path.join(test_dir, "test_results.mat")
    scipy.io.savemat(
        test_results_mat,
        {
            "noisy_data": noisy_np,
            "clean_data": clean_np,
            "denoised_data": prediction_np,
            "data": prediction_np,
            "snr": snr_all,
            "mse": mse_all,
            "time": time_all,
            "sample_indices": sample_indices,
            "mean_snr": mean_snr,
            "mean_mse": mean_mse,
            "mean_time": mean_time,
            "model": args.model,
            "checkpoint": checkpoint_path,
            "test_data_name": args.test_data_name,
            "label_data_name": args.label_data_name,
            "test_big": args.test_big,
            "test_end": args.test_end,
        },
    )

    test_snr_mat = os.path.join(test_dir, "test_snr_per_sample.mat")
    scipy.io.savemat(
        test_snr_mat,
        {
            "sample_indices": sample_indices,
            "snr": snr_all,
            "mean_snr": mean_snr,
            "model": args.model,
            "checkpoint": checkpoint_path,
            "test_data_name": args.test_data_name,
            "label_data_name": args.label_data_name,
        },
    )

    test_log = os.path.join(test_dir, "test.log")
    with open(test_log, "a", encoding="utf-8") as file:
        file.write("=" * 80 + "\n")
        file.write(f"Model: {args.model}\n")
        file.write(f"Checkpoint: {checkpoint_path}\n")
        file.write(f"Test input: {args.test_data_name}\n")
        file.write(f"Label: {args.label_data_name}\n")
        file.write(f"Test range: [{args.test_big}, {args.test_end})\n")
        file.write(f"Mean SNR: {mean_snr:.8f}\n")
        file.write(f"Mean MSE: {mean_mse:.10f}\n")
        file.write(f"Mean time: {mean_time:.8f}\n")
        file.write(f"Results mat: {test_results_mat}\n")
        file.write(f"SNR mat: {test_snr_mat}\n")
        file.write("=" * 80 + "\n")

    print("=" * 80)
    print("Testing finished.")
    print("Mean SNR:", mean_snr)
    print("Mean MSE:", mean_mse)
    print("Results mat:", test_results_mat)
    print("SNR mat:", test_snr_mat)
    print("=" * 80)

    return {
        "test_dir": test_dir,
        "test_results_mat": test_results_mat,
        "test_snr_mat": test_snr_mat,
        "mean_snr": mean_snr,
        "mean_mse": mean_mse,
        "mean_time": mean_time,
    }


def train_single_model(args):
    args.model = resolve_model_name(args.model)
    configure_data_split(args)
    args.run_name = build_run_name(args)
    set_seed(args.seed)

    device = setup_device(args.gpu_list)
    output_root = make_abs_path(args.output_root)
    run_dir = os.path.join(output_root, args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    log_path = os.path.join(run_dir, "train.log")
    metrics_path = os.path.join(run_dir, "metrics.npz")
    metrics_mat_path = os.path.join(run_dir, "train_metrics.mat")
    config_path = save_config(args, run_dir)

    print("Available models:", ", ".join(available_models()))
    print("Model:", args.model)
    print("Run name:", args.run_name)
    print("Run dir:", run_dir)
    print("Config:", config_path)

    model = build_model(args.model, device, args.allow_dataparallel)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    cosine_tmax = max(1, args.epochs - args.warm_epochs)
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        cosine_tmax,
        eta_min=args.eta_min,
    )
    scheduler = GradualWarmupScheduler(
        optimizer,
        multiplier=1,
        total_epoch=args.warm_epochs,
        after_scheduler=scheduler_cosine,
    )
    criterion = F.mse_loss

    train_loader = train_data_loader(args)
    valid_loader = valid_data_loader(args)
    start_epoch = load_resume_checkpoint(args, model, optimizer, device, scheduler)

    best_valid_snr = -1e9
    best_epoch = 0
    all_train_loss = []
    all_train_snr = []
    all_valid_loss = []
    all_valid_snr = []

    with open(log_path, "a", encoding="utf-8") as file:
        file.write("=" * 80 + "\n")
        file.write(f"Start time: {datetime.now()}\n")
        file.write(f"Project root: {PROJECT_ROOT}\n")
        file.write(f"Model: {args.model}\n")
        file.write(f"Run name: {args.run_name}\n")
        file.write(f"Run dir: {run_dir}\n")
        file.write(f"Total parameters: {total_params}\n")
        file.write(f"Trainable parameters: {trainable_params}\n")
        file.write(f"Epochs: {args.epochs}\n")
        file.write(f"Batch size: {args.batch_size}\n")
        file.write(f"Use patch: {args.use_patch}\n")
        file.write(f"Patch parts: {args.patch_parts}\n")
        file.write(f"GPU list: {args.gpu_list}\n")
        file.write(f"DataParallel: {args.allow_dataparallel}\n")
        file.write(f"LR: {args.lr}\n")
        file.write(f"Train range: [{args.train_big}, {args.train_end})\n")
        file.write(f"Valid range: [{args.vild_big}, {args.vild_end})\n")
        file.write(f"Test range: [{args.test_big}, {args.test_end})\n")
        file.write(f"Split mode: {args.split_mode}\n")
        file.write(f"Split ratios: {args.train_ratio}, {args.valid_ratio}, {args.test_ratio}\n")
        file.write(f"Sample count: {args.sample_count}\n")
        file.write(f"Train data: {args.train_data_name}\n")
        file.write(f"Valid data: {args.valid_data_name}\n")
        file.write(f"Test data: {args.test_data_name}\n")
        file.write(f"Label data: {args.label_data_name}\n")
        file.write("=" * 80 + "\n")

    for epoch in range(start_epoch, args.epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"[{args.model}] epoch {epoch:03d}/{args.epochs:03d}, lr={current_lr:.5e}")

        train_loss, train_snr = trainer.train(
            train_loader,
            model,
            criterion,
            optimizer,
            device,
        )
        scheduler.step()
        valid_loss, valid_snr = validater.validate(
            valid_loader,
            model,
            criterion,
            device,
        )

        all_train_loss.append(float(train_loss))
        all_train_snr.append(float(train_snr))
        all_valid_loss.append(float(valid_loss))
        all_valid_snr.append(float(valid_snr))

        message = (
            f"[{epoch:03d}/{args.epochs:03d}] "
            f"Train loss*1e4: {train_loss * 1e4:.8f}, "
            f"Train SNR: {train_snr:.6f}, "
            f"Valid loss*1e4: {valid_loss * 1e4:.8f}, "
            f"Valid SNR: {valid_snr:.6f}\n"
        )
        print(message)
        with open(log_path, "a", encoding="utf-8") as file:
            file.write(message)

        if epoch >= args.save_start_epoch and epoch % args.save_interval == 0:
            save_path = os.path.join(run_dir, f"net_params_{epoch}.pth")
            torch.save(
                checkpoint_state(model, optimizer, epoch, args, valid_loss, valid_snr),
                save_path,
            )
            print("Checkpoint saved:", save_path)

        if epoch >= args.best_start_epoch and epoch % args.best_interval == 0:
            if valid_snr > best_valid_snr:
                best_valid_snr = float(valid_snr)
                best_epoch = int(epoch)
                best_path = os.path.join(run_dir, "best_model.pth")
                torch.save(
                    checkpoint_state(model, optimizer, epoch, args, valid_loss, valid_snr),
                    best_path,
                )
                print("Best model saved:", best_path)

        latest_path = os.path.join(run_dir, "latest_model.pth")
        torch.save(
            checkpoint_state(model, optimizer, epoch, args, valid_loss, valid_snr),
            latest_path,
        )

    np.savez(
        metrics_path,
        train_loss=np.array(all_train_loss),
        train_snr=np.array(all_train_snr),
        valid_loss=np.array(all_valid_loss),
        valid_snr=np.array(all_valid_snr),
        best_epoch=best_epoch,
        best_valid_snr=best_valid_snr,
        model=args.model,
        run_name=args.run_name,
    )
    scipy.io.savemat(
        metrics_mat_path,
        {
            "epoch": np.arange(start_epoch, args.epochs + 1, dtype=np.int32),
            "train_loss": np.array(all_train_loss, dtype=np.float32),
            "train_snr": np.array(all_train_snr, dtype=np.float32),
            "valid_loss": np.array(all_valid_loss, dtype=np.float32),
            "valid_snr": np.array(all_valid_snr, dtype=np.float32),
            "best_epoch": best_epoch,
            "best_valid_snr": best_valid_snr,
            "model": args.model,
            "run_name": args.run_name,
        },
    )

    with open(log_path, "a", encoding="utf-8") as file:
        file.write(f"End time: {datetime.now()}\n")
        file.write(f"Best epoch: {best_epoch}\n")
        file.write(f"Best valid SNR: {best_valid_snr:.8f}\n")
        file.write(f"Metrics: {metrics_path}\n")
        file.write(f"Metrics mat: {metrics_mat_path}\n")
        file.write("=" * 80 + "\n")

    best_checkpoint = os.path.join(run_dir, "best_model.pth")
    latest_checkpoint = os.path.join(run_dir, "latest_model.pth")
    test_result = None
    if args.run_test:
        checkpoint_for_test = best_checkpoint if os.path.isfile(best_checkpoint) else latest_checkpoint
        test_result = run_test_after_training(args, run_dir, device, checkpoint_for_test)

    print("=" * 80)
    print("Training finished.")
    print("Run dir:", run_dir)
    print("Best epoch:", best_epoch)
    print("Best valid SNR:", best_valid_snr)
    print("Metrics:", metrics_path)
    print("Metrics mat:", metrics_mat_path)
    if test_result is not None:
        print("Test results mat:", test_result["test_results_mat"])
        print("Test SNR mat:", test_result["test_snr_mat"])
    print("=" * 80)

    return {
        "model": args.model,
        "run_name": args.run_name,
        "run_dir": run_dir,
        "best_epoch": best_epoch,
        "best_valid_snr": best_valid_snr,
        "metrics": metrics_path,
        "metrics_mat": metrics_mat_path,
        "test": test_result,
    }


def build_parser():
    parser = argparse.ArgumentParser(description="Train one RSyn_Net model.")

    parser.add_argument("--model", default="haar_wavelet_subband_attention_unet", type=str)
    parser.add_argument("--gpu_list", default="0", type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--allow_dataparallel", default=False, type=str2bool)

    parser.add_argument("--data_dir", default="data", type=str)
    parser.add_argument("--train_data_name", default="demo_synthetic_blending_light.mat", type=str)
    parser.add_argument("--valid_data_name", default="demo_synthetic_blending_light.mat", type=str)
    parser.add_argument("--test_data_name", default="", type=str)
    parser.add_argument("--label_data_name", default="demo_synthetic_clean.mat", type=str)

    parser.add_argument("--split_mode", default="percent", choices=["percent", "index"], type=str)
    parser.add_argument("--train_ratio", default=0.7, type=float)
    parser.add_argument("--valid_ratio", default=0.15, type=float)
    parser.add_argument("--test_ratio", default=0.15, type=float)
    parser.add_argument("--train_big", default=0, type=int)
    parser.add_argument("--train_end", default=250, type=int)
    parser.add_argument("--vild_big", default=250, type=int)
    parser.add_argument("--vild_end", default=300, type=int)
    parser.add_argument("--test_big", default=-1, type=int)
    parser.add_argument("--test_end", default=-1, type=int)

    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--use_patch", default=True, type=str2bool)
    parser.add_argument("--patch_parts", default=4, type=int)
    parser.add_argument("--lr", "--learning_rate", default=2e-4, type=float)
    parser.add_argument("--eta_min", default=5e-5, type=float)
    parser.add_argument("--warm_epochs", default=3, type=int)

    parser.add_argument("--save_start_epoch", default=50, type=int)
    parser.add_argument("--save_interval", default=50, type=int)
    parser.add_argument("--best_start_epoch", default=1, type=int)
    parser.add_argument("--best_interval", default=1, type=int)
    parser.add_argument("--resume_checkpoint", default="", type=str)
    parser.add_argument("--run_test", default=True, type=str2bool)

    parser.add_argument("--output_root", default="runs/train", type=str)
    parser.add_argument("--run_name", default="", type=str)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not args.test_data_name:
        args.test_data_name = args.valid_data_name
    train_single_model(args)


if __name__ == "__main__":
    main()
