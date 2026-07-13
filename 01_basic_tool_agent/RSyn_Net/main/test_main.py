import argparse
import os
import sys
import warnings
from pathlib import Path
from time import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
import torch.nn.functional as F

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

    raise RuntimeError("Cannot find RSyn_Net root.")


PROJECT_ROOT = find_project_root(CURRENT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from models import create_model, resolve_model_name


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
    input_shape = get_mat_data_shape(args, args.img_data_name)
    label_shape = get_mat_data_shape(args, args.label_data_name)

    if len(input_shape) != 3:
        raise ValueError(f"Expected 3D input data, got {input_shape}.")
    if input_shape[0] != label_shape[0]:
        raise ValueError(
            f"Input and label sample counts must match: input={input_shape}, label={label_shape}."
        )

    sample_count = int(input_shape[0])
    args.sample_count = sample_count

    if args.split_mode == "index":
        if args.test_big < 0:
            raise ValueError("--test_big must be set when --split_mode index.")
        args.test_end = args.test_end if args.test_end >= 0 else sample_count
        return

    ratio_sum = args.train_ratio + args.valid_ratio + args.test_ratio
    if ratio_sum <= 0:
        raise ValueError("train_ratio + valid_ratio + test_ratio must be > 0.")

    train_ratio = args.train_ratio / ratio_sum
    valid_ratio = args.valid_ratio / ratio_sum

    train_count = int(round(sample_count * train_ratio))
    valid_count = int(round(sample_count * valid_ratio))

    if train_count <= 0 or valid_count <= 0:
        raise ValueError("Percent split produced an empty train or valid set.")
    if train_count + valid_count >= sample_count:
        raise ValueError("Percent split produced an empty test set.")

    args.test_big = train_count + valid_count
    args.test_end = sample_count


def setup_device(gpu_list):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("Using device:", device)
    print("CUDA_VISIBLE_DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"])
    print("Requested physical GPU list:", gpu_list)
    if torch.cuda.is_available():
        print("Visible CUDA device count:", torch.cuda.device_count())
        print("GPU name:", torch.cuda.get_device_name(0))
        torch.cuda.empty_cache()
    print("=" * 80)
    return device


def remove_module_prefix(state_dict):
    clean_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            clean_state_dict[key[7:]] = value
        else:
            clean_state_dict[key] = value
    return clean_state_dict


def resolve_checkpoint(args):
    if args.checkpoint:
        checkpoint = make_abs_path(args.checkpoint)
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(checkpoint)
        return checkpoint

    if not args.run_dir:
        raise ValueError("Either --checkpoint or --run_dir must be provided.")

    run_dir = make_abs_path(args.run_dir)
    candidates = [
        os.path.join(run_dir, "best_model.pth"),
        os.path.join(run_dir, "latest_model.pth"),
    ]

    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(f"No best_model.pth or latest_model.pth found in {run_dir}.")


def infer_model_name(args, checkpoint):
    if args.model:
        return resolve_model_name(args.model)

    loaded = torch.load(checkpoint, map_location="cpu")
    if isinstance(loaded, dict) and "model" in loaded:
        return resolve_model_name(str(loaded["model"]))

    raise ValueError("--model is required when checkpoint does not contain model metadata.")


def avg_snr(img, imgn):
    norm_img_sq = torch.norm(img, p="fro", dim=[2, 3]) ** 2
    norm_diff_sq = torch.norm(imgn - img, p="fro", dim=[2, 3]) ** 2
    snr = 10 * torch.log10((norm_img_sq + 1e-12) / (norm_diff_sq + 1e-12))
    return torch.mean(snr)


def save_preview(input_data, label_data, pred_data, save_path=None, show=False):
    residual = label_data - pred_data
    vmax = max(
        np.max(np.abs(input_data)),
        np.max(np.abs(label_data)),
        np.max(np.abs(pred_data)),
        np.max(np.abs(residual)),
    )
    if vmax == 0:
        vmax = 1.0

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    items = [
        (input_data, "Input"),
        (label_data, "Label"),
        (pred_data, "Prediction"),
        (residual, "Residual"),
    ]

    for ax, (data, title) in zip(axes, items):
        ax.imshow(data, cmap="seismic", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("Trace")
    axes[0].set_ylabel("Time Sample")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=220, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def test_single_model(args):
    configure_data_split(args)
    checkpoint = resolve_checkpoint(args)
    args.model = infer_model_name(args, checkpoint)

    device = setup_device(args.gpu_list)
    model = create_model(args.model).to(device)

    checkpoint_data = torch.load(checkpoint, map_location=device)
    state_dict = checkpoint_data["net"] if "net" in checkpoint_data else checkpoint_data
    model.load_state_dict(remove_module_prefix(state_dict), strict=True)

    result_dir = make_abs_path(args.result_dir)
    if args.run_dir and args.result_dir == "auto":
        result_dir = os.path.join(make_abs_path(args.run_dir), "test")
    elif args.result_dir == "auto":
        checkpoint_parent = os.path.dirname(checkpoint)
        result_dir = os.path.join(checkpoint_parent, "test")
    os.makedirs(result_dir, exist_ok=True)

    show_dir = os.path.join(result_dir, "show_results")
    if args.is_show or args.is_savefig:
        os.makedirs(show_dir, exist_ok=True)

    input_path = os.path.join(make_abs_path(args.data_dir), args.img_data_name)
    label_path = os.path.join(make_abs_path(args.data_dir), args.label_data_name)

    print("Loading input:", input_path)
    print("Loading label:", label_path)
    data_test = scipy.io.loadmat(input_path)["data"]
    data_label = scipy.io.loadmat(label_path)["data"]

    data_test = torch.from_numpy(data_test).to(torch.float32)
    data_label = torch.from_numpy(data_label).to(torch.float32)
    data_test = data_test[args.test_big:args.test_end, :, :]
    data_label = data_label[args.test_big:args.test_end, :, :]

    if data_test.shape[0] == 0:
        raise ValueError(f"Empty test set: [{args.test_big}, {args.test_end}).")

    sample_count = data_test.shape[0]
    snr_all = np.zeros(sample_count, dtype=np.float32)
    mse_all = np.zeros(sample_count, dtype=np.float32)
    time_all = np.zeros(sample_count, dtype=np.float32)
    predictions = []

    print("=" * 80)
    print("Model:", args.model)
    print("Checkpoint:", checkpoint)
    print("Result dir:", result_dir)
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
            img_output_norm = model(img_input_norm)
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

            global_index = args.test_big + index
            print(
                "[%02d/%02d] sample=%04d time=%.4f s, MSE=%.8f, SNR=%.4f dB"
                % (index + 1, sample_count, global_index, time_all[index], mse_all[index], snr_all[index])
            )

            if args.is_show or args.is_savefig:
                fig_path = os.path.join(show_dir, "test_%04d_result.png" % global_index)
                save_preview(
                    input_data=img_input.detach().cpu().numpy()[0, 0, :, :],
                    label_data=img_label.detach().cpu().numpy()[0, 0, :, :],
                    pred_data=img_output.detach().cpu().numpy()[0, 0, :, :],
                    save_path=fig_path if args.is_savefig else None,
                    show=args.is_show,
                )

            del img_input, img_label, img_input_norm, img_label_norm
            del img_output_norm, img_output, img_mse, img_snr
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    prediction_tensor = torch.cat(predictions, dim=0)
    prediction_np = prediction_tensor.numpy()[:, 0, :, :]
    noisy_np = data_test.numpy()
    clean_np = data_label.numpy()

    mean_snr = float(np.mean(snr_all))
    mean_mse = float(np.mean(mse_all))
    mean_time = float(np.mean(time_all))

    log_path = os.path.join(result_dir, "test.log")
    with open(log_path, "a", encoding="utf-8") as file:
        file.write("=" * 80 + "\n")
        file.write(f"Model: {args.model}\n")
        file.write(f"Checkpoint: {checkpoint}\n")
        file.write(f"Input: {args.img_data_name}\n")
        file.write(f"Label: {args.label_data_name}\n")
        file.write(f"Test range: [{args.test_big}, {args.test_end})\n")
        file.write(f"Mean SNR: {mean_snr:.8f}\n")
        file.write(f"Mean MSE: {mean_mse:.10f}\n")
        file.write(f"Mean time: {mean_time:.8f}\n")
        file.write("=" * 80 + "\n")

    if args.is_resultsave:
        mat_path = os.path.join(result_dir, "test_results.mat")
        scipy.io.savemat(
            mat_path,
            {
                "noisy_data": noisy_np,
                "clean_data": clean_np,
                "denoised_data": prediction_np,
                "data": prediction_np,
                "snr": snr_all,
                "mse": mse_all,
                "time": time_all,
                "mean_snr": mean_snr,
                "mean_mse": mean_mse,
                "mean_time": mean_time,
                "model": args.model,
                "checkpoint": checkpoint,
                "test_big": args.test_big,
                "test_end": args.test_end,
            },
        )
        print("Saved:", mat_path)

    np.savez(
        os.path.join(result_dir, "test_metrics.npz"),
        snr=snr_all,
        mse=mse_all,
        time=time_all,
        mean_snr=mean_snr,
        mean_mse=mean_mse,
        mean_time=mean_time,
    )

    print("=" * 80)
    print("Testing finished.")
    print("Mean SNR:", mean_snr)
    print("Mean MSE:", mean_mse)
    print("Mean time:", mean_time)
    print("Result dir:", result_dir)
    print("=" * 80)


def build_parser():
    parser = argparse.ArgumentParser(description="Test one RSyn_Net model.")

    parser.add_argument("--model", default="", type=str)
    parser.add_argument("--checkpoint", default="", type=str)
    parser.add_argument("--run_dir", default="", type=str)
    parser.add_argument("--result_dir", default="auto", type=str)
    parser.add_argument("--gpu_list", default="0", type=str)

    parser.add_argument("--data_dir", default="data", type=str)
    parser.add_argument("--img_data_name", default="demo_synthetic_blending_light.mat", type=str)
    parser.add_argument("--label_data_name", default="demo_synthetic_clean.mat", type=str)

    parser.add_argument("--split_mode", default="percent", choices=["percent", "index"], type=str)
    parser.add_argument("--train_ratio", default=0.7, type=float)
    parser.add_argument("--valid_ratio", default=0.15, type=float)
    parser.add_argument("--test_ratio", default=0.15, type=float)
    parser.add_argument("--test_big", default=-1, type=int)
    parser.add_argument("--test_end", default=-1, type=int)

    parser.add_argument("--is_resultsave", default=True, type=str2bool)
    parser.add_argument("--is_show", default=False, type=str2bool)
    parser.add_argument("--is_savefig", default=True, type=str2bool)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    test_single_model(args)


if __name__ == "__main__":
    main()
