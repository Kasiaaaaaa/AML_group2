import argparse
import time as timep
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import math
import os
import platform  # for get_device()
import logging
from logging.handlers import RotatingFileHandler
from itertools import product
import csv
from typing import Optional

from audio_augment import NoiseAugment
from models.resnet import resnet19
from models.spike_vgg import vgg16
from models.spike_vgg_dvs import VGGSNNwoAP
import data_loaders
from utils import TET_loss, seed_all, Energy
from models.snn_recurrent import RecurrentSpikeModel


# ---- Logging helpers ---------------------------------------------------------
def _str_to_loglevel(name: str) -> int:
    name = (name or "INFO").upper()
    return getattr(logging, name, logging.INFO)

class StreamHandlerTqdm(logging.StreamHandler):
    """Write logs above tqdm progress bars without breaking them."""
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            pass

def setup_logger(name: str, log_dir: str = "./logs", level: str = "INFO") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(_str_to_loglevel(level))
    logger.propagate = False
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch = StreamHandlerTqdm()  # console handler that plays nice with tqdm
    ch.setLevel(_str_to_loglevel(level))
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = RotatingFileHandler(
        filename=os.path.join(log_dir, f"{name}.log"),
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
    )
    fh.setLevel(_str_to_loglevel(level))
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger
# -----------------------------------------------------------------------------


parser = argparse.ArgumentParser(description='PyTorch Temporal Efficient Training')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--dset', default='c10', type=str,
                    choices=['c10', 'c100', 'ti', 'c10dvs', 'sc_mel', 'sc_mfcc'],
                    help='dataset')
parser.add_argument('--model', default='res19', type=str, choices=['vgg16', 'res19', 'vgg'],
                    help='neural network architecture')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=256, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--seed', default=1000, type=int, help='random seed')
parser.add_argument('-T', '--time', default=4, type=int, metavar='N',
                    help='SNN simulation time (default: 4)')
parser.add_argument('--test-all', action='store_true', help='report accuracy at every timestep')
parser.add_argument('--t', default=0.9, type=float, metavar='N',
                    help='threshold for entropy/confidence (early exit)')
parser.add_argument('--TET', action='store_false', help='if use Temporal Efficient Training')
parser.add_argument('--gpu-test', action='store_true', help='use recurrent GPU runtime test')
parser.add_argument('--aet-test', action='store_true', help='test AET/empirical AET')

# NEW: data & checkpoint
parser.add_argument('--data-root', default='/scratch/s4723708/SEENN/data', type=str, help='dataset root directory')
parser.add_argument('--ckpt', default='/scratch/s4723708/SEENN/train/sc_res19_stripped.pt', type=str,
                    help='path to checkpoint (.pt). If None, falls back to ./raw/<dset>_<model>[_wotet].pt')

# --- test-time noise controls (single condition) ---
parser.add_argument(
    '--test-noise', type=str, default=None,
    choices=[None, 'clean', 'doing_the_dishes', 'dude_miaowing', 'exercise_bike', 'running_tap', 'pink', 'white'],
    help='Noise type for testing; omit or use "clean" for no noise.'
)
parser.add_argument(
    '--snr-range', nargs=2, type=float, default=[10.0, 10.0],
    metavar=('LOW', 'HIGH'),
    help='SNR range (dB) for mixing the selected noise at test time.'
)

# --- sweep multiple noises/SNRs (overrides single-condition flow if provided) ---
parser.add_argument(
    '--sweep-noise', nargs='+', default=None,
    help='List of noise types to sweep (e.g., clean doing_the_dishes exercise_bike).'
)
parser.add_argument(
    '--sweep-snr', nargs='+', type=float, default=None,
    help='List of SNR(dB) values to sweep (e.g., 0 5 10 15 20).'
)

# logging CLI
parser.add_argument('--log-dir', default='./logs', type=str, help='where to write log files')
parser.add_argument('--log-level', default='INFO', type=str,
                    choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'], help='logging level')

args = parser.parse_args()

# init logger
LOG = setup_logger(
    name=f"test_{getattr(args,'dset','run')}_{getattr(args,'model','model')}",
    log_dir=args.log_dir,
    level=args.log_level,
)


# ------------------------ helpers ------------------------
def get_device():
    # Prefer CUDA if it’s usable
    if torch.cuda.is_available():
        return torch.device("cuda")

    # MPS exists only on newer torch (>=1.12) and macOS
    if (
        hasattr(torch, "backends") and
        hasattr(torch.backends, "mps") and
        platform.system() == "Darwin" and
        torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    # Fallback to CPU
    return torch.device("cpu")


@torch.no_grad()
def throughput_test(model: RecurrentSpikeModel, test_loader, device, dynamic=True,
                    threshold=0.2, metric='entropy', timesteps=4, measure_energy=False):
    model.eval()
    overall_time = 0.0
    total = 0.0
    correct = 0.0

    for inputs, targets in test_loader:
        inputs = inputs.to(device).float()  # ensure float!
        targets_cpu = targets.cpu()

        s_time = timep.time()
        model.re_init()

        if not dynamic:
            logits = torch.zeros(inputs.size(0), model.num_classes, device=device)
            for _ in range(timesteps):
                logits += model(inputs)
            preds = logits.detach().cpu().argmax(1)
            total += float(inputs.size(0))
            correct += float((preds == targets_cpu).sum().item())
        else:
            partial = torch.zeros(inputs.size(0), model.num_classes, device=device)
            decided = torch.zeros(inputs.size(0), dtype=torch.bool, device=device)
            final_pred = torch.empty_like(targets_cpu)

            for s_i in range(timesteps):
                partial += model(inputs)

                if metric == 'entropy':
                    logp = torch.log_softmax(partial, dim=1)
                    ent = -torch.sum(logp * torch.exp(logp), dim=1) / math.log(partial.size(1))
                    score = 1.0 - ent
                elif metric == 'confidence':
                    score = torch.softmax(partial, dim=1).max(dim=1).values
                else:
                    raise NotImplementedError

                newly = (~decided) & (score > threshold)
                if newly.any():
                    preds = partial[newly].detach().cpu().argmax(1)
                    final_pred[newly] = preds
                    decided[newly] = True

                if s_i == timesteps - 1:
                    remaining = ~decided
                    if remaining.any():
                        preds = partial[remaining].detach().cpu().argmax(1)
                        final_pred[remaining] = preds
                        decided[remaining] = True
                    break

            total += float(inputs.size(0))
            correct += float((final_pred == targets_cpu).sum().item())

        e_time = timep.time()
        overall_time += (e_time - s_time)

    throughput = total / max(overall_time, 1e-8)
    final_acc = 100.0 * correct / max(total, 1.0)
    LOG.info(f"Throughput: {throughput:.2f} samples/s, accuracy: {final_acc:.2f}%")


@torch.no_grad()
def test(model, test_loader, device, dynamic=True, threshold=0.2, metric='entropy', save_image=False):
    correct = 0.0
    total = 0.0
    time_vec = np.zeros(args.time, dtype=np.float64)
    model.eval()

    score_list = []

    for inputs, targets in test_loader:
        inputs = inputs.to(device).float()
        targets_cpu = targets.cpu()
        outputs = model(inputs)                # [B, T, C]

        if not dynamic:
            mean_out = outputs.mean(1)
            preds = mean_out.detach().cpu().argmax(1)
            total += float(targets.size(0))
            correct += float((preds == targets_cpu).sum().item())
        else:
            metric_list = []
            B = targets.size(0)
            total += float(B)

            for s_i in range(args.time):
                t = s_i + 1
                part = outputs[:, :t].sum(1) if s_i > 0 else outputs[:, 0]
                if metric == 'entropy':
                    logp = torch.log_softmax(part, dim=1)
                    ent = -torch.sum(logp * torch.exp(logp), dim=1) / math.log(part.shape[1])
                    metric_list += [1 - ent]
                elif metric == 'confidence':
                    conf = torch.softmax(part, dim=1).max(dim=1).values
                    metric_list += [conf]
                else:
                    raise NotImplementedError

            score_list += [torch.stack(metric_list, dim=0)]

            final_logits = []
            for b in range(B):
                for s_i in range(args.time):
                    if metric_list[s_i][b] > threshold or s_i == args.time - 1:
                        t = s_i + 1
                        time_vec[s_i] += 1
                        final_logits.append(outputs[b, :t].sum(0) if s_i > 0 else outputs[b, 0])
                        break
            final_logits = torch.stack(final_logits, dim=0)
            preds = final_logits.detach().cpu().argmax(1)
            correct += float((preds == targets_cpu).sum().item())

    if save_image and len(score_list) > 0:
        score_list = torch.cat(score_list, dim=1)
        os.makedirs("raw", exist_ok=True)
        np.save("raw/conf_dist.npy", score_list.cpu().numpy())

    final_acc = 100.0 * correct / max(total, 1.0)
    if dynamic:
        time_steps = np.arange(1, args.time + 1, dtype=np.float64)
        total_time = float((time_vec * time_steps).sum())
        avg_time = total_time / max(total, 1.0)
        time_ratio = time_vec / max(total, 1.0)
        return final_acc, avg_time, time_ratio
    else:
        return final_acc


@torch.no_grad()
def test_all_time_steps(model, test_loader, device):
    correct = np.zeros(args.time, dtype=np.float64)
    total = 0.0
    model.eval()
    for inputs, targets in test_loader:
        inputs = inputs.float().to(device)
        outputs = model(inputs)  # [B,T,C]
        total += float(targets.size(0))
        for t in range(args.time):
            mean_out = outputs[:, :(t + 1)].mean(1) if t > 0 else outputs[:, 0]
            preds = mean_out.detach().cpu().argmax(1)
            correct[t] += float((preds == targets).sum().item())

    final_accs = 100.0 * correct / max(total, 1.0)
    LOG.info(f'Accuracy per timestep: {final_accs}')


@torch.no_grad()
def test_aet(model, test_loader, device):
    T = args.time
    correct = np.zeros(T, dtype=np.float64)     # cumulative correct by t
    newly_counts = np.zeros(T, dtype=np.float64)  # first-time-correct at t
    total = 0.0
    emp_steps_sum = 0.0
    model.eval()

    for inputs, targets in tqdm(test_loader):
        inputs = inputs.to(device).float()
        outputs = model(inputs)                 # [B,T,C]
        B = targets.size(0)
        total += float(B)
        seen_correct = np.zeros(B, dtype=bool)

        for t in range(T):
            mean_out = outputs[:, :(t+1)].mean(1) if t > 0 else outputs[:, 0]
            preds = mean_out.detach().cpu().argmax(1)
            c = (preds == targets.cpu()).numpy()    # [B] bool

            correct[t] += c.sum()

            newly = (~seen_correct) & c
            newly_counts[t] += newly.sum()
            emp_steps_sum += (t + 1) * newly.sum()
            seen_correct |= c

        emp_steps_sum += (T) * (~seen_correct).sum()  # never-correct → T steps

    # analytic AET
    aet = 0.0
    prev = 0.0
    for t in range(T):
        delta = correct[t] - prev
        aet += (t + 1) * delta
        prev = correct[t]
    aet += (total - correct[-1]) * T
    aet /= max(total, 1.0)

    # empirical AET
    emp_aet = emp_steps_sum / max(total, 1.0)

    # per-t accuracy (%)
    acc_per_t = (100.0 * correct / max(total, 1.0)).tolist()

    # per-t AET curves (optional but handy)
    aet_curve, emp_aet_curve = [], []
    steps_sum = 0.0
    prev = 0.0
    for t in range(T):
        delta = correct[t] - prev
        steps_sum += (t + 1) * delta
        aet_curve.append(float((steps_sum + (t + 1) * (total - correct[t])) / max(total, 1.0)))
        prev = correct[t]

    emp_steps_cum = 0.0
    for t in range(T):
        emp_steps_cum += (t + 1) * newly_counts[t]
        emp_aet_curve.append(float((emp_steps_cum + (t + 1) * (total - correct[t])) / max(total, 1.0)))

    LOG.info(f"AET: {aet:.3f}, Empirical AET: {emp_aet:.3f}")
    LOG.info(f"acc_per_t (%): {[round(x,3) for x in acc_per_t]}")
    return float(aet), float(emp_aet), acc_per_t, aet_curve, emp_aet_curve


@torch.no_grad()
def test_aet_cs_thresholded(model, test_loader, device, T: int, threshold: float = 0.9, metric: str = "confidence"):
    """
    Dynamic early-exit evaluation that *uses* the confidence/entropy threshold.

    Returns:
      avg_steps: float
      overall_acc_dynamic: float (%)
      portion: np.ndarray [T] (fractions exiting at each timestep, 1-based)
      acc_exited_by_t: list length T with per-t accuracies (%) computed only on
                       samples that exited *at that t due to threshold*
      acc_forced_T: float (%) accuracy of samples forced at T (never crossed thr)
    """
    model.eval()

    total = 0
    exit_t_all = []       # 1..T for each sample
    correct_all = []      # correctness of prediction at exit (True/False)
    crossed_mask_all = [] # whether exit was due to threshold (True) vs forced at T (False)

    # buckets for per-t accuracy (only those *that crossed* at that t)
    correct_by_t = np.zeros(T, dtype=np.int64)
    count_by_t   = np.zeros(T, dtype=np.int64)

    forced_correct = 0
    forced_count   = 0

    for inputs, targets in test_loader:
        inputs = inputs.to(device).float()
        targets = targets.to(device)
        B = targets.size(0)
        total += B

        # logits per time: shape [B, T, C]
        outputs = model(inputs)

        # running partial logits used for scoring/prediction
        partial = torch.zeros(B, outputs.size(-1), device=device)
        decided = torch.zeros(B, dtype=torch.bool, device=device)
        crossed = torch.zeros(B, dtype=torch.bool, device=device)  # True if crossed threshold

        exit_pred = torch.empty(B, dtype=torch.long, device=device)
        exit_t    = torch.empty(B, dtype=torch.long, device=device)  # 1..T

        for s_i in range(T):
            # accumulate logits to current time
            partial = partial + outputs[:, s_i, :]

            # compute score
            if metric == "confidence":
                score = torch.softmax(partial, dim=1).max(dim=1).values
            elif metric == "entropy":
                logp = torch.log_softmax(partial, dim=1)
                ent  = -torch.sum(torch.exp(logp) * logp, dim=1) / math.log(partial.size(1))
                score = 1.0 - ent
            else:
                raise ValueError("metric must be 'confidence' or 'entropy'")

            # which undecided samples now cross threshold?
            newly = (~decided) & (score > threshold)

            # assign exit for newly-crossed
            if newly.any():
                preds = partial[newly].argmax(dim=1)
                exit_pred[newly] = preds
                exit_t[newly]    = s_i + 1  # 1-based
                decided[newly]   = True
                crossed[newly]   = True

        # force any remaining undecided to exit at T
        remaining = ~decided
        if remaining.any():
            preds = partial[remaining].argmax(dim=1)
            exit_pred[remaining] = preds
            exit_t[remaining]    = T
            decided[remaining]   = True
            # crossed[remaining] stays False

        # correctness
        corr = (exit_pred == targets)

        # collect global arrays (detach to CPU numpy)
        exit_t_all.append(exit_t.cpu().numpy())
        correct_all.append(corr.cpu().numpy().astype(np.int64))
        crossed_mask_all.append(crossed.cpu().numpy())

        # per-t bins (only those that crossed due to threshold)
        for s in range(1, T + 1):
            mask = (exit_t == s) & (crossed)
            if mask.any():
                correct_by_t[s - 1] += corr[mask].sum().item()
                count_by_t[s - 1]   += int(mask.sum().item())

        # forced-at-T bucket
        forced_mask = (exit_t == T) & (~crossed)
        if forced_mask.any():
            forced_correct += corr[forced_mask].sum().item()
            forced_count   += int(forced_mask.sum().item())

    # stack all samples
    exit_t_all = np.concatenate(exit_t_all, axis=0)
    correct_all = np.concatenate(correct_all, axis=0)
    crossed_mask_all = np.concatenate(crossed_mask_all, axis=0)

    # portions per t (over all samples)
    portion = np.zeros(T, dtype=np.float64)
    for s in range(1, T + 1):
        portion[s - 1] = np.mean(exit_t_all == s)

    # average exit step (AET under CS rule)
    avg_steps = float(exit_t_all.mean())

    # overall dynamic accuracy (includes forced-at-T)
    overall_acc_dynamic = 100.0 * float(correct_all.mean())

    # per-t accuracy only on samples that crossed at that t
    acc_exited_by_t = []
    for s in range(T):
        if count_by_t[s] > 0:
            acc_exited_by_t.append(100.0 * correct_by_t[s] / count_by_t[s])
        else:
            acc_exited_by_t.append(float("nan"))  # no samples crossed at this t

    acc_forced_T = (100.0 * forced_correct / forced_count) if forced_count > 0 else float("nan")

    # Optional: print nicely
    LOG.info(f"[DYNAMIC-CS] thr={threshold:.3f} metric={metric} | "
             f"avg_steps={avg_steps:.3f} | overall_acc={overall_acc_dynamic:.3f}%")
    LOG.info(f"[DYNAMIC-CS] portion (t=1..T): {np.round(portion, 3)}")
    LOG.info(f"[DYNAMIC-CS] acc_exited_by_t (only threshold-crossers): "
             f"{[None if np.isnan(a) else round(a,3) for a in acc_exited_by_t]}")
    LOG.info(f"[DYNAMIC-CS] acc_forced_T (never crossed): "
             f"{None if np.isnan(acc_forced_T) else round(acc_forced_T,3)}% "
             f"(count={forced_count})")

    return avg_steps, overall_acc_dynamic, portion, acc_exited_by_t, acc_forced_T


@torch.no_grad()
def export_per_sample_metrics(
    model,
    loader,
    device,
    T: int,
    noise_name: str,
    snr_db: float,
    csv_writer,
    write_header: bool = False,
    id_prefix: str = "",
):
    """
    For each sample in `loader`, compute per-timestep confidence and correctness
    using cumulative logits, and write one CSV row per sample.

    Columns:
      noise, snr_db, sample_id,
      t1_conf..tT_conf, t1_correct..tT_correct
    """
    model.eval()

    if write_header:
        conf_cols    = [f"t{s}_conf"    for s in range(1, T + 1)]
        correct_cols = [f"t{s}_correct" for s in range(1, T + 1)]
        header = ["noise", "snr_db", "sample_id"] + conf_cols + correct_cols
        csv_writer.writerow(header)

    running_index = 0

    for batch in loader:
        # Accept (x, y) or (x, y, meta)
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            inputs, targets, meta = batch
        else:
            inputs, targets = batch
            meta = None

        inputs = inputs.to(device).float()
        targets = targets.to(device)
        B = targets.size(0)

        # forward: outputs [B, T, C]
        outputs = model(inputs)
        assert outputs.dim() == 3 and outputs.size(1) == T, \
            f"Expected outputs [B, T, C] with T={T}, got {tuple(outputs.shape)}"

        # cumulative logits across time
        partial = torch.zeros(B, outputs.size(-1), device=device)
        conf = torch.empty(B, T, device=device)            # confidence per t
        pred = torch.empty(B, T, dtype=torch.long, device=device)

        for s in range(T):
            partial = partial + outputs[:, s, :]
            prob = torch.softmax(partial, dim=1)           # [B, C]
            conf[:, s] = prob.max(dim=1).values
            pred[:, s] = partial.argmax(dim=1)

        correct = (pred == targets.unsqueeze(1))           # [B, T] bool

        # Try to extract stable sample ids
        ids = []
        if meta is not None:
            if isinstance(meta, dict):
                if "path" in meta and isinstance(meta["path"], (list, tuple)):
                    ids = list(meta["path"])
                elif "uid" in meta and isinstance(meta["uid"], (list, tuple)):
                    ids = [str(u) for u in meta["uid"]]
            elif isinstance(meta, (list, tuple)) and len(meta) == B and isinstance(meta[0], str):
                ids = list(meta)
        if not ids:
            ids = [f"{id_prefix}{running_index + i:08d}" for i in range(B)]

        conf_np    = conf.detach().cpu().numpy()
        correct_np = correct.detach().cpu().numpy().astype(np.int32)

        for i in range(B):
            row = [
                noise_name,
                snr_db,
                ids[i],
                *[f"{conf_np[i, s]:.6f}" for s in range(T)],
                *[int(correct_np[i, s]) for s in range(T)],
            ]
            csv_writer.writerow(row)

        running_index += B

# ------------------------ noise sweep helpers ------------------------
def _make_sc_noise_augment(root: str, noise_name: Optional[str], snr_db: Optional[float]):
    """
    Build a NoiseAugment for a single noise & SNR, or return None for clean.
    """
    if noise_name in (None, 'clean'):
        return None
    assert snr_db is not None, "snr_db must be provided for noisy sweep"
    test_mix = {noise_name: 1.0}
    return NoiseAugment(
        root_path=root,
        noise_mix=test_mix,
        snr_range=(snr_db, snr_db),   # fixed SNR for determinism in eval
        target_sr=16000,
    )

def _build_sc_test_dataset(dset_key: str, data_root: str, noise_aug):
    """
    (Re)build the SpeechCommands test dataset with a specific test-time augment.
    Supports 'sc_mfcc' and 'sc_mel' branches used in this file.
    Returns: test_dataset
    """
    if dset_key == 'sc_mfcc':
        sc_root = os.path.join(data_root, "SpeechCommands", "speech_commands_v0.02")
        try:
            from data_loaders import build_speechcommands_official as build_sc
            _, _, test_dataset = build_sc(data_root=data_root, noise_for_test=noise_aug)
        except Exception:
            from data_loaders import build_speechcommands as build_sc
            _, val_dataset = build_sc(data_root=data_root, noise_for_test=noise_aug)
            test_dataset = val_dataset
        return test_dataset

    elif dset_key == 'sc_mel':
        from data_loaders import build_speech_commands2_sc
        _, test_dataset = build_speech_commands2_sc(
            data_root=data_root, n_mels=100, time_steps=100, spike_encoding='rate',
            use_log_mel=True, temporal_mode='sequential', n_channels=1,
            noise_for_val=noise_aug,
        )
        return test_dataset

    else:
        raise NotImplementedError("Noise sweep is implemented for sc_mfcc and sc_mel only.")


# ------------------------ main ------------------------
if __name__ == '__main__':
    seed_all(args.seed)

    data_root = os.path.expandvars(getattr(args, 'data_root', './data'))
    has_test_split = False

    # ---------------- dataset selection ----------------
    if args.dset == 'c10':
        train_dataset, val_dataset = data_loaders.build_cifar(
            use_cifar10=True, cutout=False, auto_aug=False, data_root=data_root)
        num_cls, wd, in_c = 10, 5e-4, 3

    elif args.dset == 'c100':
        train_dataset, val_dataset = data_loaders.build_cifar(
            use_cifar10=False, cutout=True, auto_aug=True, data_root=data_root)
        num_cls, wd, in_c = 100, 5e-4, 3

    elif args.dset == 'c10dvs':
        train_dataset, val_dataset = data_loaders.build_dvscifar(
            os.path.join(data_root, 'cifar-dvs'), transform=1)
        num_cls, wd, in_c = 10, 1e-4, 2

    elif args.dset == 'sc_mfcc':
        test_noise_aug = None
        if args.test_noise not in (None, 'clean'):
            sc_root = os.path.join(data_root, "SpeechCommands", "speech_commands_v0.02")
            test_mix = {args.test_noise: 1.0}
            test_noise_aug = NoiseAugment(
                root_path=sc_root,
                noise_mix=test_mix,
                snr_range=tuple(args.snr_range),
                target_sr=16000,
            )
        
        LOG.info(f"Noise type: {args.test_noise}, SNR dB: {tuple(args.snr_range)}, "
                 f"augmentor={'ON' if test_noise_aug is not None else 'OFF'}")

        try:
            from data_loaders import build_speechcommands_official as build_sc
            train_dataset, val_dataset, test_dataset = build_sc(
                data_root=data_root, noise_for_test=test_noise_aug
            )
            has_test_split = True
        except Exception:
            from data_loaders import build_speechcommands as build_sc
            train_dataset, val_dataset = build_sc(
                data_root=data_root, noise_for_test=test_noise_aug
            )
            test_dataset = val_dataset
            has_test_split = True

        num_cls, wd, in_c = 10, 5e-4, 1

    elif args.dset == 'sc_mel':
        from data_loaders import build_speech_commands2_sc
        test_noise_aug = None
        if args.test_noise not in (None, 'clean'):
            sc_root = os.path.join(data_root, "SpeechCommands", "speech_commands_v0.02")
            test_mix = {args.test_noise: 1.0}
            test_noise_aug = NoiseAugment(
                root_path=sc_root,
                noise_mix=test_mix,
                snr_range=tuple(args.snr_range),
                target_sr=16000,
            )

        train_dataset, test_dataset = build_speech_commands2_sc(
            data_root=data_root, n_mels=100, time_steps=100, spike_encoding='rate',
            use_log_mel=True, temporal_mode='sequential', n_channels=1,
            noise_for_val=test_noise_aug,
        )
        has_test_split = True
        num_cls, wd, in_c = 8, 5e-4, 1

    else:
        raise NotImplementedError

    # ---------------- data loaders ----------------
    use_cuda = torch.cuda.is_available()
    pin_mem = True if use_cuda else False
    batch_size = 1 if args.gpu_test else args.batch_size

    # pick test source
    if has_test_split:
        test_src = test_dataset
    else:
        test_src = val_dataset

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=pin_mem
    )
    test_loader = torch.utils.data.DataLoader(
        test_src, batch_size=batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=pin_mem
    )

    # ---------------- model ----------------
    if args.model == 'vgg16':
        model = vgg16(width_mult=4, in_c=in_c, num_classes=num_cls, dspike=True, gama=3)
    elif args.model == 'vgg':
        model = VGGSNNwoAP(in_c=in_c, num_classes=num_cls)
    elif args.model == 'res19':
        model = resnet19(width_mult=8, in_c=in_c, num_classes=num_cls, use_dspike=True, gamma=3)
    else:
        raise NotImplementedError

    model.T = args.time

    # ---------------- device + checkpoint ----------------
    device = get_device()
    model.to(device)

    default_name = (f"raw/{args.dset}_{args.model}.pt"
                    if args.TET else f"raw/{args.dset}_{args.model}_wotet.pt")
    ckpt_path = os.path.expandvars(args.ckpt) if args.ckpt else default_name
    LOG.info(f"Loading checkpoint: {ckpt_path}")
    try:
        state = torch.load(ckpt_path, map_location=device)
    except Exception:
        LOG.exception(f"Failed to load checkpoint: {ckpt_path}")
        raise

    state_dict = (state if (isinstance(state, dict) and 'model' not in state)
                  else state.get('model', state))
    model.load_state_dict(state_dict)

    LOG.info(f"Device: {device} | Model: {args.model} | DSet: {args.dset} | T={args.time}")

    # ---------------- GPU throughput quick test ----------------
    if args.gpu_test:
        rec_model = RecurrentSpikeModel(model).to(device)
        throughput_test(rec_model, test_loader, device, dynamic=False,
                        threshold=args.t, metric='entropy',
                        timesteps=args.time, measure_energy=True)
        raise SystemExit

    # ---------------- Noise/SNR SWEEP ----------------
    if (args.sweep_noise is not None) or (args.sweep_snr is not None):
        if args.dset not in ('sc_mfcc', 'sc_mel'):
            LOG.warning("Noise sweep only implemented for sc_mfcc/sc_mel; running single test flow.")
        else:
            noises = args.sweep_noise if args.sweep_noise is not None else ['clean']
            snrs   = args.sweep_snr   if args.sweep_snr   is not None else [10.0]

            # Summary CSV (your existing file)
            csv_path = os.path.join(args.log_dir, f"noise_sweep_{args.dset}_{args.model}.csv")
            os.makedirs(args.log_dir, exist_ok=True)
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'noise', 'snr_db', 'threshold',
                    'avg_time_mean', 'acc_dynamic_mean',
                    'portion_t1', 'portion_t2', 'portion_t3',
                    'dyn_avg_steps', 'dyn_overall_acc',
                    'acc_exit_t1', 'acc_exit_t2', 'acc_exit_t3',
                    'acc_forced_T'
                ])

                # --- per-sample CSV path & header flag ---
                per_sample_csv = os.path.join(args.log_dir, f"noise_sweep_samples_{args.dset}_{args.model}.csv")
                os.makedirs(args.log_dir, exist_ok=True)
                write_header = (not os.path.exists(per_sample_csv)) or (os.path.getsize(per_sample_csv) == 0)

                sc_root = os.path.join(data_root, "SpeechCommands", "speech_commands_v0.02")
                for noise_name, snr_db in product(noises, snrs):
                    # build test set for this condition
                    noise_aug = _make_sc_noise_augment(sc_root, noise_name, snr_db)
                    cond_test = _build_sc_test_dataset(args.dset, data_root, noise_aug)
                    cond_loader = torch.utils.data.DataLoader(
                        cond_test, batch_size=batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=pin_mem
                    )

                    LOG.info(f"[SWEEP] noise={noise_name}, snr={snr_db} dB")

                    # --- per-sample export (append mode) ---
                    with open(per_sample_csv, "a", newline="") as f_samp:
                        sample_writer = csv.writer(f_samp)
                        export_per_sample_metrics(
                            model=model,
                            loader=cond_loader,
                            device=device,
                            T=args.time,
                            noise_name=noise_name,
                            snr_db=snr_db,
                            csv_writer=sample_writer,
                            write_header=write_header,            # ✅ write header only once
                            id_prefix=f"{noise_name}_{snr_db}_",
                        )
                    write_header = False  # ✅ next conditions won’t write the header again

                    # --- your original summaries (optional but fine to keep) ---
                    facc, avg_time, portion = test(
                        model, cond_loader, device,
                        dynamic=True, threshold=args.t, metric='confidence', save_image=False
                    )
                    LOG.info(f"[SWEEP] acc={facc:.3f}%, avg_time={avg_time:.3f}, "
                            f"portion={np.array2string(portion, precision=3)}")

                    dyn_avg_steps, dyn_overall_acc, dyn_portion, acc_by_t, acc_forced = \
                        test_aet_cs_thresholded(
                            model, cond_loader, device,
                            T=args.time, threshold=args.t, metric='confidence'
                        )

                    LOG.info(f"[DYNAMIC-CS] avg_steps={dyn_avg_steps:.3f} | "
                            f"overall_acc={dyn_overall_acc:.3f}% | "
                            f"portion={np.round(dyn_portion,3)} | "
                            f"acc_exit_t={ [None if (isinstance(a,float) and (a!=a)) else round(a,3) for a in acc_by_t] } | "
                            f"acc_forced_T={ None if (isinstance(acc_forced,float) and (acc_forced!=acc_forced)) else round(acc_forced,3) }")

                    acc_t_cols = [(None if (isinstance(a,float) and (a!=a)) else round(a,6)) for a in acc_by_t]
                    while len(acc_t_cols) < 3:
                        acc_t_cols.append(None)

                    writer.writerow([
                        noise_name, snr_db, args.t,
                        f"{avg_time:.6f}", f"{facc:.6f}",
                        f"{dyn_portion[0]:.6f}", f"{dyn_portion[1]:.6f}", f"{dyn_portion[2]:.6f}",
                        f"{dyn_avg_steps:.6f}", f"{dyn_overall_acc:.6f}",
                        acc_t_cols[0], acc_t_cols[1], acc_t_cols[2],
                        (None if (isinstance(acc_forced,float) and (acc_forced!=acc_forced)) else f"{acc_forced:.6f}")
                    ])

            LOG.info(f"Saved sweep results to: {csv_path}")
            raise SystemExit


    # ---------------- Regular single-condition tests ----------------
    LOG.info('start testing!')
    if args.test_all:
        test_all_time_steps(model, test_loader, device)

    if args.aet_test:
        test_aet(model, test_loader, device)

    facc, avg_time, portion = test(
        model, test_loader, device,
        dynamic=True, threshold=args.t, metric='confidence', save_image=False
    )
    LOG.info(f'Threshold: {args.t}, time: {avg_time:.3f}, acc: {facc:.3f}, portion: {portion}')
