import argparse
import os
import time
import csv
from datetime import datetime
import logging as logger

import torch
import torch.nn as nn
from torch import autocast
from tqdm import tqdm

# AMP: prefer device-agnostic GradScaler if available
try:
    from torch.amp import GradScaler         
except Exception:
    from torch.cuda.amp import GradScaler    

from models.spike_vgg import vgg16
from models.spike_vgg_dvs import VGGSNNwoAP
from models.resnet import resnet19
import data_loaders
from data_loaders import NoiseAugment
from utils import TET_loss, seed_all

parser = argparse.ArgumentParser(description='PyTorch Temporal Efficient Training')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--dset', default='c10', type=str, metavar='N',
                    choices=['c10', 'c100', 'ti', 'c10dvs', 'sc'],
                    help='dataset')
parser.add_argument('--model', default='res19', type=str, metavar='N',
                    choices=['vgg16', 'res19', 'vgg', 'res20'],
                    help='neural network architecture')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=64, type=int, metavar='N',
                    help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning_rate', default=0.05, type=float, metavar='LR',
                    help='initial learning rate', dest='lr')
parser.add_argument('--seed', default=1000, type=int,
                    help='seed for initializing training.')
parser.add_argument('-T', '--time', default=3, type=int, metavar='N',
                    help='SNN simulation time (default: 3)')
parser.add_argument('--means', default=1.0, type=float, metavar='N',
                    help='TET means (default: 1.0)')
parser.add_argument('--TET', action='store_false',
                    help='if use Temporal Efficient Training, default: on.')
parser.add_argument('--lamb', default=1e-3, type=float, metavar='N',
                    help='TET lambda (default: 1e-3)')
parser.add_argument('--amp', action='store_false',
                    help='if use amp training, default: on.')
parser.add_argument('--data-root', default='./data', type=str,
                    help='dataset root directory')
parser.add_argument('--outdir', default='./raw', type=str,
                    help='where to save checkpoints')

# LOG for mekhola
parser.add_argument('--logfile', default='./train.log', type=str,
                    help='path to write a human-readable log')
parser.add_argument('--csv', default='./metrics.csv', type=str,
                    help='path to write per-epoch metrics CSV')

# FOR SPEECH_COMMANDS
parser.add_argument('--sc_noise', action='store_true',
                    help='Add NoiseAugment to Speech Commands train set (off by default).')
parser.add_argument('--sc_clean_eval', action='store_true',
                    help='Also compute clean train-set accuracy in eval mode each epoch.')

# Logging upgrades
parser.add_argument('--log-dir', default='./logs', type=str,
                    help='directory to write logs and metrics CSV')
parser.add_argument('--run-name', default=None, type=str,
                    help='optional run name used in log/csv filenames')
parser.add_argument('--log-level', default='INFO',
                    choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                    help='logging level for console/file')

args = parser.parse_args()

# -------- more logging setup --------
os.makedirs(args.log_dir, exist_ok=True)

_stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
_default_run = f"{args.dset}_{args.model}_T{args.time}_bs{args.batch_size}_{_stamp}"
run_tag = args.run_name or _default_run
logfile = args.logfile if args.logfile != './train.log' else os.path.join(args.log_dir, f"{run_tag}.log")
csvfile = args.csv     if args.csv     != './metrics.csv' else os.path.join(args.log_dir, f"{run_tag}.csv")

level = getattr(logger, args.log_level.upper(), logger.INFO)
logger.basicConfig(
    level=level,
    format='[%(asctime)s][%(levelname)s] %(message)s',
    handlers=[logger.FileHandler(logfile, mode='w'), logger.StreamHandler()]
)
logger.info("Starting run")
logger.info(f"run_tag={run_tag}")
logger.info(f"logs -> {logfile}")
logger.info(f"csv  -> {csvfile}")

def get_lr(optimizer):
    for g in optimizer.param_groups:
        if 'lr' in g:
            return g['lr']
    return None


def train(model, device, train_loader, criterion, optimizer, epoch, scaler, args, use_amp):
    running_loss = 0.0
    model.train()
    total = 0.0
    correct = 0.0
    s_time = time.time()

    pbar = tqdm(train_loader, total=len(train_loader),
                desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)

    for i, (images, labels) in enumerate(pbar):
        optimizer.zero_grad(set_to_none=True)
        labels = labels.to(device, non_blocking=True).long()
        images = images.to(device, non_blocking=True).float()

        if use_amp:
            with autocast(device_type=torch.device(device).type, dtype=torch.float16):
                outputs = model(images)               # [B, T, C]
                mean_out = outputs.mean(1)           # [B, C]
                loss = TET_loss(outputs, labels, criterion, args.means, args.lamb) if args.TET \
                       else criterion(mean_out, labels)
            scaler.scale(loss.mean()).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            mean_out = outputs.mean(1)
            loss = TET_loss(outputs, labels, criterion, args.means, args.lamb) if args.TET \
                   else criterion(mean_out, labels)
            loss.mean().backward()
            optimizer.step()

        if i == 0 and epoch == args.start_epoch:
            logger.info(f"raw outputs shape from model: {outputs.shape}")

        running_loss += float(loss.item())
        total += float(labels.size(0))
        _, predicted = mean_out.detach().cpu().max(1)
        correct += float(predicted.eq(labels.detach().cpu()).sum().item())

        pbar.set_postfix(
            loss=f"{running_loss / (i + 1):.3f}",
            acc=f"{100.0 * correct / total:.2f}%"
        )

    e_time = time.time()
    return running_loss / max(i + 1, 1), 100.0 * correct / total, (e_time - s_time) / 60.0


@torch.no_grad()
def test(model, test_loader, device):
    correct = 0.0
    total = 0.0
    model.eval()
    for _, (inputs, targets) in enumerate(test_loader):
        inputs = inputs.to(device, non_blocking=True).float()
        targets = targets.to(device, non_blocking=True).long()
        outputs = model(inputs)           # [B, T, C]
        mean_out = outputs.mean(1)        # [B, C]
        _, predicted = mean_out.detach().cpu().max(1)
        total += float(targets.size(0))
        correct += float(predicted.eq(targets.detach().cpu()).sum().item())
    return 100.0 * correct / total


@torch.no_grad()
def eval_accuracy(model, loader, device, T):
    """Eval-mode accuracy on any loader (used for train-clean apples-to-apples)."""
    model.eval()
    total, correct = 0, 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True).float()
        yb = yb.to(device, non_blocking=True).long()
        logits = model(xb)            # [B,T,C]
        mean_out = logits.mean(1)     # [B,C]
        pred = mean_out.argmax(1)
        correct += (pred.detach().cpu() == yb.detach().cpu()).sum().item()
        total += yb.size(0)
    return 100.0 * correct / total


if __name__ == "__main__":
    seed_all(args.seed)

    # -------- device selection: MPS -> CUDA -> CPU --------
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(">>> Using device:", device)
    logger.info(f"Using device: {device}")

    data_root = os.path.expandvars(args.data_root)

    # -------- datasets --------
    if args.dset == "c10":
        train_dataset, test_dataset = data_loaders.build_cifar(
            use_cifar10=True, cutout=False, auto_aug=False, data_root=data_root
        )
        num_cls = 10
        wd = 5e-4
        in_c = 3

    elif args.dset == "c100":
        train_dataset, test_dataset = data_loaders.build_cifar(
            use_cifar10=False, cutout=True, auto_aug=True, data_root=data_root
        )
        num_cls = 100
        wd = 5e-4
        in_c = 3

    elif args.dset == "c10dvs":
        train_dataset, test_dataset = data_loaders.build_dvscifar(
            os.path.join(data_root, "cifar-dvs"), transform=1
        )
        num_cls = 10
        wd = 1e-4
        in_c = 2

    elif args.dset == "sc":
        sc_root = os.path.join(data_root, "SpeechCommands", "speech_commands_v0.02")

        noise_aug = NoiseAugment(
            root_path=sc_root,
            noise_mix={"clean": 0.52, "dishes": 0.16, "tap": 0.16, "biking": 0.16},
            snr_range=(5.0, 20.0),
            target_sr=16000,
        ) if args.sc_noise else None

        train_dataset, test_dataset = data_loaders.build_speechcommands(
            data_root=data_root,
            fixed_frames=100, n_mfcc=40, n_mels=64,
            noise_for_train=noise_aug,
        )
        num_cls = len(getattr(train_dataset, "classes", []))
        wd = 5e-4
        in_c = 1

    else:
        raise NotImplementedError

    logger.info(f"Dataset: {args.dset} | num_classes={num_cls} | in_c={in_c}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    logger.info(
        f"train size = {len(train_dataset)} | test size = {len(test_dataset)} | "
        f"classes = {getattr(train_dataset, 'classes', [])}"
    )
    assert len(train_dataset) > 0, "Train dataset is empty — check --data-root path."
    assert len(test_dataset)  > 0, "Test dataset is empty — check --data-root path."

    # -------- model --------
    if args.model == "vgg16":
        model = vgg16(width_mult=4, in_c=in_c, num_classes=num_cls, dspike=True, gama=3)
    elif args.model == "vgg":
        model = VGGSNNwoAP(in_c=in_c, num_classes=num_cls)
    elif args.model == "res19":
        model = resnet19(width_mult=8, in_c=in_c, num_classes=num_cls, use_dspike=True, gamma=3)
    else:
        raise NotImplementedError

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        logger.info(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    model.T = args.time
    model.to(device)

    # AMP only on CUDA
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = GradScaler(enabled=use_amp)

    # -------- checkpoints --------
    outdir = os.path.expandvars(args.outdir)
    os.makedirs(outdir, exist_ok=True)
    logger.info(f"outdir -> {outdir}")
    suffix = "" if args.TET else "_wotet"
    model_save_name = os.path.join(outdir, f"{args.dset}_{args.model}{suffix}.pt")
    print(f"Checkpoints -> {model_save_name}")
    logger.info(f"Checkpoints -> {model_save_name}")

    # -------- optim --------
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr / 256 * args.batch_size,
        weight_decay=wd,
        momentum=0.9,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, eta_min=0, T_max=args.epochs
    )

    # -------- loop --------
    best_acc = 0.0
    logger.info("start training!")

    # open CSV once and write header
    os.makedirs(os.path.dirname(csvfile) or ".", exist_ok=True)
    with open(csvfile, 'w', newline='') as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["epoch", "train_loss", "train_acc", "test_acc", "lr", "minutes", "best_acc"])

        try:
            for epoch in range(args.epochs):
                loss, acc, t_diff = train(
                    model, device, train_loader, criterion, optimizer, epoch, scaler, args, use_amp
                )
                lr_now = get_lr(optimizer)
                logger.info(f"Epoch [{epoch+1}/{args.epochs}] train_loss={loss:.5f} "
                            f"train_acc={acc:.3f}% lr={lr_now:.6f} time={t_diff:.2f}m")

                scheduler.step()

                if args.sc_clean_eval:
                    train_acc_eval = eval_accuracy(model, train_loader, device, args.time)
                    logger.info(f"Epoch [{epoch+1}/{args.epochs}] train_acc_eval={train_acc_eval:.3f}%")

                facc = test(model, test_loader, device)
                logger.info(f"Epoch [{epoch+1}/{args.epochs}] test_acc={facc:.3f}%")

                if best_acc < facc:
                    best_acc = facc
                    os.makedirs(os.path.dirname(model_save_name), exist_ok=True)
                    torch.save(model.state_dict(), model_save_name)
                    logger.info(f"[checkpoint] saved new best to {model_save_name}")

                logger.info(f"[status] best_test_acc={best_acc:.3f}%")

                writer.writerow([epoch+1, f"{loss:.5f}", f"{acc:.3f}", f"{facc:.3f}",
                                 f"{lr_now:.6f}" if lr_now is not None else "", f"{t_diff:.2f}", f"{best_acc:.3f}"])
                fcsv.flush()

        except KeyboardInterrupt:
            logger.warning("Training interrupted. Saving current model...")
            os.makedirs(os.path.dirname(model_save_name), exist_ok=True)
            torch.save(model.state_dict(), model_save_name)
            logger.info(f"Saved to {model_save_name}")
