import torch
import random
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader,Subset
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
import warnings
import os
import torchaudio
from collections import defaultdict
from typing import List, Tuple, Dict,Optional
import soundfile as sf
from autoaugment import CIFAR10Policy, Cutout
from audio_augment import NoiseAugment
from utils import safe_load_wav

warnings.filterwarnings('ignore')


def build_cifar(cutout=False, use_cifar10=True, download=True, normalize=True, auto_aug=False, data_root="./data"):
    aug = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    if auto_aug:
        aug.append(CIFAR10Policy())

    aug.append(transforms.ToTensor())

    if cutout:
        aug.append(Cutout(n_holes=1, length=16))

    if use_cifar10:
        if normalize:
            aug.append(
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), )

        transform_train = transforms.Compose(aug)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = CIFAR10(root=data_root,
                                train=True, download=download, transform=transform_train)
        val_dataset = CIFAR10(root=data_root,
                              train=False, download=download, transform=transform_test)

    else:
        if normalize:
            aug.append(
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            )
        transform_train = transforms.Compose(aug)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        train_dataset = CIFAR100(root=data_root,
                                 train=True, download=download, transform=transform_train)
        val_dataset = CIFAR100(root=data_root,
                               train=False, download=download, transform=transform_test)

    return train_dataset, val_dataset


class DVSCifar10(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.resize = transforms.Resize(size=(48, 48))
        self.rotate = transforms.RandomRotation(degrees=30)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        data, target = torch.load(self.root + '/{}.pt'.format(index))
        data = self.resize(data.permute([3, 0, 1, 2]))

        if self.transform:
            flip = random.random() > 0.5
            if flip:
                data = torch.flip(data, dims=(2,))

            choices = ['roll', 'rotate']
            aug = random.choice(choices)
            if aug == 'roll':
                off1 = random.randint(-5, 5)
                off2 = random.randint(-5, 5)
                data = torch.roll(data, shifts=(off1, off2), dims=(2, 3))
            elif aug == 'rotate':
                data = self.rotate(data)

        return data, target.long().squeeze(-1)

    def __len__(self):
        return len(os.listdir(self.root))


def build_dvscifar(path='/mnt/lustre/liyuhang1/data/cifar-dvs', transform=None):
    train_path = path + '/train'
    val_path = path + '/test'
    train_dataset = DVSCifar10(root=train_path, transform=transform)
    val_dataset = DVSCifar10(root=val_path, transform=False)

    return train_dataset, val_dataset


def build_imagenet():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    root = '/data_smr/dataset/ImageNet'
    train_root = os.path.join(root,'train')
    val_root = os.path.join(root,'val')
    train_dataset = ImageFolder(
        train_root,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    )
    val_dataset = ImageFolder(
        val_root,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    )
    return train_dataset, val_dataset




class MFCCPreprocessor:
    """
    Waveform -> MFCC tensor with per-utterance CMVN and fixed time frames.
    Output per sample: [1, 40, 100] (channel, mfcc, time)
    """
    def __init__(self, sample_rate=16000, n_mfcc=40, n_mels=64, fixed_frames=100, cmvn=True):
        self.sr = int(sample_rate)
        self.cmvn = bool(cmvn)
        self.fixed = int(fixed_frames)
        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=self.sr, n_mfcc=n_mfcc,
            melkwargs=dict(
                n_fft=400, hop_length=160, n_mels=n_mels,
                f_min=0.0, f_max=self.sr / 2, center=True,
                power=2.0, window_fn=torch.hann_window
            )
        )

    def __call__(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        if wav.size(0) > 1:
            wav = wav.mean(0, keepdim=True)
        if int(sr) != self.sr:
            wav = torchaudio.functional.resample(wav, int(sr), self.sr)
        x = self.mfcc(wav)
        if x.dim() == 3 and x.size(0) == 1:
            x = x.squeeze(0)  # [n_mfcc, T]

        if self.cmvn:
            m = x.mean(-1, keepdim=True)
            s = x.std(-1, keepdim=True).clamp_min(1e-9)
            x = (x - m) / s

        T = x.shape[-1]
        if T < self.fixed:
            x = torch.nn.functional.pad(x, (0, self.fixed - T))
        elif T > self.fixed:
            x = x[:, :self.fixed]
        return x  # [n_mfcc, fixed]


class SpeechCommandsMFCC(Dataset):
    """
    Speech Commands -> MFCC "images" for SNN.
    Each item: (tensor [1, 40, 100], int_label)
    """
    def __init__(self,data_root: str,selected: Optional[List[str]] = None,fixed_frames: int = 100,n_mfcc: int = 40,n_mels: int = 64,subset: str = "training", 
        noise_augmentor=None):
        super().__init__()
        self.base = torchaudio.datasets.SPEECHCOMMANDS(root=data_root, download=False, subset=subset)

        self.selected = selected or [
            "yes", "no", "up", "down", "left", "right", "on", "off",
            "stop", "go"]

        # Collect indices for chosen classes from torchaudio's walker
        class_to_indices: Dict[str, List[int]] = defaultdict(list)
        for i, rel in enumerate(self.base._walker):
            label_folder = os.path.basename(os.path.split(rel)[0])
            if label_folder in self.selected:
                class_to_indices[label_folder].append(i)

        self.indices: List[int] = [i for _, idxs in class_to_indices.items() for i in idxs]
        self.classes: List[str] = sorted(class_to_indices.keys())
        self.label2idx: Dict[str, int] = {c: j for j, c in enumerate(self.classes)}
        self.num_classes: int = len(self.classes)

        self.noise_augmentor = noise_augmentor
        target_sr = getattr(noise_augmentor, "target_sr", 16000) if noise_augmentor is not None else 16000
        self.pre = MFCCPreprocessor(
            sample_rate=target_sr, n_mfcc=n_mfcc, n_mels=n_mels,
            fixed_frames=fixed_frames, cmvn=True
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, k: int) -> Tuple[torch.Tensor, int]:
        rel = self.base._walker[self.indices[k]]
        path = os.path.join(self.base._path, rel)
        wav, sr = safe_load_wav(path)

        if self.noise_augmentor is not None:
            wav, sr = self.noise_augmentor(wav, sr)  # should return (wav, sr)

        label_name = os.path.basename(os.path.split(rel)[0])
        mfcc = self.pre(wav, int(sr)).contiguous().clone()  # [40, fixed]
        return mfcc.unsqueeze(0), self.label2idx[label_name]  # [1, 40, fixed], y


def build_speechcommands(data_root: str,selected_classes: Optional[List[str]] = None,fixed_frames: int = 100,n_mfcc: int = 40,n_mels: int = 64,noise_for_train=None,):
    train_ds = SpeechCommandsMFCC(data_root=data_root,selected=selected_classes,fixed_frames=fixed_frames,n_mfcc=n_mfcc,n_mels=n_mels,subset="training",
        noise_augmentor=noise_for_train,
    )
    test_ds = SpeechCommandsMFCC(data_root=data_root,selected=selected_classes,fixed_frames=fixed_frames,n_mfcc=n_mfcc,n_mels=n_mels,subset="testing",
        noise_augmentor=None, 
    )
    return train_ds, test_ds
