import os
import random

import albumentations as albu
import cv2
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

from .data_augumentation import (Compose, Normalize_Tensor, RandomMirror,
                                 RandomRotation, Resize, Scale)


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


seed = 42
seed_everything(seed)


class PolypDataset(data.Dataset):
    """
    dataloader for polyp segmentation tasks
    """

    def __init__(self, image_root, gt_root, trainsize, phase="train"):
        self.trainsize = trainsize
        self.images = [
            image_root + f
            for f in os.listdir(image_root)
            if f.endswith(".jpg") or f.endswith(".png")
        ]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith(".png")]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.filter_files()
        self.size = len(self.images)

        self._alpha = 0.2

        self.phase = phase

        self.transform3 = albu.Compose(
            [
                albu.ShiftScaleRotate(
                    shift_limit=0.15,
                    scale_limit=0.15,
                    rotate_limit=25,
                    p=0.5,
                    border_mode=0,
                ),
                # albu.RandomSizedCrop(),
                albu.ColorJitter(),
                albu.HorizontalFlip(),
                # albu.VerticalFlip()
            ]
        )

        self.img_transform = transforms.Compose(
            [
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.gt_transform = transforms.Compose(
            [transforms.Resize((self.trainsize, self.trainsize)), transforms.ToTensor()]
        )

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        if self.phase == "train":

            image = np.array(image)
            gt = np.array(gt)

            augmented = self.transform3(image=image, mask=gt)

            image, gt = augmented["image"], augmented["mask"]
            image = Image.fromarray(image)
            gt = Image.fromarray(gt)
            image = image.convert("RGB")
            gt = gt.convert("L")

        # image, gt = self.transform(image, gt, phase=self.phase)
        image = self.img_transform(image)

        gt = self.gt_transform(gt)
        return image, gt

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    def binary_loader(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert("L")

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


class PolypDatasetPreprocess(PolypDataset):
    def __init__(self, image_root, gt_root, trainsize, phase="train"):
        super(PolypDatasetPreprocess, self).__init__(
            image_root, gt_root, trainsize, phase
        )

        self.transform3 = albu.Compose(
            [
                albu.ColorJitter(),
            ]
        )

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        name = self.images[index].split("/")[-1]

        if self.phase == "train":

            image = np.array(image)
            gt = np.array(gt)

            augmented = self.transform3(image=image, mask=gt)

            image, gt = augmented["image"], augmented["mask"]
            image = Image.fromarray(image)
            gt = Image.fromarray(gt)
            image = image.convert("RGB")
            gt = gt.convert("L")

        # image, gt = self.transform(image, gt, phase=self.phase)
        image = self.img_transform(image)
        gt = self.gt_transform(gt)

        return image, gt, name


def get_loader(
    image_root,
    gt_root,
    batchsize,
    trainsize,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    phase="train",
    droplast=False,
    dataset="default",
):
    if dataset == "default":
        dataset = PolypDataset(image_root, gt_root, trainsize, phase=phase)
    elif dataset == "preprocess":
        dataset = PolypDatasetPreprocess(image_root, gt_root, trainsize, phase=phase)
    else:
        raise NotImplementedError

    data_loader = data.DataLoader(
        dataset=dataset,
        batch_size=batchsize,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        pin_memory=pin_memory,
        drop_last=droplast,
    )
    return data_loader


class test_dataset:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [
            image_root / f
            for f in os.listdir(image_root)
            if f.endswith(".jpg") or f.endswith(".png") or f.endswith(".tif")
        ]
        self.gts = [
            gt_root / f
            for f in os.listdir(gt_root)
            if f.endswith(".tif") or f.endswith(".png") or f.endswith(".jpg")
        ]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.testsize, self.testsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):

        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split("/")[-1]
        if name.endswith(".jpg"):
            name = name.split(".jpg")[0] + ".png"
        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    def binary_loader(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("L")


class test_dataset_preprocess(test_dataset):
    def __init__(self, image_root, gt_root, testsize):
        super(test_dataset_preprocess, self).__init__(image_root, gt_root, testsize)

        self.images = [
            str(image_root / f)
            for f in os.listdir(image_root)
            if f.endswith(".jpg") or f.endswith(".png") or f.endswith(".tif")
        ]
        self.gts = [
            str(gt_root / f)
            for f in os.listdir(gt_root)
            if f.endswith(".tif") or f.endswith(".png") or f.endswith(".jpg")
        ]

        self.img_transform = transforms.Compose(
            [
                transforms.Resize((self.testsize, self.testsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.gt_transform = transforms.Compose(
            [
                transforms.Resize((self.testsize, self.testsize)),
                transforms.ToTensor(),
            ]
        )

    def load_data(self):

        image = self.rgb_loader(self.images[self.index])
        gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split("/")[-1]

        if name.endswith(".jpg"):
            name = name.split(".jpg")[0] + ".png"

        image = self.img_transform(image)
        gt = self.gt_transform(gt)

        self.index += 1

        return image, gt, name
