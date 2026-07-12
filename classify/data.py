import os
import random
import re
from os.path import expanduser
from os.path import join as ospj
import json
import pickle
import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, ConcatDataset, Sampler
import torchvision as tv
from collections import defaultdict
import warnings

from utils import make_dirs
from util_data import (
    SUBSET_NAMES,
    configure_metadata, get_image_ids, get_class_labels,
    GaussianBlur, Solarization,
)

NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)
CLIP_NORM_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_NORM_STD = (0.26862954, 0.26130258, 0.27577711)

# 从torchvision导入必要工具
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS

class FixedLabelImageFolder(Dataset):
    """强制按给定的class_names顺序分配标签索引的数据集类"""
    def __init__(self, root, transform, class_names, descriptor_cache=None):
        self.root = root
        self.transform = transform
        self.class_names = class_names  # 传入SUBSET_NAMES
        self.class_to_idx = {name: i for i, name in enumerate(class_names)}  # 强制映射
        self.descriptor_cache = descriptor_cache  # ROI 描述符缓存，默认关闭以保持兼容

        # 收集所有样本路径和标签
        self.samples = []
        for cls_name in class_names:
            cls_dir = os.path.join(root, cls_name)
            if not os.path.isdir(cls_dir):
                warnings.warn(f"警告：未找到类别文件夹 {cls_dir}，将跳过该类别")
                continue
            # 遍历文件夹中的图片文件
            # 原始逻辑：for img_name in os.listdir(cls_dir):
            # 排序后加载，保证相同随机种子下样本索引完全可复现。
            for img_name in sorted(os.listdir(cls_dir)):
                if any(img_name.endswith(ext) for ext in IMG_EXTENSIONS):
                    img_path = os.path.join(cls_dir, img_name)
                    self.samples.append((img_path, self.class_to_idx[cls_name]))

        if len(self.samples) == 0:
            raise RuntimeError("未找到任何有效图片样本！")

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = default_loader(img_path).convert('RGB')  # 加载并转为RGB
        if self.transform is not None:
            image = self.transform(image)
        if self.descriptor_cache is not None:
            desc = self.descriptor_cache.get(img_path)
            if desc is not None:
                desc_tensor = torch.tensor(desc, dtype=torch.float32)
            else:
                desc_tensor = torch.full((4,), float('nan'))
            return image, label, desc_tensor
        return image, label

    def __len__(self):
        return len(self.samples)

HAND_SUBJECT_PATTERN = re.compile(r"^(?P<subject>\d+)_(?P<pose>01|02)(?:$|[_\-.])")


def filter_hand_dataset_by_pose(dataset, hand_pose):
    """按文件名姿势编号过滤手部数据；训练集和测试集必须共用该函数。"""
    if hand_pose not in {"all", "01", "02"}:
        raise ValueError(f"hand_pose必须为all、01或02，实际为：{hand_pose}")
    if hand_pose == "all":
        return dataset

    filtered_samples = []
    for path, label in dataset.samples:
        match = HAND_SUBJECT_PATTERN.match(os.path.basename(path))
        if not match:
            raise ValueError(f"无法从文件名解析手部受试者ID和姿势：{path}")
        if match.group("pose") == hand_pose:
            filtered_samples.append((path, label))
    if not filtered_samples:
        raise ValueError(f"hand_pose={hand_pose}过滤后没有任何手部图片")

    dataset.samples = filtered_samples
    # FixedLabelImageFolder主要使用samples；同时维护targets以兼容通用数据工具。
    dataset.targets = [label for _, label in filtered_samples]
    print(f"[手部姿势过滤] hand_pose={hand_pose}, images={len(filtered_samples)}")
    return dataset


class PadToSquare:
    """用白色补边扩展成正方形，避免中心裁剪切掉两侧手部。"""

    def __init__(self, fill=(255, 255, 255)):
        self.fill = fill

    def __call__(self, image):
        width, height = image.size
        size = max(width, height)
        left = (size - width) // 2
        right = size - width - left
        top = (size - height) // 2
        bottom = size - height - top
        return ImageOps.expand(
            image, border=(left, top, right, bottom), fill=self.fill
        )


def get_hand_transforms(model_type):
    """返回手部专用的轻量训练增强和确定性测试预处理。"""
    if model_type == "clip":
        norm_mean, norm_std = CLIP_NORM_MEAN, CLIP_NORM_STD
    elif model_type == "resnet50":
        norm_mean, norm_std = NORM_MEAN, NORM_STD
    else:
        raise ValueError(f"不支持的模型类型：{model_type}")

    train_transform = tv.transforms.Compose([
        tv.transforms.Lambda(lambda x: x.convert("RGB")),
        PadToSquare(fill=(255, 255, 255)),
        tv.transforms.Resize(
            (224, 224), interpolation=tv.transforms.InterpolationMode.BICUBIC
        ),
        tv.transforms.RandomHorizontalFlip(p=0.5),
        tv.transforms.RandomAffine(
            degrees=5,
            translate=(0.03, 0.03),
            scale=(0.95, 1.05),
            interpolation=tv.transforms.InterpolationMode.BICUBIC,
            fill=255,
        ),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(norm_mean, norm_std),
    ])
    test_transform = tv.transforms.Compose([
        tv.transforms.Lambda(lambda x: x.convert("RGB")),
        PadToSquare(fill=(255, 255, 255)),
        tv.transforms.Resize(
            (224, 224), interpolation=tv.transforms.InterpolationMode.BICUBIC
        ),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(norm_mean, norm_std),
    ])
    return train_transform, test_transform


class HandSubjectBalancedSampler(Sampler):
    """每轮使用全部12名营养不良者，并随机抽取12名正常者。"""

    def __init__(self, dataset, subjects_per_class=12, seed=22, history_path=None, hand_pose="all"):
        self.dataset = dataset
        self.subjects_per_class = subjects_per_class
        self.seed = seed
        self.epoch = 0
        self.history_path = history_path
        self.history = {}
        self.hand_pose = hand_pose
        self.images_per_subject = 1 if hand_pose in {"01", "02"} else 2

        required_classes = {"malnourished_hand", "normal_hand"}
        if set(dataset.class_to_idx) != required_classes:
            raise ValueError("手部采样要求类别恰好为 malnourished_hand 和 normal_hand")
        self.mal_label = dataset.class_to_idx["malnourished_hand"]
        self.normal_label = dataset.class_to_idx["normal_hand"]
        self.subject_indices = {self.mal_label: {}, self.normal_label: {}}

        for index, (path, label) in enumerate(dataset.samples):
            match = HAND_SUBJECT_PATTERN.match(os.path.basename(path))
            if not match:
                raise ValueError(f"无法从文件名解析手部受试者ID和姿势：{path}")
            subject_id = match.group("subject")
            self.subject_indices[label].setdefault(subject_id, []).append(index)

        for label, groups in self.subject_indices.items():
            incomplete = {
                subject_id: len(indices)
                for subject_id, indices in groups.items()
                if len(indices) != self.images_per_subject
            }
            if incomplete:
                raise ValueError(
                    f"hand_pose={self.hand_pose}时每名受试者应有"
                    f"{self.images_per_subject}张图片：label={label}, {incomplete}"
                )

        self.mal_subjects = sorted(self.subject_indices[self.mal_label], key=int)
        self.normal_subjects = sorted(self.subject_indices[self.normal_label], key=int)
        if len(self.mal_subjects) != self.subjects_per_class:
            raise ValueError(
                f"营养不良训练者应为{self.subjects_per_class}人，"
                f"实际为{len(self.mal_subjects)}人"
            )
        if len(self.normal_subjects) < self.subjects_per_class:
            raise ValueError("正常训练者数量不足，无法进行平衡抽样")

    def set_epoch(self, epoch):
        """显式设置轮次，使本轮随机种子严格等于 base_seed + epoch。"""
        self.epoch = int(epoch)

    def __len__(self):
        # 2个类别 × 每类受试者数 × 当前姿势对应的每人图片数。
        return self.subjects_per_class * 2 * self.images_per_subject

    def __iter__(self):

        # 动态抽样
        rng = random.Random(self.seed + self.epoch)
        selected_normal = sorted(
            rng.sample(self.normal_subjects, self.subjects_per_class), key=int
        )
        indices = []
        for subject_id in self.mal_subjects:
            indices.extend(self.subject_indices[self.mal_label][subject_id])
        for subject_id in selected_normal:
            indices.extend(self.subject_indices[self.normal_label][subject_id])
        rng.shuffle(indices)

        # 固定抽样
        # # 固定选择12名正常受试者：所有epoch都由基础seed决定
        # selection_rng = random.Random(self.seed)
        # selected_normal = sorted(
        #     selection_rng.sample(self.normal_subjects, self.subjects_per_class),
        #     key=int,
        # )
        # # 每个epoch仍采用不同顺序训练
        # shuffle_rng = random.Random(self.seed + self.epoch)
        # indices = []
        # for subject_id in self.mal_subjects:
        #     indices.extend(self.subject_indices[self.mal_label][subject_id])
        # for subject_id in selected_normal:
        #     indices.extend(self.subject_indices[self.normal_label][subject_id])
        # shuffle_rng.shuffle(indices)

        # 动态抽样
        record = {
            "epoch": self.epoch,
            "seed": self.seed + self.epoch,
            "malnourished_subject_ids": self.mal_subjects,
            "normal_subject_ids": selected_normal,
            "subject_count_per_class": self.subjects_per_class,
            "hand_pose": self.hand_pose,
            "image_count_per_class": self.subjects_per_class * self.images_per_subject,
        }
        # 固定抽样
        # record = {
        #     "epoch": self.epoch,
        #     "selection_seed": self.seed,
        #     "shuffle_seed": self.seed + self.epoch,
        #     "sampling_mode": "fixed",
        #     "malnourished_subject_ids": self.mal_subjects,
        #     "normal_subject_ids": selected_normal,
        #     "subject_count_per_class": self.subjects_per_class,
        #     "image_count_per_class": self.subjects_per_class * 2,
        # }
        self.history[str(self.epoch)] = record
        # 动态抽样
        print(
            f"[手部动态采样] epoch={self.epoch}, seed={self.seed + self.epoch}, "
            f"normal_subject_ids={selected_normal}"
        )
        # 固定抽样
        # print(
        #     f"[手部固定采样] epoch={self.epoch}, "
        #     f"selection_seed={self.seed}, shuffle_seed={self.seed + self.epoch}, "
        #     f"normal_subject_ids={selected_normal}"
        # )
        if self.history_path:
            history_path = os.path.abspath(self.history_path)
            os.makedirs(os.path.dirname(history_path), exist_ok=True)
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
        return iter(indices)


def get_transforms(model_type):
    if model_type == "clip":
        norm_mean = CLIP_NORM_MEAN
        norm_std = CLIP_NORM_STD
    elif model_type == "resnet50":
        norm_mean = NORM_MEAN
        norm_std = NORM_STD

    aux_transform = tv.transforms.Compose([
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.RandomApply(
            [
                tv.transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                )
            ],
            p=0.8,
        ),
        tv.transforms.RandomGrayscale(p=0.2),
        GaussianBlur(0.2),
        Solarization(0.2),
    ])
    train_transform = tv.transforms.Compose([
        tv.transforms.Lambda(lambda x: x.convert("RGB")),
        tv.transforms.RandAugment(),
        tv.transforms.RandomResizedCrop(
            224,
            scale=(0.25, 1.0),
            interpolation=tv.transforms.InterpolationMode.BICUBIC,
            antialias=None,
        ),
        aux_transform,
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(norm_mean, norm_std)
    ])

    test_transform = tv.transforms.Compose([
        tv.transforms.Lambda(lambda x: x.convert("RGB")),
        tv.transforms.Resize(
            224,
            interpolation=tv.transforms.functional.InterpolationMode.BICUBIC
        ),
        tv.transforms.CenterCrop(224),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(norm_mean, norm_std)
    ])

    return train_transform, test_transform


class ImageNetDatasetFromMetadata(Dataset):
    def __init__(
        self,
        data_root,
        metadata_root,
        transform,
        proxy,
        target_label=None,
        n_img_per_cls=None,
        # 原始默认值：dataset="my_dataset"
        dataset="hand_nutrition",
        n_shot=0,
        real_train_fewshot_data_dir='',
        is_pooled_fewshot=False,
    ):
        self.data_root = data_root
        self.metadata = configure_metadata(metadata_root)
        self.transform = transform
        self.image_ids = get_image_ids(self.metadata, proxy=proxy)
        self.image_labels = get_class_labels(self.metadata)
        self.is_pooled_fewshot = is_pooled_fewshot

        if not is_pooled_fewshot:
            """ full data """
            if n_img_per_cls is not None:
                value_counts = defaultdict(int)
                tmp = {}
                for k, v in self.image_labels.items():
                    if value_counts[v] < n_img_per_cls:
                        tmp[k] = v
                        value_counts[v] += 1
                self.image_labels = tmp

            if target_label is not None:
                self.image_labels = {k: v for k, v in self.image_labels.items()
                                     if v == target_label}

            self.image_ids = list(self.image_labels.keys())

        else:
            """ only fewshot data """
            self.image_paths = []
            self.image_labels = []
            reps = round(n_img_per_cls // n_shot)
            for label, class_name in enumerate(SUBSET_NAMES[dataset]):
                real_img_paths = sorted(os.listdir(
                    ospj(real_train_fewshot_data_dir, class_name)))
                real_subset = [
                    ospj(
                        real_train_fewshot_data_dir,
                        class_name,
                        real_img_paths[i]
                    ) for i in range(n_shot)
                ]
                for i in range(reps):
                    self.image_paths.extend(real_subset)
                    self.image_labels.extend([label] * n_shot)

    def get_data(self, fpath):
        x = Image.open(fpath)
        x = x.convert('RGB')
        return x

    def __getitem__(self, idx):
        if not self.is_pooled_fewshot: # full data
            image_id = self.image_ids[idx]
            image = self.get_data(ospj(self.data_root, image_id))
            image_label = self.image_labels[image_id]
        else: # few-shot
            image_id = self.image_paths[idx]
            image = self.get_data(self.image_paths[idx])
            image_label = self.image_labels[idx]
        image = self.transform(image)
        return image, image_label

    def __len__(self):
        if not self.is_pooled_fewshot:
            return len(self.image_ids)
        else:
            return len(self.image_paths)


class DatasetSynthImage(Dataset):
    def __init__(
        self,
        synth_train_data_dir,
        transform,
        target_label=None,
        n_img_per_cls=None,
        # 原始默认值：dataset='my_dataset'
        dataset='hand_nutrition',
        n_shot=0,
        real_train_fewshot_data_dir='',
        is_pooled_fewshot=False,
        descriptor_cache=None,
        **kwargs
    ):
        self.synth_train_data_dir = synth_train_data_dir
        self.transform = transform
        self.is_pooled_fewshot = is_pooled_fewshot
        self.descriptor_cache = descriptor_cache

        self.image_paths = []
        self.image_labels = []
        self.is_real_flags = []  # 0 for synth, 1 for real (few-shot pooled)

        value_counts = defaultdict(int)
        for label, class_name in enumerate(SUBSET_NAMES[dataset]):
            if target_label is not None and label != target_label:
                continue
            for fname in os.listdir(ospj(synth_train_data_dir, class_name)):
                if fname.endswith((".txt", ".json")):
                    continue
                if n_img_per_cls is not None and value_counts[label] >= n_img_per_cls:
                    continue
                self.image_paths.append(ospj(synth_train_data_dir, class_name, fname))
                self.image_labels.append(label)
                self.is_real_flags.append(0)
                value_counts[label] += 1

        if is_pooled_fewshot:
            if n_shot == 0:
                n_shot = 16
            reps = round(n_img_per_cls // n_shot)
            for label, class_name in enumerate(SUBSET_NAMES[dataset]):
                real_img_paths = os.listdir(ospj(real_train_fewshot_data_dir, class_name))
                real_subset = [
                    ospj(real_train_fewshot_data_dir, class_name, real_img_paths[i])
                    for i in range(n_shot)
                ]
                for i in range(reps):
                    self.image_paths.extend(real_subset)
                    self.image_labels.extend([label] * n_shot)
                    self.is_real_flags.extend([1] * n_shot)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image_label = self.image_labels[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        is_real = int(self.is_real_flags[idx])

        if self.descriptor_cache is not None:
            # 描述符缺失时用 NaN 标记，训练时只跳过辅助损失。
            desc = self.descriptor_cache.get(image_path)
            if desc is not None:
                desc_tensor = torch.tensor(desc, dtype=torch.float32)
            else:
                desc_tensor = torch.full((4,), float('nan'))
            if self.is_pooled_fewshot:
                return image, image_label, is_real, desc_tensor
            return image, image_label, desc_tensor

        if self.is_pooled_fewshot:
            return image, image_label, is_real
        else:
            return image, image_label

    def __len__(self):
        return len(self.image_paths)


def filter_dset(dataset, n_img_per_cls, dataset_name):
    import random
    print(n_img_per_cls)
    if dataset_name == 'pets':
        _images = dataset._images
        _labels = dataset._labels
    elif dataset_name == 'stl10':
        _images = dataset.data
        _labels = dataset.labels
    elif dataset_name in ('food101', 'fgvc_aircraft', 'dtd', 'flowers102', 'sun397'):
        _images = dataset._image_files
        _labels = dataset._labels
    elif dataset_name == 'eurosat':
        _images = [sample[0] for sample in dataset.samples]
        _labels = [sample[1] for sample in dataset.samples]
    elif dataset_name == 'cars':
        _images = [sample[0] for sample in dataset._samples]
        _labels = [sample[1] for sample in dataset._samples]
    elif dataset_name == 'caltech101':
        _images = dataset.index
        _labels = dataset.y
    else:
        raise ValueError("Please specify valid dataset.")

    new_images = []
    new_labels = []
    for i in set(_labels):
        candidates = [j for j, lab in enumerate(_labels) if lab == i]
        img_per_cls = min(n_img_per_cls, len(candidates))
        idx = random.sample(range(len(candidates)), img_per_cls)
        new_images.extend([_images[candidates[j]] for j in idx])
        new_labels.extend([_labels[candidates[j]] for j in idx])

    if dataset_name == 'pets':
        dataset._images = new_images
        dataset._labels = new_labels
    elif dataset_name == 'stl10':
        dataset.data = np.asarray(new_images)
        dataset.labels = np.asarray(new_labels)
    elif dataset_name in ('food101', 'fgvc_aircraft', 'dtd', 'flowers102', 'sun397'):
        dataset._image_files = new_images
        dataset._labels = new_labels
    elif dataset_name == 'eurosat':
        dataset.samples = list(zip(new_images, new_labels))
        dataset.targets = new_labels
    elif dataset_name == 'cars':
        dataset._samples = list(zip(new_images, new_labels))
    elif dataset_name == 'caltech101':
        dataset.index = new_images
        dataset.y = new_labels
    else:
        raise ValueError("Please specify valid dataset.")
    return dataset


# 以下为各类数据集的split函数（保持不变）
def split_eurosat(file_path, split, dataset):
    split_file_path = os.path.join(file_path, 'split_zhou_EuroSAT.json')
    if not os.path.exists(split_file_path):
        raise ValueError("Please download split_zhou_EuroSAT.json into the dataset directory.")
    with open(split_file_path) as f:
        split_files = json.load(f)
    data = [os.path.join(file_path, 'eurosat', '2750', path[0]) for path in split_files[split]]
    dataset.samples = [s for s in dataset.samples if s[0] in data]
    dataset.labels = [s[1] for s in dataset.samples]
    return dataset


def split_sun(file_path, split, dataset):
    import csv
    split_file_path = os.path.join(file_path, 'split_coop.csv')
    split_files = []
    with open(split_file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['split'] == split:
                split_files.append(row['filename'])
    file_path_full = os.path.join(file_path, 'SUN397') + '/'
    ind_to_keep = [i for i, f in enumerate(dataset._image_files) if str(f).replace(file_path_full, '') in split_files]
    dataset._image_files = [dataset._image_files[i] for i in ind_to_keep]
    dataset._labels = [dataset._labels[i] for i in ind_to_keep]
    return dataset


def split_caltech(file_path, split, dataset):
    import csv
    split_file_path = os.path.join(file_path, 'split_coop.csv')
    split_files = []
    with open(split_file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['split'] == split:
                split_files.append(row['filename'])
    ind_to_keep = [i for i in range(len(dataset.index)) if
                   os.path.join(dataset.categories[dataset.y[i]],
                                f'image_{dataset.index[i]:04d}.jpg') in split_files]
    dataset.index = [dataset.index[i] for i in ind_to_keep]
    dataset.y = [dataset.y[i] for i in ind_to_keep]
    dataset.y = [i if i < 1 else i - 1 for i in dataset.y]
    dataset.categories.remove("Faces_easy")
    dataset.annotation_categories.remove("Faces_3")
    return dataset


def split_dtd(real_train_data_dir, train_transform, split):
    import csv
    dtd_path_train = os.path.join(real_train_data_dir, 'train')
    train_dataset = tv.datasets.DTD(root=dtd_path_train, split='train', transform=train_transform, download=True)
    val_dataset = tv.datasets.DTD(root=dtd_path_train, split='val', transform=train_transform, download=True)
    test_dataset = tv.datasets.DTD(root=dtd_path_train, split='test', transform=train_transform, download=True)
    train_dataset._image_files = train_dataset._image_files + val_dataset._image_files + test_dataset._image_files
    train_dataset._labels = train_dataset._labels + val_dataset._labels + test_dataset._labels

    split_file_path = os.path.join(dtd_path_train, 'split_coop.csv')
    split_files = []
    with open(split_file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['split'] == split:
                split_files.append(row['filename'])
    file_path_full = os.path.join(dtd_path_train, 'dtd', 'dtd', 'images') + '/'
    ind_to_keep = [i for i, f in enumerate(train_dataset._image_files) if str(f).replace(file_path_full, '') in split_files]
    train_dataset._image_files = [train_dataset._image_files[i] for i in ind_to_keep]
    train_dataset._labels = [train_dataset._labels[i] for i in ind_to_keep]
    return train_dataset


def split_flowers(real_train_data_dir, train_transform, split):
    import csv
    flowers_path_train = os.path.join(real_train_data_dir, 'train')
    train_dataset = tv.datasets.Flowers102(root=flowers_path_train, split='train', transform=train_transform, download=True)
    val_dataset = tv.datasets.Flowers102(root=flowers_path_train, split='val', transform=train_transform, download=True)
    test_dataset = tv.datasets.Flowers102(root=flowers_path_train, split='test', transform=train_transform, download=True)
    train_dataset._image_files = train_dataset._image_files + val_dataset._image_files + test_dataset._image_files
    train_dataset._labels = train_dataset._labels + val_dataset._labels + test_dataset._labels

    split_file_path = os.path.join(flowers_path_train, 'split_coop.csv')
    split_files = []
    with open(split_file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['split'] == split:
                split_files.append(row['filename'])
    file_path_full = os.path.join(flowers_path_train, 'flowers-102', 'jpg') + '/'
    ind_to_keep = [i for i, f in enumerate(train_dataset._image_files) if str(f).replace(file_path_full, '') in split_files]
    train_dataset._image_files = [train_dataset._image_files[i] for i in ind_to_keep]
    train_dataset._labels = [train_dataset._labels[i] for i in ind_to_keep]
    return train_dataset


def split_food(real_train_data_dir, train_transform, split):
    import csv
    food_path_train = os.path.join(real_train_data_dir, 'train')
    train_dataset = tv.datasets.Food101(root=food_path_train, split='train', transform=train_transform, download=True)
    test_dataset = tv.datasets.Food101(root=food_path_train, split='test', transform=train_transform, download=True)
    train_dataset._image_files = train_dataset._image_files + test_dataset._image_files
    train_dataset._labels = train_dataset._labels + test_dataset._labels

    split_file_path = os.path.join(food_path_train, 'split_coop.csv')
    split_files = []
    with open(split_file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['split'] == split:
                split_files.append(row['filename'])
    file_path_full = os.path.join(food_path_train, 'food-101', 'images') + '/'
    ind_to_keep = [i for i, f in enumerate(train_dataset._image_files) if str(f).replace(file_path_full, '') in split_files]
    train_dataset._image_files = [train_dataset._image_files[i] for i in ind_to_keep]
    train_dataset._labels = [train_dataset._labels[i] for i in ind_to_keep]
    return train_dataset


def split_pets(real_train_data_dir, train_transform, split):
    import csv
    pets_path_train = os.path.join(real_train_data_dir, 'train')
    train_dataset = tv.datasets.OxfordIIITPet(root=pets_path_train, split='trainval', target_types='category', transform=train_transform, download=True)
    test_dataset = tv.datasets.OxfordIIITPet(root=pets_path_train, split='test', target_types='category', transform=train_transform, download=True)
    train_dataset._images = train_dataset._images + test_dataset._images
    train_dataset._labels = train_dataset._labels + test_dataset._labels

    split_file_path = os.path.join(pets_path_train, 'split_coop.csv')
    split_files = []
    with open(split_file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['split'] == split:
                split_files.append(row['filename'].split('/')[-1])
    file_path_full = os.path.join(pets_path_train, 'oxford-iiit-pet', 'images') + '/'
    ind_to_keep = [i for i, f in enumerate(train_dataset._images) if str(f).replace(file_path_full, '') in split_files]
    train_dataset._images = [train_dataset._images[i] for i in ind_to_keep]
    train_dataset._labels = [train_dataset._labels[i] for i in ind_to_keep]
    return train_dataset


def get_data_loader(
    real_train_data_dir="",
    real_test_data_dir="",
    metadata_dir="metadata",
    # 原始默认值：dataset="my_dataset"
    dataset="hand_nutrition",
    bs=32,
    eval_bs=32,
    is_rand_aug=True,
    target_label=None,
    n_img_per_cls=None,
    is_synth_train=False,
    n_shot=0,
    real_train_fewshot_data_dir='',
    is_pooled_fewshot=False,
    model_type=None,
    descriptor_cache=None,
    use_hand_transforms=False,
    hand_pose="all",
    is_hand_subject_balanced=False,
    hand_subjects_per_class=12,
    sampling_seed=22,
    sampling_history_path=None,
):
    # 原始逻辑：train_transform, test_transform = get_transforms(model_type)
    # 仅手部入口启用方形补边和轻量增强，其他实验保持原预处理。
    if use_hand_transforms:
        train_transform, test_transform = get_hand_transforms(model_type)
    else:
        train_transform, test_transform = get_transforms(model_type)
    train_loader = None
    test_dataset = None  # 初始化test_dataset，避免作用域错误

    # ====================== 训练集处理 ======================
    if not is_synth_train:
        if dataset == 'imagenet':
            train_dataset = ImageNetDatasetFromMetadata(
                data_root=real_train_data_dir,
                metadata_root=ospj(metadata_dir, 'train'),
                transform=train_transform if is_rand_aug else test_transform,
                proxy=False,
                target_label=target_label,
                n_img_per_cls=n_img_per_cls,
                dataset=dataset,
                n_shot=n_shot,
                real_train_fewshot_data_dir=real_train_fewshot_data_dir,
                is_pooled_fewshot=is_pooled_fewshot,
            )
        elif dataset == 'pets':
            train_dataset = split_pets(real_train_data_dir, train_transform, 'train')
            train_dataset = filter_dset(dataset=train_dataset, n_img_per_cls=n_img_per_cls, dataset_name=dataset)
        elif dataset == 'stl10':
            stl10_path_train = os.path.join(real_train_data_dir, 'train')
            train_dataset = tv.datasets.STL10(
                root=stl10_path_train, split='train', download=True, transform=train_transform
            )
            train_dataset = filter_dset(dataset=train_dataset, n_img_per_cls=n_img_per_cls, dataset_name=dataset)
        elif dataset == 'food101':
            train_dataset = split_food(real_train_data_dir, train_transform, 'train')
            train_dataset = filter_dset(dataset=train_dataset, n_img_per_cls=n_img_per_cls, dataset_name=dataset)
        elif dataset == 'fgvc_aircraft':
            aircraft_path_train = os.path.join(real_train_data_dir, 'train')
            train_dataset = tv.datasets.FGVCAircraft(
                root=aircraft_path_train, split='trainval', annotation_level='variant',
                transform=train_transform, download=True
            )
            train_dataset = filter_dset(dataset=train_dataset, n_img_per_cls=n_img_per_cls, dataset_name=dataset)
        elif dataset == 'eurosat':
            eurosat_path_train = os.path.join(real_train_data_dir, 'train')
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context
            train_dataset = tv.datasets.EuroSAT(
                root=eurosat_path_train, transform=train_transform, download=True
            )
            train_dataset = split_eurosat(eurosat_path_train, 'train', train_dataset)
            train_dataset = filter_dset(dataset=train_dataset, n_img_per_cls=n_img_per_cls, dataset_name=dataset)
        elif dataset == 'cars':
            cars_path_train = os.path.join(real_train_data_dir, 'train')
            train_dataset = tv.datasets.StanfordCars(
                root=cars_path_train, split='train', transform=train_transform, download=False
            )
            train_dataset = filter_dset(dataset=train_dataset, n_img_per_cls=n_img_per_cls, dataset_name=dataset)
        elif dataset == 'dtd':
            train_dataset = split_dtd(real_train_data_dir, train_transform, 'train')
            train_dataset = filter_dset(dataset=train_dataset, n_img_per_cls=n_img_per_cls, dataset_name=dataset)
        elif dataset == 'flowers102':
            train_dataset = split_flowers(real_train_data_dir, train_transform, 'train')
            train_dataset = filter_dset(dataset=train_dataset, n_img_per_cls=n_img_per_cls, dataset_name=dataset)
        elif dataset == 'sun397':
            sun_path_train = os.path.join(real_train_data_dir, 'train')
            train_dataset = tv.datasets.SUN397(
                root=sun_path_train, transform=train_transform, download=True
            )
            train_dataset = split_sun(sun_path_train, 'train', train_dataset)
            train_dataset = filter_dset(dataset=train_dataset, n_img_per_cls=n_img_per_cls, dataset_name=dataset)
        elif dataset == 'caltech101':
            caltech_path_train = os.path.join(real_train_data_dir, 'train')
            train_dataset = tv.datasets.Caltech101(
                root=caltech_path_train, transform=train_transform, download=True
            )
            train_dataset = split_caltech(caltech_path_train, 'train', train_dataset)
            train_dataset = filter_dset(dataset=train_dataset, n_img_per_cls=n_img_per_cls, dataset_name=dataset)
        # 原始自定义数据分支：elif dataset == 'my_dataset':
        elif dataset == 'hand_nutrition':
            # 训练集：强制按SUBSET_NAMES映射标签
            subset_names = SUBSET_NAMES[dataset]
            train_dataset = FixedLabelImageFolder(
                root=real_train_data_dir,
                transform=train_transform if is_rand_aug else test_transform,
                class_names=subset_names,
                descriptor_cache=descriptor_cache,
            )
            train_dataset = filter_hand_dataset_by_pose(train_dataset, hand_pose)
            # 打印训练集标签映射
            print("\n===== 训练集标签映射 =====")
            for idx, cls_name in enumerate(subset_names):
                print(f"标签索引 {idx} -> 类别名称: {cls_name}")

            # 小样本过滤
            if n_img_per_cls:
                class_counts = defaultdict(int)
                filtered_samples = []
                for path, label in train_dataset.samples:
                    if class_counts[label] < n_img_per_cls:
                        filtered_samples.append((path, label))
                        class_counts[label] += 1
                train_dataset.samples = filtered_samples
                train_dataset.targets = [s[1] for s in filtered_samples]

            # 混合合成数据（若需要）
            # 若指定了n_img_per_cls对应args.NIPC(例如500), 则每个类别只取前n_img_per_cls张图片
            if is_synth_train:
                synth_dataset = DatasetSynthImage(
                    synth_train_data_dir=real_train_data_dir,  # 根据实际路径调整
                    transform=train_transform,
                    dataset=dataset,
                    n_img_per_cls=n_img_per_cls,
                    descriptor_cache=descriptor_cache,
                )
                train_dataset = ConcatDataset([train_dataset, synth_dataset])
        else:
            raise ValueError("Please specify a valid dataset.")

        hand_sampler = None
        if is_hand_subject_balanced:
            if not isinstance(train_dataset, FixedLabelImageFolder):
                raise TypeError("手部动态平衡采样仅支持 FixedLabelImageFolder")
            hand_sampler = HandSubjectBalancedSampler(
                dataset=train_dataset,
                subjects_per_class=hand_subjects_per_class,
                seed=sampling_seed,
                history_path=sampling_history_path,
                hand_pose=hand_pose,
            )

        # 原始训练 DataLoader 逻辑保留如下：
        # train_loader = torch.utils.data.DataLoader(
        #     train_dataset, batch_size=bs,
        #     shuffle=is_rand_aug,
        #     prefetch_factor=4, pin_memory=True,
        #     num_workers=16
        # )
        # 动态采样器负责可复现乱序，因此启用时关闭 DataLoader 自带 shuffle。
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=bs,
            sampler=hand_sampler,
            shuffle=is_rand_aug and hand_sampler is None,
            prefetch_factor=4, pin_memory=True,
            num_workers=16
        )

    # ====================== 测试集处理 ======================
    if dataset == 'imagenet':
        test_dataset = ImageNetDatasetFromMetadata(
            data_root=real_test_data_dir,
            metadata_root=ospj(metadata_dir, 'test'),
            transform=test_transform,
            proxy=False,
            dataset=dataset,
        )
    elif dataset == 'pets':
        test_dataset = split_pets(real_train_data_dir, test_transform, 'test')
    elif dataset == 'stl10':
        stl10_path_test = os.path.join(real_train_data_dir, 'train')
        test_dataset = tv.datasets.STL10(
            root=stl10_path_test, split='test', download=True, transform=test_transform
        )
    elif dataset == 'food101':
        test_dataset = split_food(real_train_data_dir, test_transform, 'test')
    elif dataset == 'fgvc_aircraft':
        aircraft_path_test = os.path.join(real_train_data_dir, 'train')
        test_dataset = tv.datasets.FGVCAircraft(
            root=aircraft_path_test, split='test', annotation_level='variant',
            transform=test_transform, download=True
        )
    elif dataset == 'eurosat':
        eurosat_path_test = os.path.join(real_train_data_dir, 'train')
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        test_dataset = tv.datasets.EuroSAT(
            root=eurosat_path_test, transform=test_transform, download=True
        )
        test_dataset = split_eurosat(eurosat_path_test, 'test', test_dataset)
    elif dataset == 'cars':
        cars_path_test = os.path.join(real_train_data_dir, 'train')
        test_dataset = tv.datasets.StanfordCars(
            root=cars_path_test, split='test', transform=test_transform, download=False
        )
    elif dataset == 'dtd':
        test_dataset = split_dtd(real_train_data_dir, test_transform, 'test')
    elif dataset == 'flowers102':
        test_dataset = split_flowers(real_train_data_dir, test_transform, 'test')
    elif dataset == 'sun397':
        sun_path_test = os.path.join(real_train_data_dir, 'train')
        test_dataset = tv.datasets.SUN397(
            root=sun_path_test, transform=test_transform, download=True
        )
        test_dataset = split_sun(sun_path_test, 'test', test_dataset)
    elif dataset == 'caltech101':
        caltech_path_test = os.path.join(real_train_data_dir, 'train')
        test_dataset = tv.datasets.Caltech101(
            root=caltech_path_test, transform=test_transform, download=True
        )
        test_dataset = split_caltech(caltech_path_test, 'test', test_dataset)
    # 原始自定义数据分支：elif dataset == 'my_dataset':
    elif dataset == 'hand_nutrition':
        # 测试集：强制按SUBSET_NAMES映射标签
        subset_names = SUBSET_NAMES[dataset]
        test_dataset = FixedLabelImageFolder(
            root=real_test_data_dir,
            transform=test_transform,
            class_names=subset_names
        )
        test_dataset = filter_hand_dataset_by_pose(test_dataset, hand_pose)
        # 打印测试集标签映射
        print("\n===== 测试集标签映射 =====")
        for idx, cls_name in enumerate(subset_names):
            print(f"标签索引 {idx} -> 类别名称: {cls_name}")

        # 检查缺失类别
        present_classes = set(cls for cls in subset_names if os.path.isdir(os.path.join(real_test_data_dir, cls)))
        missing_classes = set(subset_names) - present_classes
        if missing_classes:
            warnings.warn(f"测试集缺少类别文件夹: {missing_classes}")
    else:
        raise ValueError("Please specify a valid dataset.")

    # 创建测试集DataLoader
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=eval_bs, shuffle=False,
        num_workers=16, pin_memory=True
    )

    return train_loader, test_loader


def get_synth_train_data_loader(
    synth_train_data_dir="data_synth",
    bs=32,
    is_rand_aug=True,
    target_label=None,
    n_img_per_cls=None,
    # 原始默认值：dataset='my_dataset'
    dataset='hand_nutrition',
    n_shot=0,
    real_train_fewshot_data_dir='',
    is_pooled_fewshot=False,
    model_type=None,
    descriptor_cache=None,
):
    train_transform, test_transform = get_transforms(model_type)
    train_dataset = DatasetSynthImage(
        synth_train_data_dir=synth_train_data_dir,
        transform=train_transform if is_rand_aug else test_transform,
        target_label=target_label,
        n_img_per_cls=n_img_per_cls,
        dataset=dataset,
        n_shot=n_shot,
        real_train_fewshot_data_dir=real_train_fewshot_data_dir,
        is_pooled_fewshot=is_pooled_fewshot,
        descriptor_cache=descriptor_cache,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=bs,
        shuffle=is_rand_aug,
        num_workers=16, pin_memory=True,
    )
    return train_loader