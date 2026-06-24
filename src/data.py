"""EuroSAT_RGB 数据加载、划分和小批量迭代工具。"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image


EUROSAT_CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]


@dataclass(frozen=True)
class DatasetSplit:
    """保存某个数据划分的特征、标签和原图路径。

    x 使用展平后的 RGB 像素，形状为 ``[样本数, 64 * 64 * 3]``；
    y 使用类别编号，便于训练时转换为 one-hot 或直接统计准确率。
    """

    x: np.ndarray
    y: np.ndarray
    paths: list[str]


@dataclass(frozen=True)
class EuroSATData:
    """完整数据划分结果。"""

    train: DatasetSplit
    val: DatasetSplit
    test: DatasetSplit
    class_names: list[str]
    image_shape: tuple[int, int, int]


def ensure_dataset(data_dir: Path, zip_path: Path | None = None) -> Path:
    """确保本地存在 EuroSAT_RGB 数据目录。

    如果 ``data_dir`` 已存在就直接使用；否则在给定压缩包存在时自动解压。
    这样既支持用户手动解压后的目录，也支持本仓库当前只有 ``hw1.zip`` 的状态。
    """

    if data_dir.exists():
        return data_dir

    if zip_path is None or not zip_path.exists():
        raise FileNotFoundError(
            f"未找到数据目录 {data_dir}，也未找到可解压的数据压缩包 {zip_path}。"
        )

    data_dir.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(data_dir.parent)
    if not data_dir.exists():
        raise FileNotFoundError(f"压缩包已解压，但没有生成预期目录：{data_dir}")
    return data_dir


def discover_images(data_dir: Path, max_per_class: int | None = None) -> tuple[list[Path], np.ndarray, list[str]]:
    """扫描图片路径并生成类别编号。

    类别优先使用 EuroSAT 官方顺序；如果目录中存在额外类别，则按字母序追加。
    这种做法可以让报告里的类别顺序稳定，同时兼容轻微变化的数据目录。
    """

    existing_classes = sorted([p.name for p in data_dir.iterdir() if p.is_dir()])
    class_names = [name for name in EUROSAT_CLASSES if name in existing_classes]
    class_names.extend(name for name in existing_classes if name not in class_names)

    image_paths: list[Path] = []
    labels: list[int] = []
    for label, class_name in enumerate(class_names):
        class_dir = data_dir / class_name
        class_paths = sorted(class_dir.glob("*.jpg"))
        if max_per_class is not None:
            # 快速验证或调参时可以限制每类图片数量，避免首次开发调试读取完整数据集。
            class_paths = class_paths[:max_per_class]
        for path in class_paths:
            image_paths.append(path)
            labels.append(label)

    if not image_paths:
        raise ValueError(f"数据目录中没有找到 jpg 图片：{data_dir}")
    return image_paths, np.asarray(labels, dtype=np.int64), class_names


def split_indices(
    labels: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按类别分层划分训练集、验证集和测试集。

    分层划分可以避免某些类别在验证集或测试集中数量过少，评估结果比简单随机切分更稳定。
    """

    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []

    for label in np.unique(labels):
        class_indices = np.flatnonzero(labels == label)
        rng.shuffle(class_indices)
        n_total = len(class_indices)
        n_train = int(round(n_total * train_ratio))
        n_val = int(round(n_total * val_ratio))

        train_parts.append(class_indices[:n_train])
        val_parts.append(class_indices[n_train : n_train + n_val])
        test_parts.append(class_indices[n_train + n_val :])

    train_idx = np.concatenate(train_parts)
    val_idx = np.concatenate(val_parts)
    test_idx = np.concatenate(test_parts)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def load_images(paths: list[Path], image_size: int) -> tuple[np.ndarray, tuple[int, int, int]]:
    """读取图片并展平成 MLP 输入向量。

    EuroSAT_RGB 原图通常是 64x64。这里仍保留 ``image_size`` 参数，方便快速实验时
    降低分辨率，比如设为 32 来减少 MLP 参数量和训练时间。
    """

    arrays: list[np.ndarray] = []
    for path in paths:
        with Image.open(path) as image:
            image = image.convert("RGB").resize((image_size, image_size))
            arrays.append(np.asarray(image, dtype=np.float32) / 255.0)

    stacked = np.stack(arrays, axis=0)
    image_shape = (image_size, image_size, 3)
    return stacked.reshape(stacked.shape[0], -1), image_shape


def build_dataset(
    data_dir: Path,
    zip_path: Path | None,
    image_size: int = 64,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    cache_dir: Path | None = None,
    max_per_class: int | None = None,
) -> EuroSATData:
    """构建完整数据集，并可选缓存为 ``npz`` 文件。

    图片读取是训练前最慢的准备步骤之一。缓存只保存归一化后的数组和划分结果，
    不改变原始数据，后续重复训练可以明显减少等待时间。
    """

    data_dir = ensure_dataset(data_dir, zip_path)
    cache_path = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        sample_tag = "full" if max_per_class is None else f"max{max_per_class}"
        cache_path = cache_dir / f"eurosat_{image_size}_seed{seed}_{sample_tag}.npz"
        meta_path = cache_dir / f"eurosat_{image_size}_seed{seed}_{sample_tag}.json"
        if cache_path.exists() and meta_path.exists():
            cached = np.load(cache_path, allow_pickle=True)
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            return EuroSATData(
                train=DatasetSplit(cached["x_train"], cached["y_train"], cached["paths_train"].tolist()),
                val=DatasetSplit(cached["x_val"], cached["y_val"], cached["paths_val"].tolist()),
                test=DatasetSplit(cached["x_test"], cached["y_test"], cached["paths_test"].tolist()),
                class_names=meta["class_names"],
                image_shape=tuple(meta["image_shape"]),
            )

    image_paths, labels, class_names = discover_images(data_dir, max_per_class=max_per_class)
    train_idx, val_idx, test_idx = split_indices(labels, train_ratio, val_ratio, seed)

    x_all, image_shape = load_images(image_paths, image_size)
    path_strings = np.asarray([str(path) for path in image_paths], dtype=object)

    data = EuroSATData(
        train=DatasetSplit(x_all[train_idx], labels[train_idx], path_strings[train_idx].tolist()),
        val=DatasetSplit(x_all[val_idx], labels[val_idx], path_strings[val_idx].tolist()),
        test=DatasetSplit(x_all[test_idx], labels[test_idx], path_strings[test_idx].tolist()),
        class_names=class_names,
        image_shape=image_shape,
    )

    if cache_path is not None:
        np.savez_compressed(
            cache_path,
            x_train=data.train.x,
            y_train=data.train.y,
            paths_train=np.asarray(data.train.paths, dtype=object),
            x_val=data.val.x,
            y_val=data.val.y,
            paths_val=np.asarray(data.val.paths, dtype=object),
            x_test=data.test.x,
            y_test=data.test.y,
            paths_test=np.asarray(data.test.paths, dtype=object),
        )
        meta_path.write_text(
            json.dumps({"class_names": class_names, "image_shape": image_shape}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return data


def iterate_minibatches(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
    shuffle: bool = True,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """按小批量生成训练数据。"""

    indices = np.arange(len(x))
    if shuffle:
        rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        yield x[batch_idx], y[batch_idx]
