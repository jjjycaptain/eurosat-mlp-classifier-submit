"""生成训练曲线、混淆矩阵、错误样本和第一层权重可视化。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.data import build_dataset
from src.evaluate import load_model
from src.metrics import confusion_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成实验报告所需图片")
    parser.add_argument("--data-dir", type=Path, default=Path("EuroSAT_RGB"))
    parser.add_argument("--zip-path", type=Path, default=Path("hw1.zip"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/checkpoints/best_model.npz"))
    parser.add_argument("--history", type=Path, default=Path("outputs/logs/history.json"))
    parser.add_argument("--metrics", type=Path, default=Path("outputs/test_metrics.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-class", type=int, default=None, help="每类最多读取多少张图片；默认继承 checkpoint 配置")
    return parser.parse_args()


def save_training_curves(history_path: Path, output_dir: Path) -> None:
    """绘制训练/验证 loss 和 accuracy 曲线。"""

    if not history_path.exists():
        return
    history = json.loads(history_path.read_text(encoding="utf-8"))
    epochs = [row["epoch"] for row in history]

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, [row["train_loss"] for row in history], label="train loss")
    plt.plot(epochs, [row["val_loss"] for row in history], label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, [row["train_accuracy"] for row in history], label="train accuracy")
    plt.plot(epochs, [row["val_accuracy"] for row in history], label="val accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_curve.png", dpi=160)
    plt.close()


def save_confusion_matrix(matrix: np.ndarray, class_names: list[str], output_dir: Path) -> None:
    """绘制混淆矩阵热力图。"""

    plt.figure(figsize=(9, 8))
    plt.imshow(matrix, cmap="Blues")
    plt.colorbar()
    plt.xticks(np.arange(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(np.arange(len(class_names)), class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=180)
    plt.close()


def normalize_image(array: np.ndarray) -> np.ndarray:
    """把任意权重数组缩放到 0-1，方便作为图片显示。"""

    low, high = np.percentile(array, [2, 98])
    clipped = np.clip(array, low, high)
    return (clipped - clipped.min()) / max(clipped.max() - clipped.min(), 1e-8)


def save_first_layer_weights(model, image_shape: tuple[int, int, int], output_dir: Path, count: int = 16) -> None:
    """把第一层部分神经元权重恢复成图像尺寸。"""

    weights = model.params["W1"]
    count = min(count, weights.shape[1])
    cols = 4
    rows = int(np.ceil(count / cols))
    plt.figure(figsize=(cols * 2, rows * 2))
    for i in range(count):
        weight_image = weights[:, i].reshape(image_shape)
        plt.subplot(rows, cols, i + 1)
        plt.imshow(normalize_image(weight_image))
        plt.axis("off")
        plt.title(f"unit {i}")
    plt.tight_layout()
    plt.savefig(output_dir / "first_layer_weights.png", dpi=180)
    plt.close()


def save_error_examples(model, dataset, class_names: list[str], output_dir: Path, max_items: int = 12) -> None:
    """保存测试集中若干分类错误样本。"""

    preds = model.predict(dataset.test.x)
    wrong_indices = np.flatnonzero(preds != dataset.test.y)[:max_items]
    if len(wrong_indices) == 0:
        return

    cols = 4
    rows = int(np.ceil(len(wrong_indices) / cols))
    plt.figure(figsize=(cols * 3, rows * 3))
    for plot_idx, data_idx in enumerate(wrong_indices, start=1):
        image = Image.open(dataset.test.paths[int(data_idx)]).convert("RGB")
        true_name = class_names[int(dataset.test.y[data_idx])]
        pred_name = class_names[int(preds[data_idx])]
        plt.subplot(rows, cols, plot_idx)
        plt.imshow(image)
        plt.axis("off")
        plt.title(f"T:{true_name}\nP:{pred_name}", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "error_examples.png", dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, config, class_names, image_shape = load_model(args.checkpoint)
    # 可视化错误样本时需要重新读取测试集；默认沿用训练时的样本上限，便于快速烟测。
    max_per_class = args.max_per_class
    if max_per_class is None:
        max_per_class = config.get("max_per_class")
    dataset = build_dataset(
        data_dir=args.data_dir,
        zip_path=args.zip_path,
        image_size=image_shape[0],
        seed=args.seed,
        cache_dir=Path("outputs/cache"),
        max_per_class=max_per_class,
    )

    save_training_curves(args.history, args.output_dir)
    preds = model.predict(dataset.test.x)
    matrix = confusion_matrix(dataset.test.y, preds, len(class_names))
    save_confusion_matrix(matrix, class_names, args.output_dir)
    save_first_layer_weights(model, image_shape, args.output_dir)
    save_error_examples(model, dataset, class_names, args.output_dir)
    print(f"可视化图片已保存到：{args.output_dir}")


if __name__ == "__main__":
    main()
