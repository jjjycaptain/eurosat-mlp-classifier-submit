"""训练 EuroSAT MLP 分类器的命令行入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.data import build_dataset, iterate_minibatches
from src.metrics import accuracy
from src.model import MLPClassifier, cross_entropy


def parse_args() -> argparse.Namespace:
    """解析训练参数。"""

    parser = argparse.ArgumentParser(description="使用纯 NumPy 训练 EuroSAT 三层 MLP 分类器")
    parser.add_argument("--data-dir", type=Path, default=Path("EuroSAT_RGB"))
    parser.add_argument("--zip-path", type=Path, default=Path("hw1.zip"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--hidden-dim1", type=int, default=256)
    parser.add_argument("--hidden-dim2", type=int, default=128)
    parser.add_argument("--activation", choices=["relu", "sigmoid", "tanh"], default="relu")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--lr-decay", type=float, default=0.95)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache", action="store_true", help="缓存读取后的数据数组，加快重复实验")
    parser.add_argument("--max-per-class", type=int, default=None, help="每类最多读取多少张图片；默认读取完整数据集")
    return parser.parse_args()


def evaluate(model: MLPClassifier, x: np.ndarray, y: np.ndarray, batch_size: int) -> tuple[float, float]:
    """计算给定数据集上的损失和准确率。"""

    probs = model.predict_proba(x, batch_size=batch_size)
    preds = np.argmax(probs, axis=1)
    return cross_entropy(probs, y), accuracy(y, preds)


def save_checkpoint(path: Path, model: MLPClassifier, config: dict, class_names: list[str], image_shape: tuple[int, int, int]) -> None:
    """保存最佳模型权重和复现实验所需的元信息。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **{name: value for name, value in model.state_dict().items() if isinstance(value, np.ndarray)},
        activation=np.asarray(model.activation_name),
        config=json.dumps(config, ensure_ascii=False),
        class_names=np.asarray(class_names, dtype=object),
        image_shape=np.asarray(image_shape),
    )


def main() -> None:
    """执行完整训练流程。"""

    args = parse_args()
    rng = np.random.default_rng(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(
        data_dir=args.data_dir,
        zip_path=args.zip_path,
        image_size=args.image_size,
        seed=args.seed,
        cache_dir=args.output_dir / "cache" if args.cache else None,
        max_per_class=args.max_per_class,
    )

    config = vars(args).copy()
    config["data_dir"] = str(args.data_dir)
    config["zip_path"] = str(args.zip_path)
    config["output_dir"] = str(args.output_dir)

    model = MLPClassifier(
        input_dim=dataset.train.x.shape[1],
        hidden_dim1=args.hidden_dim1,
        hidden_dim2=args.hidden_dim2,
        num_classes=len(dataset.class_names),
        activation=args.activation,
        seed=args.seed,
    )

    history: list[dict[str, float]] = []
    best_val_acc = -1.0
    best_path = args.output_dir / "checkpoints" / "best_model.npz"

    for epoch in range(1, args.epochs + 1):
        lr = args.learning_rate * (args.lr_decay ** (epoch - 1))
        batch_losses: list[float] = []
        for x_batch, y_batch in iterate_minibatches(dataset.train.x, dataset.train.y, args.batch_size, rng):
            loss, grads = model.loss_and_grads(x_batch, y_batch, weight_decay=args.weight_decay)
            model.step(grads, learning_rate=lr)
            batch_losses.append(loss)

        train_loss, train_acc = evaluate(model, dataset.train.x, dataset.train.y, args.batch_size)
        val_loss, val_acc = evaluate(model, dataset.val.x, dataset.val.y, args.batch_size)
        row = {
            "epoch": float(epoch),
            "learning_rate": float(lr),
            "batch_loss": float(np.mean(batch_losses)),
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} lr={lr:.6f} train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(best_path, model, config, dataset.class_names, dataset.image_shape)

    history_path = args.output_dir / "logs" / "history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"最佳验证集准确率：{best_val_acc:.4f}")
    print(f"最佳模型已保存：{best_path}")


if __name__ == "__main__":
    main()
