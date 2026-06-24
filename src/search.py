"""简单超参数搜索入口。"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from src.data import build_dataset, iterate_minibatches
from src.metrics import accuracy
from src.model import MLPClassifier, cross_entropy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对 EuroSAT MLP 进行网格超参数搜索")
    parser.add_argument("--data-dir", type=Path, default=Path("EuroSAT_RGB"))
    parser.add_argument("--zip-path", type=Path, default=Path("hw1.zip"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/search"))
    parser.add_argument("--image-size", type=int, default=32, help="搜索阶段默认降采样，加快比较不同参数")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dims", type=str, default="128-64,256-128")
    parser.add_argument("--learning-rates", type=str, default="0.03,0.01")
    parser.add_argument("--weight-decays", type=str, default="0,0.0001")
    parser.add_argument("--activations", type=str, default="relu,tanh")
    parser.add_argument("--lr-decay", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-class", type=int, default=500, help="搜索阶段每类最多读取多少张图片")
    return parser.parse_args()


def parse_hidden_dims(value: str) -> list[tuple[int, int]]:
    """解析 ``128-64,256-128`` 形式的隐藏层配置。"""

    dims: list[tuple[int, int]] = []
    for item in value.split(","):
        first, second = item.split("-")
        dims.append((int(first), int(second)))
    return dims


def parse_float_list(value: str) -> list[float]:
    """解析逗号分隔的浮点数列表。"""

    return [float(item) for item in value.split(",") if item]


def evaluate(model: MLPClassifier, x: np.ndarray, y: np.ndarray, batch_size: int) -> tuple[float, float]:
    probs = model.predict_proba(x, batch_size=batch_size)
    preds = np.argmax(probs, axis=1)
    return cross_entropy(probs, y), accuracy(y, preds)


def run_one_config(dataset, config: dict, seed: int) -> dict:
    """训练并评估单组超参数。"""

    rng = np.random.default_rng(seed)
    model = MLPClassifier(
        input_dim=dataset.train.x.shape[1],
        hidden_dim1=config["hidden_dim1"],
        hidden_dim2=config["hidden_dim2"],
        num_classes=len(dataset.class_names),
        activation=config["activation"],
        seed=seed,
    )

    best_val_acc = -1.0
    best_val_loss = float("inf")
    for epoch in range(1, config["epochs"] + 1):
        lr = config["learning_rate"] * (config["lr_decay"] ** (epoch - 1))
        for x_batch, y_batch in iterate_minibatches(dataset.train.x, dataset.train.y, config["batch_size"], rng):
            _, grads = model.loss_and_grads(x_batch, y_batch, weight_decay=config["weight_decay"])
            model.step(grads, learning_rate=lr)
        val_loss, val_acc = evaluate(model, dataset.val.x, dataset.val.y, config["batch_size"])
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss

    return {**config, "best_val_loss": best_val_loss, "best_val_accuracy": best_val_acc}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = build_dataset(
        data_dir=args.data_dir,
        zip_path=args.zip_path,
        image_size=args.image_size,
        seed=args.seed,
        cache_dir=args.output_dir / "cache",
        max_per_class=args.max_per_class,
    )

    grid = itertools.product(
        parse_hidden_dims(args.hidden_dims),
        parse_float_list(args.learning_rates),
        parse_float_list(args.weight_decays),
        [item for item in args.activations.split(",") if item],
    )

    results: list[dict] = []
    for index, (hidden_dims, learning_rate, weight_decay, activation) in enumerate(grid, start=1):
        config = {
            "hidden_dim1": hidden_dims[0],
            "hidden_dim2": hidden_dims[1],
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "activation": activation,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr_decay": args.lr_decay,
            "image_size": args.image_size,
        }
        result = run_one_config(dataset, config, seed=args.seed + index)
        results.append(result)
        print(
            f"[{index}] hidden={hidden_dims} lr={learning_rate} wd={weight_decay} "
            f"act={activation} val_acc={result['best_val_accuracy']:.4f}"
        )

    results.sort(key=lambda row: row["best_val_accuracy"], reverse=True)
    (args.output_dir / "search_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("最佳配置：")
    print(json.dumps(results[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
