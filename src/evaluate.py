"""加载最佳模型，在测试集上评估并输出混淆矩阵。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.data import build_dataset
from src.metrics import accuracy, confusion_matrix
from src.model import MLPClassifier, cross_entropy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估已训练的 EuroSAT MLP 模型")
    parser.add_argument("--data-dir", type=Path, default=Path("EuroSAT_RGB"))
    parser.add_argument("--zip-path", type=Path, default=Path("hw1.zip"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/checkpoints/best_model.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-class", type=int, default=None, help="每类最多读取多少张图片；默认继承 checkpoint 配置")
    return parser.parse_args()


def load_model(checkpoint: Path) -> tuple[MLPClassifier, dict, list[str], tuple[int, int, int]]:
    """从 checkpoint 恢复模型结构和参数。"""

    state = np.load(checkpoint, allow_pickle=True)
    config = json.loads(str(state["config"]))
    class_names = state["class_names"].tolist()
    image_shape = tuple(int(v) for v in state["image_shape"].tolist())
    model = MLPClassifier(
        input_dim=int(np.prod(image_shape)),
        hidden_dim1=int(config["hidden_dim1"]),
        hidden_dim2=int(config["hidden_dim2"]),
        num_classes=len(class_names),
        activation=str(state["activation"]),
        seed=int(config["seed"]),
    )
    model.load_state_dict({name: state[name] for name in ("W1", "b1", "W2", "b2", "W3", "b3")} | {"activation": str(state["activation"])})
    return model, config, class_names, image_shape


def main() -> None:
    args = parse_args()
    model, config, class_names, image_shape = load_model(args.checkpoint)
    # 若命令行未指定样本上限，则继承训练 checkpoint 中的配置，保证烟测训练和评估使用同一数据规模。
    max_per_class = args.max_per_class
    if max_per_class is None:
        max_per_class = config.get("max_per_class")
    dataset = build_dataset(
        data_dir=args.data_dir,
        zip_path=args.zip_path,
        image_size=image_shape[0],
        seed=args.seed,
        cache_dir=args.output_dir / "cache",
        max_per_class=max_per_class,
    )

    probs = model.predict_proba(dataset.test.x, batch_size=args.batch_size)
    preds = np.argmax(probs, axis=1)
    test_loss = cross_entropy(probs, dataset.test.y)
    test_acc = accuracy(dataset.test.y, preds)
    matrix = confusion_matrix(dataset.test.y, preds, len(class_names))

    metrics = {
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "class_names": class_names,
        "confusion_matrix": matrix.tolist(),
        "config": config,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "test_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"测试集 Loss：{test_loss:.4f}")
    print(f"测试集 Accuracy：{test_acc:.4f}")
    print("混淆矩阵：")
    print(matrix)


if __name__ == "__main__":
    main()
