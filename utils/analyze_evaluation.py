import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def main(args):
    prediction = pd.read_csv(args.filepath)
    print(prediction.head())

    print(prediction.info())
    print(prediction.describe())
    save_root = Path(args.save_root_path)
    save_root.mkdir(parents=True, exist_ok=True)
    plot_histogram(prediction["acc"], save_root / "acc.png")
    plot_histogram(prediction["dice"], save_root / "dice.png")
    plot_histogram(prediction["iou"], save_root / "iou.png")


def plot_histogram(prediction: pd.DataFrame, save_path: Path):
    plt.figure()
    prediction.hist()
    plt.ylim(0, 40)
    plt.savefig(save_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str, required=True)
    parser.add_argument("--save_root_path", type=str, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)