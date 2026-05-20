from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()

    import torchvision

    for split in ("train", "val", "test"):
        torchvision.datasets.Flowers102(
            root=args.data_dir,
            split=split,
            download=True,
        )
        print(f"prepared split={split}")


if __name__ == "__main__":
    main()
