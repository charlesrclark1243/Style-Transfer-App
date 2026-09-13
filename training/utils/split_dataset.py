from collections import defaultdict
from pathlib import Path
import argparse
import random
import csv
import sys
import os

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png'}

def parse_args():
    parser = argparse.ArgumentParser(description="Split a dataset into training and validation sets.")

    parser.add_argument('--dataset-rootdir', type=str, required=True, help='Path to the dataset directory. Images anywhere below it are moved into train/ and val/ subdirectories.')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='Ratio of images to use for training (between 0 and 1, exclusive).')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')

    args = parser.parse_args()
    if not 0 < args.train_ratio < 1:
        parser.error(f"--train-ratio must be between 0 and 1, exclusive, got {args.train_ratio}")

    return args

def get_all_nested_image_paths(root_dir: Path):
    # Sorted so the seeded shuffle gives the same split regardless of filesystem order
    return sorted(
        path for path in root_dir.rglob('*')
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

def group_by_filename(paths: list[Path]):
    # The same painting can appear in several subdirectories; grouping keeps every copy in the same split
    groups = defaultdict(list)
    for path in paths:
        groups[path.name].append(path)

    return groups

def split_dataset(filenames: list[str], train_ratio: float, seed: int):
    filenames = sorted(filenames)
    random.Random(seed).shuffle(filenames)

    train_size = int(len(filenames) * train_ratio)
    train_filenames = filenames[:train_size]
    val_filenames = filenames[train_size:]

    if not train_filenames or not val_filenames:
        raise ValueError(f"Splitting {len(filenames)} images with train ratio {train_ratio} leaves train or val empty")

    return train_filenames, val_filenames

def plan_moves(root_dir: Path, groups: dict[str, list[Path]], train_filenames: list[str], val_filenames: list[str]):
    moves = []
    for split_dir, filenames in ((root_dir / "train", train_filenames), (root_dir / "val", val_filenames)):
        for filename in filenames:
            copies = groups[filename]
            for path in copies:
                if len(copies) == 1:
                    new_name = filename
                else:
                    # Copies sharing a filename can differ slightly, so all are kept, prefixed by their directories
                    new_name = "__".join(path.relative_to(root_dir).parts)

                moves.append((path, split_dir / new_name))

    destinations = [new_path for _, new_path in moves]
    if len(set(destinations)) != len(destinations):
        raise ValueError("Two images would be moved to the same destination; nothing has been moved")

    return moves

def move_images(root_dir: Path, moves: list[tuple[Path, Path]], output_file: Path):
    (root_dir / "train").mkdir()
    (root_dir / "val").mkdir()

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["original", "new"])

        for original_path, new_path in moves:
            os.rename(original_path, new_path)
            writer.writerow([original_path.relative_to(root_dir), new_path.relative_to(root_dir)])

def main():
    args = parse_args()

    dataset_rootdir = Path(args.dataset_rootdir)
    if not dataset_rootdir.is_dir():
        sys.exit(f"{dataset_rootdir} is not a directory")

    # A second run would pick up the already split images and reshuffle them
    existing_splits = [name for name in ("train", "val") if (dataset_rootdir / name).exists()]
    if existing_splits:
        sys.exit(f"{dataset_rootdir} already contains {' and '.join(existing_splits)}/; refusing to split again")

    all_image_paths = get_all_nested_image_paths(dataset_rootdir)
    groups = group_by_filename(all_image_paths)
    train_filenames, val_filenames = split_dataset(list(groups), args.train_ratio, args.seed)

    # Plan every move up front so a destination clash aborts before any file is touched
    moves = plan_moves(dataset_rootdir, groups, train_filenames, val_filenames)

    output_file = dataset_rootdir / "dataset_split.csv"
    move_images(dataset_rootdir, moves, output_file)

    num_train = sum(len(groups[filename]) for filename in train_filenames)
    print(f"Moved {num_train} images to train/ and {len(moves) - num_train} to val/. Record written to {output_file}")

if __name__ == "__main__":
    main()
