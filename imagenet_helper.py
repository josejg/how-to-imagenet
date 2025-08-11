#!/usr/bin/env python3

import io
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob
from pathlib import Path
from typing import Annotated

import numpy as np
import questionary
import typer
import xxhash
from PIL import Image
from tqdm import tqdm

app = typer.Typer(
    help="ImageNet ILSVRC2012 dataset download, extraction, and MDS conversion tool"
)

# Default paths
DEFAULT_TORRENT_DIR = "./bt"
DEFAULT_TAR_DIR = "./data/tar"
DEFAULT_UNTAR_DIR = "./data/untar"
DEFAULT_MDS_DIR = "./data/mds"
DEFAULT_AUX_DIR = "./aux"

# File hashes for verification
EXPECTED_HASHES = {
    "ILSVRC2012_img_val.tar": "20199bbecc0a0340",
    "ILSVRC2012_img_train.tar": "c5b744425289080b",
}

# Disk space requirements in GB
DISK_SPACE_REQUIREMENTS = {"train": 150, "val": 6}


def check_disk_space(path: str, required_gb: float) -> bool:
    """Check if there's enough free disk space, creating directories if needed."""
    try:
        # Create directory if it doesn't exist
        Path(path).mkdir(parents=True, exist_ok=True)

        stat = shutil.disk_usage(path)
        free_gb = stat.free / (1024**3)
        if free_gb < required_gb:
            typer.echo(
                f"❌ Insufficient disk space. Required: {required_gb}GB, Available: {free_gb:.1f}GB",
                err=True,
            )
            return False
        typer.echo(
            f"✅ Disk space check passed. Required: {required_gb}GB, Available: {free_gb:.1f}GB"
        )
        return True
    except Exception as e:
        typer.echo(f"❌ Error checking disk space: {e}", err=True)
        return False


def verify_file_hash(file_path: Path, expected_hash: str) -> bool:
    """Verify file hash using xxhash."""
    if not file_path.exists():
        return False

    typer.echo(f"Verifying hash for {file_path.name}...")
    hasher = xxhash.xxh64()

    with open(file_path, "rb") as f:
        with tqdm(desc="Hashing", unit="B", unit_scale=True) as pbar:
            while chunk := f.read(8192):
                hasher.update(chunk)
                pbar.update(len(chunk))

    actual_hash = hasher.hexdigest()
    if actual_hash == expected_hash:
        typer.echo(f"✅ Hash verification passed: {actual_hash}")
        return True
    else:
        typer.echo("❌ Hash verification failed!")
        typer.echo(f"  Expected: {expected_hash}")
        typer.echo(f"  Actual:   {actual_hash}")
        typer.echo(f"  Suggestion: Re-download {file_path.name}")
        return False


def download_with_aria2c(
    torrent_path: Path, output_dir: Path, cmd: list[str] | None = None
) -> bool:
    """Download file using aria2c subprocess."""
    try:
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Use provided command or build default
        if cmd is None:
            cmd = [
                "aria2c",
                "-c",  # Continue partial downloads
                "--seed-time=0",  # Don't seed after download
                "--file-allocation=falloc",  # Pre-allocate disk space
                "--bt-max-peers=256",  # Max peers for torrent
                str(torrent_path),
                "-d",
                str(output_dir),
            ]

        typer.echo(f"Starting download: {torrent_path.name}")

        # Run aria2c with direct stdout/stderr for live progress
        result = subprocess.run(cmd, check=False)

        if result.returncode == 0:
            typer.echo(f"✅ Download completed: {torrent_path.name}")
            return True
        else:
            typer.echo(f"❌ Download failed with exit code: {result.returncode}")
            return False

    except FileNotFoundError:
        typer.echo("❌ aria2c not found. Please install aria2:")
        typer.echo("   https://command-not-found.com/aria2c")
        return False
    except Exception as e:
        typer.echo(f"❌ Download error: {e}", err=True)
        return False


def get_tar_file_count(tar_path: Path) -> int:
    """Get the number of files in a tar archive."""
    try:
        result = subprocess.run(
            ["tar", "-tf", str(tar_path)], capture_output=True, text=True, check=True
        )
        return len(result.stdout.strip().split("\n")) if result.stdout.strip() else 0
    except subprocess.CalledProcessError:
        return 0


def extract_tar_with_progress(
    tar_path: Path, output_dir: Path, desc: str = "Extracting"
) -> bool:
    """Extract tar file with progress tracking using subprocess."""
    try:
        # Get total file count for progress bar
        total_files = get_tar_file_count(tar_path)
        if total_files == 0:
            typer.echo(f"❌ Could not determine file count in {tar_path}")
            return False

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Extract with verbose output to track progress
        process = subprocess.Popen(
            ["tar", "-xf", str(tar_path), "-C", str(output_dir), "-v"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        with tqdm(total=total_files, desc=desc, unit="files") as pbar:
            for line in iter(process.stdout.readline, ""):
                if line.strip():  # Non-empty line means a file was extracted
                    pbar.update(1)

            process.wait()

        if process.returncode != 0:
            stderr_output = process.stderr.read()
            typer.echo(f"❌ Extraction failed: {stderr_output}")
            return False

        return True

    except Exception as e:
        typer.echo(f"❌ Error during extraction: {e}")
        return False


def extract_nested_tar_with_progress(tar_path: Path, output_dir: Path) -> bool:
    """Extract nested tar file and clean up."""
    try:
        class_name = tar_path.stem
        class_dir = output_dir / class_name

        # Extract the nested tar
        if not extract_tar_with_progress(
            tar_path, class_dir, f"Extracting {class_name}"
        ):
            return False

        # Remove the nested tar file
        tar_path.unlink()
        return True

    except Exception as e:
        typer.echo(f"❌ Error extracting nested tar {tar_path}: {e}")
        return False


def check_dependencies(datasets: list[str], input_dir: str, operation: str) -> bool:
    """Check if required files exist for the operation."""
    if operation == "untar":
        # Check if tar files exist
        for dataset in datasets:
            tar_file = Path(input_dir) / f"ILSVRC2012_img_{dataset}.tar"
            if not tar_file.exists():
                typer.echo(f"❌ Required file not found: {tar_file}")
                typer.echo(f"Run 'download --only-{dataset}' first")
                return False
    elif operation == "mds":
        # Check if extracted directories exist
        for dataset in datasets:
            extract_dir = Path(input_dir) / f"ILSVRC2012_img_{dataset}"
            if not extract_dir.exists():
                typer.echo(f"❌ Required directory not found: {extract_dir}")
                typer.echo(f"Run 'untar --only-{dataset}' first")
                return False
    return True


@app.command()
def download(
    only_train: Annotated[
        bool, typer.Option("--only-train", help="Download only training set")
    ] = False,
    only_val: Annotated[
        bool, typer.Option("--only-val", help="Download only validation set")
    ] = False,
    torrent_dir: Annotated[
        str, typer.Option(help="Directory containing torrent files")
    ] = DEFAULT_TORRENT_DIR,
    output_dir: Annotated[
        str, typer.Option(help="Output directory for downloaded files")
    ] = DEFAULT_TAR_DIR,
    skip_confirmation: Annotated[
        bool, typer.Option("-y", help="Skip confirmation prompts")
    ] = False,
):
    """Download ImageNet dataset using torrents."""
    if only_train and only_val:
        typer.echo("❌ --only-train and --only-val are mutually exclusive", err=True)
        raise typer.Exit(1)

    # Determine which datasets to download
    datasets = []
    if only_train:
        datasets = ["train"]
    elif only_val:
        datasets = ["val"]
    else:
        datasets = ["train", "val"]

    # Check disk space requirements
    total_space_needed = sum(DISK_SPACE_REQUIREMENTS[ds] for ds in datasets)
    if not check_disk_space(output_dir, total_space_needed):
        raise typer.Exit(1)

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Process each dataset
    for dataset in datasets:
        typer.echo(f"\nProcessing {dataset} dataset...")

        # Check if torrent file exists
        torrent_file = Path(torrent_dir) / f"ILSVRC2012_img_{dataset}.tar.torrent"
        if not torrent_file.exists():
            typer.echo(f"❌ Torrent file not found: {torrent_file}")
            continue

        # Check if file already exists and verify hash
        output_file = Path(output_dir) / f"ILSVRC2012_img_{dataset}.tar"
        expected_hash = EXPECTED_HASHES[f"ILSVRC2012_img_{dataset}.tar"]

        if output_file.exists():
            if verify_file_hash(output_file, expected_hash):
                typer.echo(
                    f"✅ {output_file.name} already exists and verified, skipping download"
                )
                continue
            else:
                if not skip_confirmation:
                    if not questionary.confirm(
                        f"Hash verification failed for {output_file.name}. Re-download?"
                    ).ask():
                        continue

        # Show the exact aria2c command that will be run
        cmd = [
            "aria2c",
            "-c",  # Continue partial downloads
            "--seed-time=0",  # Don't seed after download
            "--file-allocation=falloc",  # Pre-allocate disk space
            "--bt-max-peers=256",  # Max peers for torrent
            str(torrent_file),
            "-d",
            str(Path(output_dir)),
        ]

        if not skip_confirmation:
            typer.echo(f"\nCommand to run: {' '.join(cmd)}")

        # Confirm download if not skipping
        if not skip_confirmation:
            file_size = "~150GB" if dataset == "train" else "~6GB"
            if not questionary.confirm(
                f"Download {dataset} dataset ({file_size})?"
            ).ask():
                continue

        # Download the file
        typer.echo(f"Downloading {dataset} dataset using torrent...")
        if download_with_aria2c(torrent_file, Path(output_dir), cmd):
            # Verify downloaded file
            if verify_file_hash(output_file, expected_hash):
                typer.echo(f"✅ {dataset} dataset downloaded and verified successfully")
            else:
                typer.echo(
                    f"❌ {dataset} dataset download completed but hash verification failed"
                )
                raise typer.Exit(1)
        else:
            typer.echo(f"❌ Failed to download {dataset} dataset")
            raise typer.Exit(1)


def extract_train_dataset(tar_file: Path, output_dir: Path) -> bool:
    """Extract training dataset with nested tar handling using subprocess."""
    try:
        train_dir = output_dir / "ILSVRC2012_img_train"
        train_dir.mkdir(parents=True, exist_ok=True)

        typer.echo(f"Extracting main training archive {tar_file.name}...")

        # Extract main tar file using subprocess
        if not extract_tar_with_progress(
            tar_file, train_dir, "Extracting main archive"
        ):
            return False

        # Find and extract nested tar files
        nested_tars = list(train_dir.glob("*.tar"))
        if not nested_tars:
            typer.echo("❌ No nested tar files found in training dataset")
            return False

        typer.echo(f"Found {len(nested_tars)} class tar files to extract...")

        # Extract nested tars with parallel processing for better performance
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for nested_tar in nested_tars:
                future = executor.submit(
                    extract_nested_tar_with_progress, nested_tar, train_dir
                )
                futures.append((future, nested_tar.stem))

            # Track progress
            with tqdm(
                total=len(futures), desc="Extracting class archives", unit="classes"
            ) as pbar:
                for future, class_name in futures:
                    if not future.result():
                        typer.echo(f"❌ Failed to extract {class_name}")
                        return False
                    pbar.update(1)

        typer.echo(f"✅ Training dataset extracted to {train_dir}")
        return True

    except Exception as e:
        typer.echo(f"❌ Error extracting training dataset: {e}")
        return False


def extract_val_dataset(tar_file: Path, output_dir: Path, aux_dir: Path) -> bool:
    """Extract validation dataset and sort by ground truth using subprocess."""
    try:
        val_dir = output_dir / "ILSVRC2012_img_val"
        val_raw_dir = output_dir / "ILSVRC2012_img_val_raw"

        # Create directories
        val_raw_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)

        typer.echo(f"Extracting {tar_file.name} to temporary directory...")

        # Extract to raw directory using subprocess
        if not extract_tar_with_progress(
            tar_file, val_raw_dir, "Extracting validation archive"
        ):
            return False

        # Load auxiliary files
        sorted_wnids_file = Path(aux_dir) / "sorted_wnids.txt"
        mapping_file = Path(aux_dir) / "ILSVRC2012_mapping.txt"
        ground_truth_file = Path(aux_dir) / "ILSVRC2012_validation_ground_truth.txt"

        if not all(
            f.exists() for f in [sorted_wnids_file, mapping_file, ground_truth_file]
        ):
            typer.echo("❌ Required auxiliary files not found:")
            typer.echo(f"  - {sorted_wnids_file}")
            typer.echo(f"  - {mapping_file}")
            typer.echo(f"  - {ground_truth_file}")
            return False

        # Load sorted wordnet IDs
        with open(sorted_wnids_file, "r") as f:
            sorted_wnids = [line.strip() for line in f]

        # Load mapping from integer to wordnet ID
        int_to_wnid = {}
        with open(mapping_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    int_id, wnid = parts
                    int_to_wnid[int(int_id)] = wnid

        # Load ground truth labels
        with open(ground_truth_file, "r") as f:
            ground_truth = [int(line.strip()) for line in f]

        # Create class directories
        typer.echo(f"Creating {len(sorted_wnids)} class directories...")
        for wnid in tqdm(sorted_wnids, desc="Creating directories"):
            class_dir = val_dir / wnid
            class_dir.mkdir(exist_ok=True)

        # Sort validation images
        val_images = list(val_raw_dir.glob("*.JPEG"))
        val_images.sort()  # Ensure consistent order

        if len(val_images) != len(ground_truth):
            typer.echo(
                f"❌ Mismatch: {len(val_images)} images vs {len(ground_truth)} ground truth labels"
            )
            return False

        typer.echo(
            f"📂 Sorting {len(val_images)} validation images into class folders..."
        )

        # Sort images with better progress tracking
        for i, img_file in enumerate(tqdm(val_images, desc="Sorting images")):
            # Get ground truth class ID and map to wordnet ID
            class_int_id = ground_truth[i]
            if class_int_id not in int_to_wnid:
                typer.echo(f"❌ Unknown class ID: {class_int_id}")
                return False

            wnid = int_to_wnid[class_int_id]
            target_dir = val_dir / wnid
            target_file = target_dir / img_file.name

            # Move image to class directory
            img_file.rename(target_file)

        # Clean up raw directory
        if val_raw_dir.exists() and not list(val_raw_dir.iterdir()):
            val_raw_dir.rmdir()
            typer.echo("Cleaned up temporary raw directory")

        typer.echo(f"✅ Validation dataset extracted and sorted to {val_dir}")
        return True

    except Exception as e:
        typer.echo(f"❌ Error extracting validation dataset: {e}")
        return False


@app.command()
def untar(
    only_train: Annotated[
        bool, typer.Option("--only-train", help="Extract only training set")
    ] = False,
    only_val: Annotated[
        bool, typer.Option("--only-val", help="Extract only validation set")
    ] = False,
    input_dir: Annotated[
        str, typer.Option(help="Directory containing tar files")
    ] = DEFAULT_TAR_DIR,
    output_dir: Annotated[
        str, typer.Option(help="Output directory for extracted files")
    ] = DEFAULT_UNTAR_DIR,
    aux_dir: Annotated[
        str, typer.Option(help="Directory containing auxiliary files")
    ] = DEFAULT_AUX_DIR,
    skip_confirmation: Annotated[
        bool, typer.Option("-y", help="Skip confirmation prompts")
    ] = False,
):
    """Extract ImageNet tar files."""
    if only_train and only_val:
        typer.echo("❌ --only-train and --only-val are mutually exclusive", err=True)
        raise typer.Exit(1)

    # Determine which datasets to extract
    datasets = []
    if only_train:
        datasets = ["train"]
    elif only_val:
        datasets = ["val"]
    else:
        datasets = ["train", "val"]

    # Check dependencies
    if not check_dependencies(datasets, input_dir, "untar"):
        raise typer.Exit(1)

    # Check disk space requirements
    total_space_needed = sum(DISK_SPACE_REQUIREMENTS[ds] for ds in datasets)
    if not check_disk_space(output_dir, total_space_needed):
        raise typer.Exit(1)

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Process each dataset
    for dataset in datasets:
        typer.echo(f"\nProcessing {dataset} dataset...")

        # Check if already extracted
        extracted_dir = Path(output_dir) / f"ILSVRC2012_img_{dataset}"
        if extracted_dir.exists() and list(extracted_dir.iterdir()):
            if not skip_confirmation:
                if not questionary.confirm(
                    f"{extracted_dir} already exists. Re-extract?"
                ).ask():
                    continue
            # Remove existing directory
            shutil.rmtree(extracted_dir)

        # Confirm extraction if not skipping
        if not skip_confirmation:
            if not questionary.confirm(f"Extract {dataset} dataset?").ask():
                continue

        # Extract the dataset
        tar_file = Path(input_dir) / f"ILSVRC2012_img_{dataset}.tar"

        if dataset == "train":
            success = extract_train_dataset(tar_file, Path(output_dir))
        else:  # val
            success = extract_val_dataset(tar_file, Path(output_dir), Path(aux_dir))

        if not success:
            typer.echo(f"❌ Failed to extract {dataset} dataset")
            raise typer.Exit(1)


def load_image(file_path: str, validate: bool = False) -> bytes:
    """Load image bytes from file."""
    with open(file_path, "rb") as f:
        image_bytes = f.read()

    if validate:
        try:
            Image.open(io.BytesIO(image_bytes)).verify()
        except Exception as e:
            raise ValueError(f"Invalid image {file_path}: {e}")

    return image_bytes


def convert_to_mds(
    input_path: Path,
    output_path: Path,
    split: str,
    compression: str | None = None,
    hashes_list: list[str] | None = None,
    size_limit: int = 1 << 26,
    seed: int = 42,
    shuffle: bool = True,
    validate: bool = False,
    extensions: set[str] | None = None,
    max_workers: int = 8,
    progress_bar: bool = True,
) -> bool:
    """Convert ImageNet directories to MDS format."""

    # This import is here because they take a bit to evaluate
    from streaming.base import MDSWriter

    try:
        if extensions is None:
            extensions = {"jpeg", "jpg", "png"}

        # Step 1: Discover all class directories and create class mapping
        typer.echo("Discovering class directories and creating class mapping...")
        class_dirs = []
        for item in os.listdir(input_path):
            item_path = os.path.join(input_path, item)
            if os.path.isdir(item_path):
                class_dirs.append(item)

        # Sort to create deterministic class mapping
        class_dirs.sort()
        class_mapping = {class_dir: idx for idx, class_dir in enumerate(class_dirs)}

        typer.echo(f"Found {len(class_dirs)} classes")

        # Step 2: Build complete file inventory
        typer.echo("Building complete file inventory...")
        all_files = []

        for class_dir in tqdm(
            class_dirs, desc="Scanning class directories", disable=not progress_bar
        ):
            class_path = os.path.join(input_path, class_dir)
            class_id = class_mapping[class_dir]
            wnid = class_dir  # The directory name is the wnid (e.g., "n01440764")

            # Get all files in the directory
            pattern = os.path.join(class_path, "*")
            image_files = glob(pattern)

            # Filter and sort files
            valid_files = []
            for file_path in image_files:
                if not os.path.isfile(file_path):
                    continue

                # Check extension
                if extensions:
                    ext = file_path.split(".")[-1].lower()
                    if ext not in extensions:
                        continue

                valid_files.append(file_path)

            # Sort files for deterministic order
            valid_files.sort()

            for file_path in valid_files:
                all_files.append((file_path, class_id, wnid))

        typer.echo(f"Found {len(all_files)} total images")

        # Step 3: Apply global permutation if shuffling
        if shuffle:
            typer.echo(f"Applying global permutation with seed {seed}...")
            rng = np.random.default_rng(seed)
            permuted_indices = rng.permutation(len(all_files))
            permuted_files = [all_files[i] for i in permuted_indices]
        else:
            typer.echo("Preserving original order (no shuffling)")
            permuted_files = all_files

        # Step 4: Process with threading
        typer.echo("Converting to MDS format...")

        # Prepare output directory
        output_path.mkdir(parents=True, exist_ok=True)

        # Create MDS writer
        columns = {
            "i": "int",  # image index
            "x": "jpeg",  # image data (auto-decoded to PIL)
            "y": "int",  # class label
            "wnid": "str",  # wordnet id (e.g., "n01440764")
        }

        with MDSWriter(
            out=str(output_path),
            columns=columns,
            compression=compression,
            hashes=hashes_list,
            size_limit=size_limit,
        ) as out:

            def process_image(args):
                idx, (file_path, class_id, wnid) = args
                try:
                    image_bytes = load_image(file_path, validate)
                    return idx, {
                        "i": idx,
                        "x": image_bytes,
                        "y": class_id,
                        "wnid": wnid,
                    }
                except Exception as e:
                    raise RuntimeError(f"Failed to process {file_path}: {e}")

            # Process images in batches to avoid OOM
            batch_size = 2000  # Process 2000 images at a time
            total_batches = (len(permuted_files) + batch_size - 1) // batch_size

            # Single progress bar for all processing
            with tqdm(
                total=len(permuted_files),
                desc="Processing images",
                disable=not progress_bar,
            ) as pbar:
                for batch_num in range(total_batches):
                    start_idx = batch_num * batch_size
                    end_idx = min(start_idx + batch_size, len(permuted_files))
                    batch_files = permuted_files[start_idx:end_idx]

                    # Process this batch with threading
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        future_to_idx = {
                            executor.submit(
                                process_image, (start_idx + i, file_info)
                            ): i
                            for i, file_info in enumerate(batch_files)
                        }

                        # Collect batch results in order
                        batch_results = [None] * len(batch_files)

                        for future in as_completed(future_to_idx):
                            try:
                                idx, sample = future.result()
                                relative_idx = future_to_idx[future]
                                batch_results[relative_idx] = sample
                                pbar.update(
                                    1
                                )  # Update progress bar for each completed image
                            except Exception as e:
                                # Cancel all remaining futures in this batch
                                for f in future_to_idx:
                                    f.cancel()
                                raise e

                    # Write batch results in order
                    for sample in batch_results:
                        if sample is not None:
                            out.write(sample)

        typer.echo(f"✅ Conversion complete! Output saved to {output_path}")
        return True

    except Exception as e:
        typer.echo(f"❌ Error during MDS conversion: {e}")
        return False


@app.command()
def mds(
    only_train: Annotated[
        bool, typer.Option("--only-train", help="Convert only training set")
    ] = False,
    only_val: Annotated[
        bool, typer.Option("--only-val", help="Convert only validation set")
    ] = False,
    input_dir: Annotated[
        str, typer.Option(help="Directory containing extracted images")
    ] = DEFAULT_UNTAR_DIR,
    output_dir: Annotated[
        str, typer.Option(help="Output directory for MDS files")
    ] = DEFAULT_MDS_DIR,
    compression: Annotated[str, typer.Option(help="Compression algorithm to use")] = "",
    hashes: Annotated[str, typer.Option(help="Hashing algorithms")] = "sha1,xxh64",
    size_limit: Annotated[int, typer.Option(help="Shard size limit in bytes")] = 1
    << 26,
    seed: Annotated[
        int, typer.Option(help="Random seed for shuffling (train only)")
    ] = 42,
    validate: Annotated[
        bool, typer.Option(help="Validate that files are valid images")
    ] = False,
    extensions: Annotated[
        str, typer.Option(help="Valid filename extensions")
    ] = "jpeg,jpg",
    max_workers: Annotated[int, typer.Option(help="Number of worker threads")] = 8,
    skip_confirmation: Annotated[
        bool, typer.Option("-y", help="Skip confirmation prompts")
    ] = False,
):
    """Convert ImageNet images to MDS format."""

    from streaming.base.util import get_list_arg

    if only_train and only_val:
        typer.echo("❌ --only-train and --only-val are mutually exclusive", err=True)
        raise typer.Exit(1)

    # Determine which datasets to convert
    datasets = []
    if only_train:
        datasets = ["train"]
    elif only_val:
        datasets = ["val"]
    else:
        datasets = ["train", "val"]

    # Check dependencies
    if not check_dependencies(datasets, input_dir, "mds"):
        raise typer.Exit(1)

    # Check disk space requirements (same as input for MDS conversion)
    total_space_needed = sum(DISK_SPACE_REQUIREMENTS[ds] for ds in datasets)
    if not check_disk_space(output_dir, total_space_needed):
        raise typer.Exit(1)

    # Parse extensions and hashes
    extensions_set = (
        set(ext.lower() for ext in extensions.split(","))
        if extensions
        else {"jpeg", "jpg"}
    )
    hashes_list = get_list_arg(hashes) if hashes else None
    compression_arg = compression if compression else None

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Process each dataset
    for dataset in datasets:
        typer.echo(f"\nProcessing {dataset} dataset...")

        # Check if already converted
        output_path = Path(output_dir) / f"ILSVRC2012_img_{dataset}"
        if output_path.exists() and list(output_path.iterdir()):
            if not skip_confirmation:
                if not questionary.confirm(
                    f"{output_path} already exists. Re-convert?"
                ).ask():
                    continue
            # Remove existing directory
            shutil.rmtree(output_path)

        # Confirm conversion if not skipping
        if not skip_confirmation:
            shuffle_msg = " (shuffled)" if dataset == "train" else " (ordered)"
            if not questionary.confirm(
                f"Convert {dataset} dataset to MDS{shuffle_msg}?"
            ).ask():
                continue

        # Convert the dataset
        input_path = Path(input_dir) / f"ILSVRC2012_img_{dataset}"
        shuffle = dataset == "train"  # Shuffle training data, preserve validation order

        success = convert_to_mds(
            input_path=input_path,
            output_path=output_path,
            split=dataset,
            compression=compression_arg,
            hashes_list=hashes_list,
            size_limit=size_limit,
            seed=seed,
            shuffle=shuffle,
            validate=validate,
            extensions=extensions_set,
            max_workers=max_workers,
            progress_bar=True,
        )

        if not success:
            typer.echo(f"❌ Failed to convert {dataset} dataset to MDS")
            raise typer.Exit(1)


@app.command()
def status(
    torrent_dir: Annotated[
        str, typer.Option(help="Directory containing torrent files")
    ] = DEFAULT_TORRENT_DIR,
    tar_dir: Annotated[
        str, typer.Option(help="Directory containing tar files")
    ] = DEFAULT_TAR_DIR,
    untar_dir: Annotated[
        str, typer.Option(help="Directory containing extracted files")
    ] = DEFAULT_UNTAR_DIR,
    mds_dir: Annotated[
        str, typer.Option(help="Directory containing MDS files")
    ] = DEFAULT_MDS_DIR,
):
    """Show status of ImageNet dataset processing."""
    typer.echo("ImageNet Dataset Status")
    typer.echo("=" * 50)

    # Check torrent files
    typer.echo("\nTorrent Files:")
    torrent_path = Path(torrent_dir)
    if torrent_path.exists():
        train_torrent = torrent_path / "ILSVRC2012_img_train.tar.torrent"
        val_torrent = torrent_path / "ILSVRC2012_img_val.tar.torrent"
        typer.echo(
            f"  Train: {'✅' if train_torrent.exists() else '❌'} {train_torrent}"
        )
        typer.echo(f"  Val:   {'✅' if val_torrent.exists() else '❌'} {val_torrent}")
    else:
        typer.echo(f"  ❌ Directory not found: {torrent_dir}")

    # Check downloaded tar files
    typer.echo("\nDownloaded Files:")
    tar_path = Path(tar_dir)
    if tar_path.exists():
        train_tar = tar_path / "ILSVRC2012_img_train.tar"
        val_tar = tar_path / "ILSVRC2012_img_val.tar"

        for tar_file in [train_tar, val_tar]:
            if tar_file.exists():
                size_gb = tar_file.stat().st_size / (1024**3)
                typer.echo(f"  ✅ {tar_file.name} ({size_gb:.1f}GB)")
            else:
                typer.echo(f"  ❌ {tar_file.name}")
    else:
        typer.echo(f"  ❌ Directory not found: {tar_dir}")

    # Check extracted files
    typer.echo("\nExtracted Files:")
    untar_path = Path(untar_dir)
    if untar_path.exists():
        train_untar = untar_path / "ILSVRC2012_img_train"
        val_untar = untar_path / "ILSVRC2012_img_val"

        for extract_dir in [train_untar, val_untar]:
            if extract_dir.exists() and extract_dir.is_dir():
                try:
                    file_count = len(list(extract_dir.rglob("*")))
                    typer.echo(f"  ✅ {extract_dir.name} ({file_count} files)")
                except Exception:
                    typer.echo(f"  ✅ {extract_dir.name} (exists)")
            else:
                typer.echo(f"  ❌ {extract_dir.name}")
    else:
        typer.echo(f"  ❌ Directory not found: {untar_dir}")

    # Check MDS files
    typer.echo("\nMDS Files:")
    mds_path = Path(mds_dir)
    if mds_path.exists():
        train_mds = mds_path / "ILSVRC2012_img_train"
        val_mds = mds_path / "ILSVRC2012_img_val"

        for mds_subdir in [train_mds, val_mds]:
            if mds_subdir.exists() and mds_subdir.is_dir():
                try:
                    shard_count = len(list(mds_subdir.glob("*.mds")))
                    typer.echo(f"  ✅ {mds_subdir.name} ({shard_count} shards)")
                except Exception:
                    typer.echo(f"  ✅ {mds_subdir.name} (exists)")
            else:
                typer.echo(f"  ❌ {mds_subdir.name}")
    else:
        typer.echo(f"  ❌ Directory not found: {mds_dir}")


if __name__ == "__main__":
    app()
