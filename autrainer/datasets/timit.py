import os
from pathlib import Path
import random
import shutil
from typing import Dict, List, Optional, Union
from zipfile import ZipFile

from omegaconf import DictConfig
import pandas as pd
import requests
from tqdm import tqdm


try:
    import gdown

    GDOWN_AVAILABLE = True
except ImportError:
    GDOWN_AVAILABLE = False

from autrainer.transforms import SmartCompose

from .abstract_dataset import BaseClassificationDataset


# Google Drive file ID for TIMIT dataset
TIMIT_GDRIVE_ID = "1x3n4aDBPqH1ajBVQAxosFLhJu8HH7zdt"
TIMIT_GDRIVE_URL = f"https://drive.google.com/uc?id={TIMIT_GDRIVE_ID}"
TIMIT_ZIP_URL = f"https://drive.google.com/uc?export=download&id={TIMIT_GDRIVE_ID}"

# Dialect mapping from region codes to names
DIALECT_MAPPING = {
    "1": "New_England",
    "2": "Northern",
    "3": "North_Midland",
    "4": "South_Midland",
    "5": "Southern",
    "6": "New_York_City",
    "7": "Western",
    "8": "Army_Brat",
}


class TIMIT(BaseClassificationDataset):
    """TIMIT dataset for various classification tasks.

    The TIMIT corpus is a well-known speech database containing broadband
    recordings of 630 speakers of eight major dialects of American English,
    each reading ten phonetically rich sentences.

    This dataset class supports multiple classification tasks:
    - Dialect classification (8 dialects)
    - Gender classification (Male/Female)
    - Sentence type classification (SA/SI/SX)

    The dataset will be automatically downloaded and prepared when using
    the `autrainer fetch` command.

    For more information on the TIMIT corpus, see:
    https://catalog.ldc.upenn.edu/LDC93S1
    """

    def __init__(
        self,
        path: str,
        features_subdir: Optional[str],
        seed: int,
        metrics: List[Union[str, DictConfig, Dict]],
        tracking_metric: Union[str, DictConfig, Dict],
        index_column: str,
        target_column: str,
        file_type: str,
        file_handler: Union[str, DictConfig, Dict],
        features_path: Optional[str] = None,
        train_transform: Optional[SmartCompose] = None,
        dev_transform: Optional[SmartCompose] = None,
        test_transform: Optional[SmartCompose] = None,
        stratify: Optional[List[str]] = None,
        task_type: Optional[str] = None,
    ) -> None:
        """Initialize TIMIT dataset.

        Args:
            path: Root path to the dataset.
            features_subdir: Subdirectory containing the features.
                If `None`, defaults to audio subdirectory,
                which is `default` for the standard format,
                but can be overridden in the dataset specification.
            seed: Seed for reproducibility.
            metrics: List of metrics to calculate.
            tracking_metric: Metric to track.
            index_column: Index column of the dataframe (typically 'path').
            target_column: Target column of the dataframe. Should be one of:
                - 'dialect_name' for dialect classification
                - 'gender' for gender classification
                - 'sentence_type' for sentence type classification
            file_type: File type of the features (e.g., 'npy').
            file_handler: File handler to load the data.
            features_path: Root path to features. Useful
                when features need to be extracted and stored
                in a different directory than the root of the dataset.
                If `None`, will be set to `path`. Defaults to `None`.
            train_transform: Transform to apply to the training set.
                Defaults to None.
            dev_transform: Transform to apply to the development set.
                Defaults to None.
            test_transform: Transform to apply to the test set.
                Defaults to None.
            stratify: Columns to stratify the dataset on. Defaults to None.
            task_type: Optional task type identifier ('dialect', 'gender',
                'sentence_type'). Used for metadata tracking. Defaults to None.
        """
        self.task_type = task_type

        super().__init__(
            path=path,
            features_subdir=features_subdir,
            seed=seed,
            metrics=metrics,
            tracking_metric=tracking_metric,
            index_column=index_column,
            target_column=target_column,
            file_type=file_type,
            file_handler=file_handler,
            features_path=features_path,
            train_transform=train_transform,
            dev_transform=dev_transform,
            test_transform=test_transform,
            stratify=stratify,
        )

    @staticmethod
    def download(path: str) -> None:  # pragma: no cover
        """Download and prepare the TIMIT dataset.

        Downloads the TIMIT dataset from Google Drive, extracts it,
        and prepares the dataset structure with train/dev/test splits
        for different classification tasks.

        Args:
            path: Path to the directory to download the dataset to.
        """
        print(f"\n{'=' * 60}")
        print("TIMIT Dataset Preparation")
        print(f"{'=' * 60}\n")

        # Determine task type from path
        task_type = _get_task_type_from_path(path)
        print(f"Task type: {task_type}")

        # Check if dataset is already prepared
        if _is_dataset_prepared(path):
            print(f"Dataset already prepared at {path}")
            return

        # Get base directory (parent of task-specific directory)
        base_path = Path(path).parent
        timit_shared_path = base_path / "TIMIT-shared"
        timit_raw_path = timit_shared_path / "TIMIT_raw"

        # Download and extract raw TIMIT data (only once for all tasks)
        if not timit_raw_path.exists():
            _download_timit_archive(str(timit_shared_path))
            _extract_timit_archive(str(timit_shared_path))
        else:
            print(f"TIMIT raw data already exists at {timit_raw_path}")

        # Parse speaker information
        speakers_info = _parse_speaker_info(timit_raw_path)

        # Collect audio files
        audio_files = _collect_audio_files(timit_raw_path, speakers_info)

        # Create shared audio structure with symlinks (only once)
        shared_default_path = timit_shared_path / "default"
        if not shared_default_path.exists():
            _create_shared_audio_structure(audio_files, shared_default_path)

        # Prepare task-specific dataset
        _prepare_task_dataset(path, audio_files, task_type, shared_default_path)

        print(f"\n{'=' * 60}")
        print(f"TIMIT {task_type} dataset preparation complete!")
        print(f"Output directory: {path}")
        print(f"{'=' * 60}\n")


def _get_task_type_from_path(path: str) -> str:
    """Determine task type from path."""
    path_lower = path.lower()
    if "dialect" in path_lower:
        return "dialect"
    if "gender" in path_lower:
        return "gender"
    if "sentence" in path_lower or "sentencetype" in path_lower:
        return "sentence_type"
    raise ValueError(
        f"Cannot determine task type from path '{path}'. "
        "Path should contain 'dialect', 'gender', or 'sentence_type'."
    )


def _is_dataset_prepared(path: str) -> bool:
    """Check if dataset is already prepared."""
    path_obj = Path(path)
    return (
        path_obj.exists()
        and (path_obj / "train.csv").exists()
        and (path_obj / "dev.csv").exists()
        and (path_obj / "test.csv").exists()
        and (path_obj / "default").exists()
    )


def _download_timit_archive(download_path: str) -> None:
    """Download TIMIT archive from Google Drive."""
    os.makedirs(download_path, exist_ok=True)
    zip_path = Path(download_path) / "TIMIT.zip"

    if zip_path.exists():
        # Check if it's a valid zip file (not an HTML error page)
        if zip_path.stat().st_size > 10000:  # More than 10KB
            print(f"TIMIT.zip already downloaded at {zip_path}")
            return
        # Invalid file, remove and re-download
        print(f"Removing invalid TIMIT.zip (size: {zip_path.stat().st_size} bytes)")
        zip_path.unlink()

    print("Downloading TIMIT dataset from Google Drive...")
    print("This may take several minutes (file size: ~400MB)...")

    if GDOWN_AVAILABLE:
        try:
            url = f"https://drive.google.com/uc?id={TIMIT_GDRIVE_ID}"
            gdown.download(url, str(zip_path), quiet=False, fuzzy=True)

            if zip_path.stat().st_size < 10000:
                print(
                    f"Download failed: file too small ({zip_path.stat().st_size} bytes)"
                )
                zip_path.unlink()
                raise Exception("Downloaded file is too small, likely HTML error page")

            print(f"Download complete: {zip_path}")
            return
        except (OSError, RuntimeError) as e:
            print(f"gdown failed: {e}")
            print("Falling back to requests method...")

    print("Using requests method for download...")
    session = requests.Session()

    response = session.get(TIMIT_ZIP_URL, stream=True)

    content_type = response.headers.get("content-type", "")
    if "text/html" in content_type:
        html_content = response.text
        if "confirm=" in html_content:
            import re

            match = re.search(r'name="confirm"\s+value="([^"]+)"', html_content)
            if match:
                confirm_token = match.group(1)
                params = {"id": TIMIT_GDRIVE_ID, "confirm": confirm_token}
                response = session.get(
                    "https://drive.usercontent.google.com/download",
                    params=params,
                    stream=True,
                )
                print("Got virus scan warning, retrying with confirm token...")

    total_size = int(response.headers.get("content-length", 0))

    with (
        open(zip_path, "wb") as f,
        tqdm(
            desc="Downloading TIMIT.zip",
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar,
    ):
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                size = f.write(chunk)
                pbar.update(size)

    print(f"Download complete: {zip_path}")


def _extract_timit_archive(extract_path: str) -> None:
    """Extract TIMIT archive."""
    zip_path = Path(extract_path) / "TIMIT.zip"
    timit_raw_path = Path(extract_path) / "TIMIT_raw"

    if timit_raw_path.exists():
        print(f"TIMIT already extracted at {timit_raw_path}")
        return

    print("Extracting TIMIT archive...")

    with ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)

    # The archive may extract to different paths depending on how it was created
    # Common structures: data/lisa/data/timit/raw/TIMIT/, TIMIT/, timit/TIMIT/
    possible_paths = [
        Path(extract_path) / "data" / "lisa" / "data" / "timit" / "raw" / "TIMIT",
        Path(extract_path) / "TIMIT",
        Path(extract_path) / "timit" / "TIMIT",
    ]

    timit_source = None
    for p in possible_paths:
        if p.exists() and p.is_dir():
            timit_source = p
            break

    if timit_source:
        shutil.move(str(timit_source), str(timit_raw_path))
        print(f"Moved TIMIT data from {timit_source} to {timit_raw_path}")

        # Clean up extracted directory structure
        if (Path(extract_path) / "data").exists():
            shutil.rmtree(Path(extract_path) / "data")
    else:
        raise ValueError(
            f"Could not find TIMIT directory after extraction in {extract_path}"
        )

    print(f"Extraction complete: {timit_raw_path}")


def _parse_speaker_info(timit_raw_path: Path) -> Dict[str, Dict]:
    """Parse SPKRINFO.TXT file to extract speaker metadata."""
    spkr_info_path = timit_raw_path / "DOC" / "SPKRINFO.TXT"
    speakers_info = {}

    # If SPKRINFO.TXT doesn't exist, try alternative locations
    if not spkr_info_path.exists():
        alt_paths = [
            timit_raw_path / "SPKRINFO.TXT",
            timit_raw_path / "doc" / "SPKRINFO.TXT",
            timit_raw_path / "doc" / "spkrinfo.txt",
        ]
        for alt_path in alt_paths:
            if alt_path.exists():
                spkr_info_path = alt_path
                break

    if not spkr_info_path.exists():
        print("Warning: SPKRINFO.TXT not found, speaker info will be inferred")
        return speakers_info

    print(f"Reading speaker info from {spkr_info_path}")

    with open(spkr_info_path, "r") as f:
        for line in f:
            if not line.strip() or line.startswith(";"):
                continue

            parts = line.split()
            if len(parts) >= 4:
                speaker_id = parts[0]
                speakers_info[speaker_id] = {
                    "sex": parts[1],
                    "dialect": parts[2],
                    "dialect_name": DIALECT_MAPPING.get(parts[2], "Unknown"),
                    "use": parts[3],
                }

    print(f"Loaded info for {len(speakers_info)} speakers")
    return speakers_info


def _collect_audio_files(
    timit_raw_path: Path, speakers_info: Dict[str, Dict]
) -> List[Dict]:
    """Collect all audio files from TIMIT dataset."""
    audio_files = []

    for split in ["TRAIN", "TEST"]:
        split_dir = timit_raw_path / split

        # Try lowercase if uppercase doesn't exist
        if not split_dir.exists():
            split_dir = timit_raw_path / split.lower()

        if not split_dir.exists():
            print(f"Warning: {split} directory not found")
            continue

        print(f"Processing {split} directory...")

        for dr_dir in sorted(split_dir.iterdir()):
            if not dr_dir.is_dir():
                continue

            # Check if this is a dialect region directory
            if not (dr_dir.name.startswith("DR") or dr_dir.name.startswith("dr")):
                continue

            for speaker_dir in sorted(dr_dir.iterdir()):
                if not speaker_dir.is_dir():
                    continue

                speaker_id = speaker_dir.name
                speaker_info = speakers_info.get(speaker_id)

                # If no speaker info, infer from directory structure
                if not speaker_info:
                    dialect_num = dr_dir.name.upper().replace("DR", "")
                    speaker_info = {
                        "sex": "M" if speaker_id[0].upper() == "M" else "F",
                        "dialect": dialect_num,
                        "dialect_name": DIALECT_MAPPING.get(dialect_num, "Unknown"),
                        "use": "TRN" if split == "TRAIN" else "TST",
                    }

                # Find WAV files (case insensitive)
                wav_files = list(speaker_dir.glob("*.WAV")) + list(
                    speaker_dir.glob("*.wav")
                )

                for wav_file in sorted(wav_files):
                    sentence_id = wav_file.stem
                    sentence_type = sentence_id[:2].upper()

                    # Create relative path
                    rel_path = f"{split}/{dr_dir.name}/{speaker_id}/{wav_file.name}"

                    audio_files.append(
                        {
                            "path": rel_path,
                            "absolute_path": wav_file,
                            "speaker_id": speaker_id,
                            "sentence_id": sentence_id,
                            "sentence_type": sentence_type,
                            "dialect": speaker_info["dialect"],
                            "dialect_name": speaker_info["dialect_name"],
                            "gender": speaker_info["sex"],
                            "split": split,
                        }
                    )

    print(f"Collected {len(audio_files)} audio files")
    return audio_files


def _create_shared_audio_structure(audio_files: List[Dict], shared_path: Path) -> None:
    """Create shared audio structure with symlinks."""
    shared_path.mkdir(parents=True, exist_ok=True)

    print(f"\nCreating shared audio structure in {shared_path}")

    for file_info in audio_files:
        src_file = file_info["absolute_path"]
        dst_file = shared_path / file_info["path"]

        dst_file.parent.mkdir(parents=True, exist_ok=True)

        if not dst_file.exists():
            try:
                # Try to create symlink, fall back to copy if symlink fails
                try:
                    dst_file.symlink_to(src_file.resolve())
                except (OSError, NotImplementedError):
                    # Symlinks might not be supported, copy instead
                    shutil.copy2(src_file, dst_file)
            except (OSError, FileNotFoundError) as e:
                print(f"Warning: Could not create link/copy for {dst_file}: {e}")

    print("Shared audio structure created")


def _prepare_task_dataset(
    task_path: str, audio_files: List[Dict], task_type: str, shared_default_path: Path
) -> None:
    """Prepare dataset for a specific task."""
    print(f"\n{'=' * 60}")
    print(f"Preparing {task_type.upper()} classification task")
    print(f"{'=' * 60}")

    # Create task directory
    task_path_obj = Path(task_path)
    task_path_obj.mkdir(parents=True, exist_ok=True)

    # Create splits
    train_files, dev_files, test_files = _create_splits(audio_files, task_type)

    # Write CSV files
    _write_csv_files(train_files, dev_files, test_files, task_type, task_path_obj)

    # Create symlink to shared audio
    default_link = task_path_obj / "default"
    if not default_link.exists():
        try:
            # Calculate relative path from task directory to shared default
            rel_path = os.path.relpath(shared_default_path, task_path_obj)
            default_link.symlink_to(rel_path)
            print(f"Created symlink: {default_link} -> {shared_default_path}")
        except (OSError, NotImplementedError):
            # If symlinks not supported, copy the directory
            print(f"Symlinks not supported, copying audio files to {default_link}")
            shutil.copytree(shared_default_path, default_link)


def _create_splits(audio_files: List[Dict], task_type: str, seed: int = 42) -> tuple:
    """Create train/dev/test splits with 70/15/15 ratio."""
    random.seed(seed)

    # For dialect task, exclude SA sentences
    if task_type == "dialect":
        filtered_files = [f for f in audio_files if f["sentence_type"] != "SA"]
    else:
        filtered_files = audio_files

    # Get unique speakers and shuffle
    all_speakers = list({f["speaker_id"] for f in filtered_files})
    random.shuffle(all_speakers)

    # Calculate split sizes (70/15/15)
    total_speakers = len(all_speakers)
    train_size = int(total_speakers * 0.70)
    dev_size = int(total_speakers * 0.15)

    train_speakers = set(all_speakers[:train_size])
    dev_speakers = set(all_speakers[train_size : train_size + dev_size])
    test_speakers = set(all_speakers[train_size + dev_size :])

    # Split files by speaker
    train_split = [f for f in filtered_files if f["speaker_id"] in train_speakers]
    dev_split = [f for f in filtered_files if f["speaker_id"] in dev_speakers]
    test_split = [f for f in filtered_files if f["speaker_id"] in test_speakers]

    print(f"\nSplit statistics for {task_type} task (seed={seed}):")
    print(
        f"  Train: {len(train_split)} files from {len(train_speakers)} speakers "
        f"({100 * len(train_speakers) / total_speakers:.1f}%)"
    )
    print(
        f"  Dev:   {len(dev_split)} files from {len(dev_speakers)} speakers "
        f"({100 * len(dev_speakers) / total_speakers:.1f}%)"
    )
    print(
        f"  Test:  {len(test_split)} files from {len(test_speakers)} speakers "
        f"({100 * len(test_speakers) / total_speakers:.1f}%)"
    )

    return train_split, dev_split, test_split


def _write_csv_files(
    train_files: List[Dict],
    dev_files: List[Dict],
    test_files: List[Dict],
    task_type: str,
    task_dir: Path,
) -> None:
    """Write CSV files for train/dev/test splits."""

    # Determine columns based on task and create row getter function
    def get_row(f: Dict) -> Dict:
        if task_type == "gender":
            return {"path": f["path"], "gender": f["gender"]}
        if task_type == "dialect":
            return {"path": f["path"], "dialect_name": f["dialect_name"]}
        if task_type == "sentence_type":
            return {"path": f["path"], "sentence_type": f["sentence_type"]}
        raise ValueError(f"Unknown task type: {task_type}")

    # Write train.csv
    train_df = pd.DataFrame([get_row(f) for f in train_files])
    train_df.to_csv(task_dir / "train.csv", index=False)

    # Write dev.csv
    dev_df = pd.DataFrame([get_row(f) for f in dev_files])
    dev_df.to_csv(task_dir / "dev.csv", index=False)

    # Write test.csv
    test_df = pd.DataFrame([get_row(f) for f in test_files])
    test_df.to_csv(task_dir / "test.csv", index=False)

    print(f"CSV files written to {task_dir}")
