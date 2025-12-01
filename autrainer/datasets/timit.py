from functools import cached_property
import os
from typing import Dict, List, Optional, Union

from omegaconf import DictConfig
import pandas as pd

from autrainer.transforms import SmartCompose

from .abstract_dataset import BaseClassificationDataset


class TIMIT(BaseClassificationDataset):
    """TIMIT dataset for various classification tasks.

    The TIMIT corpus is a well-known speech database containing broadband
    recordings of 630 speakers of eight major dialects of American English,
    each reading ten phonetically rich sentences.

    This dataset class supports multiple classification tasks:
    - Dialect classification (8 dialects)
    - Gender classification (Male/Female)
    - Sentence type classification (SA/SI/SX)

    The dataset expects the following structure:
    - path/
      - train.csv (with columns: path, target_column)
      - dev.csv (with columns: path, target_column)
      - test.csv (with columns: path, target_column)
      - features_subdir/ (e.g., log_mel_16k/)
        - [preprocessed features as .npy files]

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

        # Validate the dataset structure
        self._validate_timit_structure()

    def _validate_timit_structure(self) -> None:
        """Validate that the TIMIT dataset has the expected structure."""
        # Check that CSV files exist and have the correct columns
        required_columns = [self.index_column, self.target_column]

        for split_name, df in [
            ("train", self.df_train),
            ("dev", self.df_dev),
            ("test", self.df_test),
        ]:
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(
                        f"Column '{col}' not found in {split_name}.csv. "
                        f"Available columns: {list(df.columns)}"
                    )

            # Validate that paths are relative and follow TIMIT structure
            if not all(df[self.index_column].str.contains("/")):
                raise ValueError(
                    f"{split_name}.csv should contain relative paths "
                    "with directory structure (e.g., TRAIN/DR1/FDAW0/SA1.WAV)"
                )

    @cached_property
    def class_distribution(self) -> Dict[str, int]:
        """Get the class distribution in the training set.

        Returns:
            Dictionary mapping class names to their counts.
        """
        return dict(self.df_train[self.target_column].value_counts())

    @cached_property
    def dataset_info(self) -> Dict[str, any]:
        """Get comprehensive dataset information.

        Returns:
            Dictionary containing dataset statistics and metadata.
        """
        return {
            "task_type": self.task_type,
            "target_column": self.target_column,
            "num_classes": len(self.target_transform),
            "class_names": self.target_transform.labels,
            "train_samples": len(self.df_train),
            "dev_samples": len(self.df_dev),
            "test_samples": len(self.df_test),
            "total_samples": len(self.df_train) + len(self.df_dev) + len(self.df_test),
            "class_distribution": self.class_distribution,
            "features_subdir": self.features_subdir,
            "file_type": self.file_type,
        }

    def get_speaker_id(self, file_path: str) -> str:
        """Extract speaker ID from TIMIT file path.

        TIMIT paths follow the pattern: SPLIT/DIALECT/SPEAKER_ID/UTTERANCE.WAV

        Args:
            file_path: TIMIT file path (e.g., 'TRAIN/DR1/FDAW0/SA1.WAV')

        Returns:
            Speaker ID (e.g., 'FDAW0')
        """
        parts = file_path.split('/')
        if len(parts) >= 3:
            return parts[2]
        raise ValueError(f"Invalid TIMIT path format: {file_path}")

    def get_dialect_region(self, file_path: str) -> str:
        """Extract dialect region from TIMIT file path.

        Args:
            file_path: TIMIT file path (e.g., 'TRAIN/DR1/FDAW0/SA1.WAV')

        Returns:
            Dialect region code (e.g., 'DR1')
        """
        parts = file_path.split('/')
        if len(parts) >= 2:
            return parts[1]
        raise ValueError(f"Invalid TIMIT path format: {file_path}")

    @cached_property
    def speaker_ids(self) -> Dict[str, List[str]]:
        """Get speaker IDs for each split.

        Returns:
            Dictionary mapping split names to lists of speaker IDs.
        """
        return {
            "train": sorted(self.df_train[self.index_column].apply(
                self.get_speaker_id
            ).unique().tolist()),
            "dev": sorted(self.df_dev[self.index_column].apply(
                self.get_speaker_id
            ).unique().tolist()),
            "test": sorted(self.df_test[self.index_column].apply(
                self.get_speaker_id
            ).unique().tolist()),
        }

    @staticmethod
    def download(path: str) -> None:
        """TIMIT is a licensed dataset and cannot be automatically downloaded.

        To use this dataset:
        1. Obtain the TIMIT corpus from LDC: https://catalog.ldc.upenn.edu/LDC93S1
        2. Prepare the dataset structure with train.csv, dev.csv, test.csv
        3. Extract features and place them in the features_subdir

        Args:
            path: Path where the dataset should be located.

        Raises:
            NotImplementedError: Always, as TIMIT requires manual setup.
        """
        raise NotImplementedError(
            "TIMIT is a licensed dataset and must be obtained from LDC. "
            "Please visit https://catalog.ldc.upenn.edu/LDC93S1 for access. "
            "After obtaining the dataset, prepare the CSV files and extract features "
            "according to the TIMIT dataset class documentation."
        )
