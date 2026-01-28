"""
Data loading utilities for JSONL and JSON files.
"""
import json
from pathlib import Path
from typing import List, Dict, Generator, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import DATA_DIR, RAW_DATA_DIR, DataFiles


class DataLoader:
    """Unified data loader for training and evaluation datasets."""

    @staticmethod
    def load_jsonl(file_path: Path, limit: Optional[int] = None) -> List[Dict]:
        """
        Load JSONL file into list of dictionaries.

        Args:
            file_path: Path to JSONL file
            limit: Maximum number of records to load (None for all)

        Returns:
            List of parsed JSON objects
        """
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return data

    @staticmethod
    def load_jsonl_stream(file_path: Path) -> Generator[Dict, None, None]:
        """
        Stream JSONL file line by line (memory efficient).

        Args:
            file_path: Path to JSONL file

        Yields:
            Parsed JSON objects one at a time
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue

    @staticmethod
    def load_json(file_path: Path) -> Dict:
        """Load JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def save_jsonl(data: List[Dict], file_path: Path):
        """Save list of dictionaries to JSONL file."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    @staticmethod
    def save_json(data: Dict, file_path: Path, indent: int = 2):
        """Save dictionary to JSON file."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)

    @staticmethod
    def count_lines(file_path: Path) -> int:
        """Count number of lines in a file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return sum(1 for line in f if line.strip())

    @classmethod
    def load_training_data(cls) -> List[Dict]:
        """Load main training dataset."""
        return cls.load_jsonl(DataFiles.TRAIN_FINAL)

    @classmethod
    def load_test_data(cls) -> List[Dict]:
        """Load test dataset for evaluation."""
        return cls.load_jsonl(DataFiles.TEST_DATA)

    @classmethod
    def merge_datasets(cls, *file_paths: Path) -> List[Dict]:
        """Merge multiple JSONL files into one list."""
        merged = []
        for path in file_paths:
            if path.exists():
                merged.extend(cls.load_jsonl(path))
        return merged
