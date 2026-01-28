"""
Data cleaning and validation utilities.
"""
import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import CATEGORIES


class DataCleaner:
    """Data cleaning and validation for training datasets."""

    REQUIRED_FIELDS = ['instruction', 'input', 'output']

    @classmethod
    def clean_jsonl(
        cls,
        input_path: Path,
        output_path: Path,
        validate_output: bool = True
    ) -> Tuple[int, int]:
        """
        Clean and validate JSONL file.

        Args:
            input_path: Path to input JSONL file
            output_path: Path to save cleaned data
            validate_output: Whether to validate output JSON structure

        Returns:
            Tuple of (valid_count, invalid_count)
        """
        valid_count = 0
        invalid_count = 0

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(input_path, 'r', encoding='utf-8') as fin, \
             open(output_path, 'w', encoding='utf-8') as fout:

            for line in fin:
                line = line.strip()
                if not line:
                    continue

                cleaned = cls.clean_record(line, validate_output)
                if cleaned:
                    fout.write(json.dumps(cleaned, ensure_ascii=False) + '\n')
                    valid_count += 1
                else:
                    invalid_count += 1

        return valid_count, invalid_count

    @classmethod
    def clean_record(
        cls,
        line: str,
        validate_output: bool = True
    ) -> Optional[Dict]:
        """
        Clean and validate a single JSONL record.

        Args:
            line: Raw JSON line string
            validate_output: Whether to validate output field

        Returns:
            Cleaned dictionary or None if invalid
        """
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            # Try to fix common JSON issues
            record = cls._try_fix_json(line)
            if not record:
                return None

        # Check required fields
        for field in cls.REQUIRED_FIELDS:
            if field not in record:
                return None

        # Clean fields
        record['instruction'] = cls._clean_text(record['instruction'])
        record['input'] = cls._clean_text(record['input'])

        # Validate output if required
        if validate_output:
            output_valid = cls._validate_output(record['output'])
            if not output_valid:
                return None

        return record

    @classmethod
    def _clean_text(cls, text: str) -> str:
        """Clean text by removing markdown artifacts and normalizing whitespace."""
        # Remove markdown code blocks
        text = re.sub(r'```[a-z]*\n?', '', text)
        text = re.sub(r'```', '', text)

        # Normalize whitespace
        text = ' '.join(text.split())

        return text.strip()

    @classmethod
    def _validate_output(cls, output: str) -> bool:
        """Validate that output contains valid category."""
        try:
            # Try parsing as JSON
            if isinstance(output, str):
                output_dict = json.loads(output)
            else:
                output_dict = output

            category = output_dict.get('category', '')
            return category in CATEGORIES

        except (json.JSONDecodeError, AttributeError):
            return False

    @classmethod
    def _try_fix_json(cls, line: str) -> Optional[Dict]:
        """Attempt to fix malformed JSON."""
        # Find JSON block
        start = line.find('{')
        end = line.rfind('}')

        if start == -1 or end == -1:
            return None

        try:
            return json.loads(line[start:end+1])
        except json.JSONDecodeError:
            return None

    @classmethod
    def remove_duplicates(cls, data: List[Dict], key: str = 'input') -> List[Dict]:
        """Remove duplicate records based on a key field."""
        seen = set()
        unique = []

        for record in data:
            value = record.get(key, '')
            if value not in seen:
                seen.add(value)
                unique.append(record)

        return unique

    @classmethod
    def filter_by_category(
        cls,
        data: List[Dict],
        categories: List[str]
    ) -> List[Dict]:
        """Filter records by category."""
        filtered = []
        for record in data:
            try:
                output = json.loads(record.get('output', '{}'))
                if output.get('category') in categories:
                    filtered.append(record)
            except json.JSONDecodeError:
                continue
        return filtered

    @classmethod
    def get_category_distribution(cls, data: List[Dict]) -> Dict[str, int]:
        """Get distribution of categories in dataset."""
        distribution = {cat: 0 for cat in CATEGORIES}

        for record in data:
            try:
                output = json.loads(record.get('output', '{}'))
                category = output.get('category', '')
                if category in distribution:
                    distribution[category] += 1
            except json.JSONDecodeError:
                continue

        return distribution
