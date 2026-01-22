"""
데이터 전처리 스크립트

CSV 파일을 읽어 학습용 JSONL 형식으로 변환합니다.
당근마켓 모임 데이터를 파인튜닝용 데이터셋으로 가공합니다.
"""

import json
import pandas as pd
from typing import Any


def load_csv(file_path: str) -> pd.DataFrame:
    """
    CSV 파일 로드

    Args:
        file_path: CSV 파일 경로

    Returns:
        로드된 DataFrame
    """
    pass


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    데이터 정제

    Args:
        df: 원본 DataFrame

    Returns:
        정제된 DataFrame
    """
    pass


def create_training_example(row: pd.Series) -> dict[str, Any]:
    """
    학습 데이터 샘플 생성

    Args:
        row: DataFrame의 한 행

    Returns:
        학습용 데이터 딕셔너리
    """
    pass


def convert_to_jsonl(df: pd.DataFrame, output_path: str) -> None:
    """
    DataFrame을 JSONL 형식으로 변환하여 저장

    Args:
        df: 변환할 DataFrame
        output_path: 출력 JSONL 파일 경로
    """
    pass


def main():
    """메인 실행 함수"""
    pass


if __name__ == "__main__":
    main()
