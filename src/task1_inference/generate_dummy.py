"""
더미 사용자 데이터 생성 스크립트

이 스크립트는 테스트용 더미 사용자 데이터를 생성합니다.
각 사용자는 상품 목록과 예상 관심사를 포함합니다.
"""

import json
import random
from typing import Any


def generate_user_data(user_id: int) -> dict[str, Any]:
    """
    더미 사용자 데이터 1건 생성

    Args:
        user_id: 사용자 고유 ID

    Returns:
        사용자 데이터 딕셔너리 (user_id, items, target_inference 포함)
    """
    pass


def generate_items_by_category(category: str, count: int = 5) -> list[str]:
    """
    카테고리별 상품 목록 생성

    Args:
        category: 상품 카테고리 (예: '운동', '육아', '게이밍')
        count: 생성할 상품 수

    Returns:
        상품명 리스트
    """
    pass


def save_to_json(data: list[dict], output_path: str) -> None:
    """
    데이터를 JSON 파일로 저장

    Args:
        data: 저장할 데이터 리스트
        output_path: 출력 파일 경로
    """
    pass


def main():
    """메인 실행 함수"""
    pass


if __name__ == "__main__":
    main()
