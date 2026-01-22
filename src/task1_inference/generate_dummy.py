"""
더미 사용자 데이터 생성 스크립트

당근마켓과 유사한 중고거래 플랫폼의 유저 페르소나를 반영한
더미 데이터 30개를 생성합니다.
"""

import json
import random
from pathlib import Path


# 카테고리별 상품 및 관심사 정의
CATEGORIES = {
    "운동_러닝": {
        "items": ["나이키 러닝화", "무릎 보호대", "가민 워치", "러닝 벨트", "스포츠 선글라스"],
        "inference": "운동, 러닝, 건강"
    },
    "운동_요가": {
        "items": ["요가 매트", "폼롤러", "저항 밴드", "요가 블록", "요가 스트랩"],
        "inference": "운동, 요가, 스트레칭"
    },
    "운동_헬스": {
        "items": ["덤벨 세트", "푸쉬업바", "턱걸이 바", "운동 장갑", "쉐이커"],
        "inference": "운동, 헬스, 근력"
    },
    "육아_신생아": {
        "items": ["유모차", "아기띠", "젖병 소독기", "기저귀", "수유쿠션"],
        "inference": "육아, 신생아, 생활"
    },
    "육아_유아": {
        "items": ["블록 장난감", "유아용 식판", "범보 의자", "아기 책", "퍼즐 매트"],
        "inference": "육아, 유아, 놀이"
    },
    "게임_콘솔": {
        "items": ["PS5 듀얼센스", "게이밍 헤드셋", "스팀 기프트카드", "닌텐도 스위치 케이스"],
        "inference": "게임, 콘솔, 엔터테인먼트"
    },
    "게임_PC": {
        "items": ["기계식 키보드", "게이밍 마우스", "대형 마우스패드", "게이밍 모니터"],
        "inference": "게임, PC, 장비"
    },
    "캠핑_장비": {
        "items": ["캠핑 의자", "랜턴", "타프", "텐트", "침낭"],
        "inference": "캠핑, 아웃도어, 여행"
    },
    "캠핑_요리": {
        "items": ["버너", "코펠", "아이스박스", "캠핑 테이블", "화로대"],
        "inference": "캠핑, 요리, 아웃도어"
    },
    "베이킹": {
        "items": ["오븐형 에어프라이어", "실리콘 베이킹몰드", "계량컵", "핸드믹서", "베이킹 온도계"],
        "inference": "베이킹, 홈카페, 요리"
    },
    "개발_재택": {
        "items": ["맥북 스탠드", "코딩용 모니터암", "USB 허브", "인체공학 키보드"],
        "inference": "개발, 재택, 생산성"
    },
    "개발_메이커": {
        "items": ["라즈베리파이 키트", "점퍼 케이블", "브레드보드", "아두이노"],
        "inference": "개발, 메이커, IoT"
    },
    "인테리어_감성": {
        "items": ["무드등", "원목 사이드테이블", "러그", "페이크플랜트"],
        "inference": "인테리어, 홈데코, 감성"
    },
    "인테리어_리빙": {
        "items": ["커튼", "벽걸이 액자", "디퓨저", "라탄 바구니"],
        "inference": "인테리어, 리빙, 홈스타일"
    },
    "사진_필름": {
        "items": ["필름 카메라", "35mm 필름", "카메라 스트랩", "필름 스캐너"],
        "inference": "사진, 필름, 취미"
    },
    "자전거": {
        "items": ["로드자전거 헬멧", "싸이클 장갑", "공기주입 펌프", "자전거 라이트"],
        "inference": "자전거, 라이딩, 운동"
    },
    "등산": {
        "items": ["등산 스틱", "방수 등산화", "배낭", "등산 모자"],
        "inference": "등산, 아웃도어, 건강"
    },
    "낚시": {
        "items": ["낚시대", "릴", "미끼 세트", "낚시 조끼"],
        "inference": "낚시, 레저, 아웃도어"
    },
    "반려견": {
        "items": ["반려견 하네스", "배변패드", "자동 급식기", "강아지 간식"],
        "inference": "반려동물, 강아지, 돌봄"
    },
    "반려묘": {
        "items": ["고양이 캣타워", "스크래처", "고양이 간식", "자동 화장실"],
        "inference": "반려동물, 고양이, 놀이"
    },
    "음악_아날로그": {
        "items": ["LP 턴테이블", "바이닐 레코드", "스피커", "헤드폰 앰프"],
        "inference": "음악, 아날로그, 취향"
    },
    "독서": {
        "items": ["전자책 리더기", "독서등", "북커버", "책갈피 세트"],
        "inference": "독서, 취미, 자기계발"
    },
    "스케이트보드": {
        "items": ["스케이트보드", "보호대 세트", "스트릿 스니커즈", "스케이트 툴"],
        "inference": "스트릿, 보드, 스포츠"
    },
    "드론": {
        "items": ["드론", "여분 배터리", "휴대용 랜딩패드", "ND 필터"],
        "inference": "드론, 촬영, 취미"
    },
    "탁구": {
        "items": ["탁구 라켓", "탁구공 세트", "스포츠 양말", "라켓 케이스"],
        "inference": "운동, 탁구, 건강"
    },
    "커피": {
        "items": ["핸드드립 세트", "커피 원두", "전동 그라인더", "드립 포트"],
        "inference": "커피, 홈카페, 취미"
    },
}

# 혼합 카테고리 유저 (서로 다른 두 카테고리가 섞인 유저)
MIXED_USERS = [
    {
        "items": "아기 장난감, 게이밍 키보드, 소음 방지 헤드셋",
        "target_inference": "육아, 게임, 재택"
    },
    {
        "items": "캠핑 텐트, 커피 원두, 휴대용 핸드드립 세트",
        "target_inference": "캠핑, 커피, 아웃도어"
    },
    {
        "items": "요가 매트, 베이킹 오븐, 앞치마",
        "target_inference": "운동, 베이킹, 라이프스타일"
    },
    {
        "items": "레고 세트, NAS 하드, 랜 케이블",
        "target_inference": "취미, 수집, 개발"
    },
]


def generate_single_category_users(count: int = 26) -> list[dict]:
    """
    단일 카테고리 유저 데이터 생성

    Args:
        count: 생성할 유저 수

    Returns:
        유저 데이터 리스트
    """
    users = []
    category_keys = list(CATEGORIES.keys())

    for i in range(count):
        # 카테고리 순환 선택 (다양성 보장)
        category_key = category_keys[i % len(category_keys)]
        category = CATEGORIES[category_key]

        # 3~4개 아이템 랜덤 선택
        item_count = random.randint(3, 4)
        selected_items = random.sample(category["items"], min(item_count, len(category["items"])))

        users.append({
            "user_id": f"u_{i + 1:03d}",
            "items": ", ".join(selected_items),
            "target_inference": category["inference"]
        })

    return users


def generate_mixed_category_users(start_id: int = 27) -> list[dict]:
    """
    혼합 카테고리 유저 데이터 생성

    Args:
        start_id: 시작 유저 ID 번호

    Returns:
        유저 데이터 리스트
    """
    users = []
    for i, mixed in enumerate(MIXED_USERS):
        users.append({
            "user_id": f"u_{start_id + i:03d}",
            "items": mixed["items"],
            "target_inference": mixed["target_inference"]
        })
    return users


def generate_all_users(total: int = 30) -> list[dict]:
    """
    전체 유저 데이터 생성

    Args:
        total: 총 유저 수

    Returns:
        전체 유저 데이터 리스트
    """
    mixed_count = len(MIXED_USERS)
    single_count = total - mixed_count

    single_users = generate_single_category_users(single_count)
    mixed_users = generate_mixed_category_users(single_count + 1)

    return single_users + mixed_users


def save_to_json(data: list[dict], output_path: str) -> None:
    """
    데이터를 JSON 파일로 저장

    Args:
        data: 저장할 데이터 리스트
        output_path: 출력 파일 경로
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[완료] {len(data)}명의 유저 데이터가 {output_path}에 저장되었습니다.")


def main():
    """메인 실행 함수"""
    output_path = "./data/raw/dummy_users1.json"

    users = generate_all_users(30)
    save_to_json(users, output_path)

    # 생성된 데이터 샘플 출력
    print("\n[샘플] 생성된 데이터:")
    for user in users[:3]:
        print(f"  {user['user_id']}: {user['items'][:40]}...")
    print(f"  ... 외 {len(users) - 3}명")


if __name__ == "__main__":
    main()
