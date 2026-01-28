"""
데이터 정제 스크립트 (clean_jsonl.py)

GPT-5.2에서 생성된 raw_data_1000.jsonl을 정제하여 학습용 데이터로 변환합니다.

기능:
1. Markdown Backtick 제거
2. JSON 파싱 검증
3. output 필드 내부 JSON 검증 (category, tags, reasoning)
4. 파싱 오류 자동 복구 시도
5. 통계 출력
"""

import json
import re
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

# 프로젝트 루트 경로
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# 파일 경로
INPUT_FILE = PROJECT_ROOT / "data/raw/data_300.jsonl"
OUTPUT_FILE = PROJECT_ROOT / "data/train_add_300.jsonl"

# 데이터 양식 정의 (두 가지 양식 지원)
# 양식 1 (1~900): category, tags, reasoning
# 양식 2 (901~1000): category, hard_negative
FORMAT_LEGACY = ["category", "tags", "reasoning"]
FORMAT_HARD_NEGATIVE = ["category", "hard_negative"]


def clean_markdown(text: str) -> str:
    """Markdown 코드 블록 및 불필요한 문자 제거"""
    # ```json 또는 ``` 제거
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    # 앞뒤 공백 제거
    text = text.strip()
    return text


def try_fix_json(text: str) -> Optional[str]:
    """
    일반적인 JSON 오류 복구 시도
    - 따옴표 누락 복구
    - trailing comma 제거
    """
    # 1. reasoning 필드에서 따옴표 누락된 경우 복구
    # 예: "reasoning": 전문적인... -> "reasoning": "전문적인..."
    pattern = r'"reasoning"\s*:\s*([^"{\[].+?)(\s*}\s*$)'
    match = re.search(pattern, text)
    if match:
        value = match.group(1).strip()
        # 값 끝에 따옴표가 없으면 추가
        if not value.endswith('"'):
            fixed_value = f'"reasoning": "{value}"'
            text = re.sub(pattern, fixed_value + r'\2', text)

    # 2. trailing comma 제거
    text = re.sub(r',\s*}', '}', text)
    text = re.sub(r',\s*]', ']', text)

    return text


def validate_output_json(output_str: str) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    """
    output 필드 내부 JSON 검증 (두 가지 양식 지원)
    - 양식 1: category, tags, reasoning
    - 양식 2: category, hard_negative

    Returns:
        (is_valid, parsed_json, error_message, format_type)
    """
    try:
        # 1차 파싱 시도
        output_obj = json.loads(output_str)
    except json.JSONDecodeError:
        # 복구 시도
        fixed_str = try_fix_json(output_str)
        if fixed_str is None:
            return False, None, "JSON 파싱 실패: 복구 불가능"
        try:
            output_obj = json.loads(fixed_str)
        except json.JSONDecodeError as e:
            return False, None, f"JSON 파싱 실패: {str(e)}"

    # category 필드 검증 (공통)
    category = output_obj.get("category", "")
    if not category or len(category) < 2:
        return False, None, "category가 비어있거나 너무 짧음"

    # 양식 판별: hard_negative가 있으면 양식 2, 아니면 양식 1
    if "hard_negative" in output_obj:
        # ===== 양식 2: hard_negative 구조 검증 =====
        hard_negative = output_obj.get("hard_negative", {})
        if not isinstance(hard_negative, dict):
            return False, None, "hard_negative가 객체가 아님"

        if "confusing" not in hard_negative:
            return False, None, "hard_negative.confusing 필드 누락"
        if "reason" not in hard_negative:
            return False, None, "hard_negative.reason 필드 누락"

        reason = hard_negative.get("reason", "")
        if len(reason) < 10:
            return False, None, f"hard_negative.reason이 너무 짧음 ({len(reason)}자)"

        if not re.search(r'[가-힣]', reason):
            return False, None, "hard_negative.reason에 한글이 없음"

    else:
        # ===== 양식 1: tags, reasoning 검증 =====
        if "tags" not in output_obj:
            return False, None, "tags 필드 누락"
        if "reasoning" not in output_obj:
            return False, None, "reasoning 필드 누락"

        if not isinstance(output_obj.get("tags"), list):
            return False, None, "tags 필드가 리스트가 아님"

        reasoning = output_obj.get("reasoning", "")
        if len(reasoning) < 10:
            return False, None, f"reasoning이 너무 짧음 ({len(reasoning)}자)"

        if not re.search(r'[가-힣]', reasoning):
            return False, None, "reasoning에 한글이 없음"

    return True, output_obj, ""


def process_line(line: str, line_num: int) -> Tuple[bool, Optional[Dict], str]:
    """
    한 줄의 JSONL 데이터 처리

    Returns:
        (is_valid, processed_entry, error_message)
    """
    # 빈 줄 스킵
    line = line.strip()
    if not line:
        return False, None, "빈 줄"

    # Markdown 정제
    line = clean_markdown(line)

    # 전체 라인 JSON 파싱
    try:
        entry = json.loads(line)
    except json.JSONDecodeError:
        # JSON 복구 시도
        fixed_line = try_fix_json(line)
        if fixed_line is None:
            return False, None, "라인 JSON 파싱 실패: 복구 불가능"
        try:
            entry = json.loads(fixed_line)
        except json.JSONDecodeError as e:
            return False, None, f"라인 JSON 파싱 실패: {str(e)}"

    # 필수 필드 확인
    required_fields = ["instruction", "input", "output"]
    for field in required_fields:
        if field not in entry:
            return False, None, f"필수 필드 누락: {field}"

    # output 필드 검증
    output_str = entry["output"]

    # output이 이미 dict인 경우 (잘못된 형식이지만 처리)
    if isinstance(output_str, dict):
        output_obj = output_str
        # category는 필수
        if "category" not in output_obj:
            return False, None, "output에 category 필드 누락"
        # 양식 1 또는 양식 2 중 하나 충족 필요
        has_legacy = "tags" in output_obj and "reasoning" in output_obj
        has_hard_neg = "hard_negative" in output_obj
        if not has_legacy and not has_hard_neg:
            return False, None, "output 양식 불일치 (tags/reasoning 또는 hard_negative 필요)"
    else:
        # output이 문자열인 경우 (정상)
        is_valid, output_obj, error_msg = validate_output_json(output_str)
        if not is_valid:
            return False, None, error_msg

    # 정제된 데이터 구성
    processed_entry = {
        "instruction": entry["instruction"],
        "input": entry["input"],
        "output": json.dumps(output_obj, ensure_ascii=False)
    }

    return True, processed_entry, ""


def main():
    print("=" * 60)
    print("데이터 정제 시작")
    print(f"입력: {INPUT_FILE}")
    print(f"출력: {OUTPUT_FILE}")
    print("=" * 60)

    # 입력 파일 존재 확인
    if not INPUT_FILE.exists():
        print(f"[ERROR] 입력 파일이 존재하지 않습니다: {INPUT_FILE}")
        return

    # 통계 변수
    total_lines = 0
    valid_count = 0
    duplicate_count = 0
    error_counts = {}

    # 중복 체크용 Set (전체 라인 기준)
    seen_lines = set()

    # 출력 디렉토리 생성
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(INPUT_FILE, 'r', encoding='utf-8') as fin, \
         open(OUTPUT_FILE, 'w', encoding='utf-8') as fout:

        for line_num, line in enumerate(fin, start=1):
            total_lines += 1

            is_valid, processed_entry, error_msg = process_line(line, line_num)

            if is_valid:
                # 중복 체크 (전체 라인 기준)
                line_key = json.dumps(processed_entry, ensure_ascii=False, sort_keys=True)
                if line_key in seen_lines:
                    duplicate_count += 1
                    continue  # 중복이면 스킵

                seen_lines.add(line_key)
                fout.write(json.dumps(processed_entry, ensure_ascii=False) + "\n")
                valid_count += 1
            else:
                # 오류 유형별 집계
                error_type = error_msg.split(":")[0] if ":" in error_msg else error_msg
                error_counts[error_type] = error_counts.get(error_type, 0) + 1

                # 처음 5개 오류만 상세 출력
                if sum(error_counts.values()) <= 5:
                    print(f"  [Line {line_num}] {error_msg}")

    # 결과 출력
    print("\n" + "=" * 60)
    print("정제 완료!")
    print("=" * 60)
    print(f"총 라인 수: {total_lines}")
    if total_lines == 0:
        print("?? ???: 0 (0.0%)")
        print("?? ???: 0")
        print("[WARN] ?? ??? ??????. ??? ??/?? ??? ?????")
        print(f"??? ??: {OUTPUT_FILE}")
        return
    print(f"유효 데이터: {valid_count} ({valid_count/total_lines*100:.1f}%)")
    print(f"제외 데이터: {total_lines - valid_count}")

    if error_counts:
        print("\n[오류 유형별 집계]")
        for error_type, count in sorted(error_counts.items(), key=lambda x: -x[1]):
            print(f"  - {error_type}: {count}건")

    print(f"\n저장된 파일: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
