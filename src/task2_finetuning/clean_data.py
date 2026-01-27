"""
데이터 전처리 스크립트

CSV 파일을 읽어 학습용 JSONL 형식으로 변환합니다.
당근마켓 모임 데이터를 파인튜닝용 데이터셋으로 가공합니다.
"""

import json
import re

# 파일 경로 (생성된 파일명과 일치시켜주세요)
INPUT_FILE = "synthetic_train_data.jsonl"
OUTPUT_FILE = "cleaned_train_data.jsonl"

# 정규표현식: JSON 객체 { ... } 패턴을 찾기 위함
json_pattern = re.compile(r'\{.*?\}', re.DOTALL)

def clean_and_normalize(text):
    # 1. <think> ... </think> 태그 및 내용 제거
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
    # 2. 마크다운 코드 블록 제거 (```json, ```)
    text = re.sub(r'```json', '', text)
    text = re.sub(r'```', '', text)
    
    # 3. JSON처럼 생긴 가장 첫 번째 덩어리만 추출 (뒤에 붙은 잡설 제거)
    match = json_pattern.search(text)
    if match:
        return match.group(0)
    return None

valid_count = 0

print("🧹 데이터 정제 시작...")

with open(INPUT_FILE, 'r', encoding='utf-8') as fin, \
     open(OUTPUT_FILE, 'w', encoding='utf-8') as fout:
    
    for line_num, line in enumerate(fin):
        try:
            entry = json.loads(line)
            raw_output = entry['output']
            
            # 1차 정제: 텍스트 추출
            cleaned_text = clean_and_normalize(raw_output)
            if not cleaned_text:
                continue

            # JSON 파싱 시도
            try:
                output_json = json.loads(cleaned_text)
            except:
                continue # 파싱 실패시 과감히 버림

            # 2. Key 값 통일 (표준화 작업)
            # 모델이 뱉어낸 다양한 Key들을 'recommendation', 'reasoning' 2개로 강제 통일
            rec_val = (output_json.get('추천 모임') or output_json.get('추천_모임') or 
                       output_json.get('모임명') or output_json.get('Recommended Group') or 
                       output_json.get('추천모임'))
            
            reason_val = (output_json.get('추천 사유') or output_json.get('추천_사유') or 
                          output_json.get('사유') or output_json.get('논리적 추천 사유') or 
                          output_json.get('Reason for Recommendation') or output_json.get('추천이유'))

            # 3. 필수 검증 (값이 없거나, 한글이 없으면 탈락)
            if not rec_val or not reason_val:
                continue
            
            # 사유에 한글이 하나도 없으면(영어/중국어) 탈락
            if not re.search(r'[가-힣]', reason_val):
                continue
            
            # 추천 사유가 너무 짧거나(10자 이하) 잘린 경우 탈락
            if len(reason_val) < 10:
                continue

            # 4. 최종 포맷팅 (Alpaca Style Output)
            # 학습 모델이 헷갈리지 않게 깔끔한 JSON 문자열로 변환
            final_output_obj = {
                "recommendation": rec_val,
                "reasoning": reason_val
            }
            
            new_entry = {
                "instruction": entry['instruction'],
                "input": entry['input'],
                "output": json.dumps(final_output_obj, ensure_ascii=False) # 문자열로 저장
            }
            
            fout.write(json.dumps(new_entry, ensure_ascii=False) + "\n")
            valid_count += 1
            
        except Exception as e:
            # 에러난 라인은 건너뜀
            continue

print(f"✨ 정제 완료! 총 {valid_count}개의 고품질 데이터가 '{OUTPUT_FILE}'에 저장되었습니다.")
print(f"📁 저장된 파일: {OUTPUT_FILE}")