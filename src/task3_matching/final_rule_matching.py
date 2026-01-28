import random
import json
import re
from unsloth import FastLanguageModel

# ==========================================
# 1. World: 당근마켓 모임 DB 생성 (Dummy)
# ==========================================
# 4가지 태그 분류 (고정된 선택지)
TAG_POOLS = {
    "숙련도": ["#초보환영", "#고인물"],
    "지속성": ["#정기모임", "#번개"],
    "분위기": ["#가벼움", "#진지함"],
    "연령대": ["#또래중심", "#전연령"]
}

CATEGORIES = ["운동", "가족/육아", "취미/오락", "반려동물", "자기계발"]

def generate_dummy_db(count=50):
    """랜덤한 태그 조합을 가진 모임 50개 생성"""
    db = []
    for i in range(count):
        cat = random.choice(CATEGORIES)
        
        # 모임마다 랜덤하게 성격 부여
        tags = {
            "숙련도": random.choice(TAG_POOLS["숙련도"]),
            "지속성": random.choice(TAG_POOLS["지속성"]),
            "분위기": random.choice(TAG_POOLS["분위기"]),
            "연령대": random.choice(TAG_POOLS["연령대"])
        }
        
        # 모임 제목 생성 (예: [운동] 배드민턴 모임 #초보환영...)
        title = f"{cat} 모임 {i+1}호"
        
        db.append({
            "id": i,
            "category": cat,
            "title": title,
            "tags": tags
        })
    return db

# ==========================================
# 2. Brain: 유저 분석 및 태그 도출 (LLM)
# ==========================================
# 모델 로드 (이미 로드되어 있다면 생략 가능)
model_path = "outputs_1.7b"
print(f"🧠 [Brain] 분석 모델 로드 중... ({model_path})")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path,
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

def analyze_user_profile(user_input_str):
    """
    유저 입력(물건목록, 가격, 빈도) -> 규칙에 따른 태그 도출
    """
    print(f"\n🔍 [User Analysis] 유저 데이터 분석 중...")
    print(f"   입력값: {user_input_str}")
    
    # 제시해주신 규칙(Rule)을 시스템 프롬프트에 그대로 이식
    system_prompt = """
    당신은 사용자 구매 데이터를 기반으로 '모임 추천 태그'를 결정하는 AI 엔진입니다.
    아래 [결정 로직]을 엄격하게 준수하여 사용자의 태그를 추출하세요.

    [결정 로직]
    1. 카테고리: 물건 목록에서 가장 많이 등장하는 카테고리 (운동, 가족/육아, 취미/오락, 반려동물, 자기계발)
    2. 숙련도:
       - 물건 개수 5개 이하 OR 가격 '저가' -> #초보환영
       - 물건 개수 6개 이상 AND 가격 '중가'/'고가' -> #고인물
    3. 지속성:
       - 촬영빈도 '자주' -> #정기모임
       - 촬영빈도 '가끔' -> #번개
    4. 분위기:
       - 가격 '고가' -> #진지함
       - 가격 '중가'/'저가' -> #가벼움
    5. 연령대:
       - 카테고리가 '운동' 또는 '취미/오락' -> #또래중심
       - 그 외 카테고리 -> #전연령

    [출력 형식 (JSON Only)]
    {
        "category": "카테고리명",
        "tags": {
            "숙련도": "#태그명",
            "지속성": "#태그명",
            "분위기": "#태그명",
            "연령대": "#태그명"
        },
        "reasoning": "결정 이유 요약"
    }
    """
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input_str}
    ]
    
    inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False, return_tensors="pt").to("cuda")
    
    outputs = model.generate(input_ids=inputs, max_new_tokens=512, temperature=0.1) # 로직 수행이므로 창의성 낮춤
    decoded = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    
    try:
        json_str = decoded[decoded.find("{"):decoded.rfind("}")+1]
        profile = json.loads(json_str)
        print(f"   👉 [Tagging Result]: {profile['category']} / {profile['tags']}")
        return profile
    except:
        print(f"   ⚠️ 파싱 실패. 원문: {decoded}")
        return None

# ==========================================
# 3. Matching Engine: 3단계 필터링 & 스코어링
# ==========================================
def run_matching_system(user_profile, meeting_db):
    print(f"🚀 [Matching] 매칭 엔진 가동...")
    
    candidates = []
    
    # ---------------------------------------
    # 1단계 (Hard Filter): 카테고리 일치 여부
    # ---------------------------------------
    target_category = user_profile['category']
    filtered_db = [m for m in meeting_db if m['category'] == target_category]
    
    print(f"   [Step 1] Hard Filter: '{target_category}' 카테고리 모임 {len(filtered_db)}개 발견")
    
    if not filtered_db:
        return []

    # ---------------------------------------
    # 2단계 (Soft Score): 태그 매칭 점수 계산
    # ---------------------------------------
    # 태그 4개 중 몇 개가 맞는지 계산 (만점 100점, 개당 25점)
    for meeting in filtered_db:
        score = 0
        match_details = []
        
        user_tags = user_profile['tags']
        meeting_tags = meeting['tags']
        
        # 4가지 속성 비교
        for key in ["숙련도", "지속성", "분위기", "연령대"]:
            if user_tags[key] == meeting_tags[key]:
                score += 25
                match_details.append(f"{key}(⭕)")
            else:
                match_details.append(f"{key}(❌)")
        
        candidates.append({
            "info": meeting,
            "score": score,
            "details": match_details
        })

    # ---------------------------------------
    # 3단계 (Top-N): 점수순 정렬 및 상위 추출
    # ---------------------------------------
    # 점수 높은 순 -> 점수 같으면 ID순
    candidates.sort(key=lambda x: x['score'], reverse=True)
    
    return candidates[:3] # Top 3 반환

# ==========================================
# 4. Main Execution
# ==========================================
if __name__ == "__main__":
    # 1. DB 생성
    db = generate_dummy_db(100) # 모임 100개 생성
    print(f"✅ 가상 모임 데이터베이스 구축 완료 (총 {len(db)}개)")

    # 2. 테스트 케이스 (빈도 포함, 물건 개수 다양화)
    test_cases = [
        # Case A: 운동 / 초보 / 정기 / 가벼움 / 또래
        """
        물건목록: [배드민턴 입문용 라켓, 셔틀콕 12입, 스포츠 양말, 손목 보호대]
        물건개수: 4개
        가격: 저가
        촬영빈도: 자주
        """,
        
        # Case B: 취미 / 고인물 / 번개 / 진지함 / 또래
        """
        물건목록: [전문가용 DSLR 카메라, 광각 렌즈, 삼각대, 카메라 가방, 렌즈 필터, 청소 키트, 메모리 카드]
        물건개수: 7개
        가격: 고가
        촬영빈도: 가끔
        """,
        
        # Case C: 육아 / 초보 / 정기 / 가벼움 / 전연령
        """
        물건목록: [아기 턱받이, 이유식 용기, 유모차 컵홀더]
        물건개수: 3개
        가격: 중가
        촬영빈도: 자주
        """
    ]

    for i, case in enumerate(test_cases):
        print("\n" + "="*70)
        print(f"🧪 [Test Case {i+1}]")
        print("="*70)
        
        # 1. 유저 분석
        profile = analyze_user_profile(case.strip())
        if not profile: continue
        
        # 2. 매칭 수행
        recommendations = run_matching_system(profile, db)
        
        # 3. 결과 출력
        print(f"\n🏆 [최종 추천 결과 (Top 3)]")
        if not recommendations:
            print("   👉 조건에 맞는 모임이 없습니다.")
        else:
            for rank, item in enumerate(recommendations):
                m = item['info']
                print(f"   🥇 {rank+1}위 [매칭률 {item['score']}%] {m['title']}")
                print(f"      └ 태그: {m['tags']}")
                print(f"      └ 상세: {', '.join(item['details'])}")