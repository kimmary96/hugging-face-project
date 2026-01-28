import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from final_rule_matching import generate_dummy_db, analyze_user_profile, run_matching_system

# ==========================================
# 1. 카테고리 정규화 (Mapping Layer) - [수정됨]
# ==========================================
CATEGORY_MAP = {
    "육아": "가족/육아",
    "운동": "운동",
    "취미": "취미/오락",
    "오락": "취미/오락",
    "반려": "반려동물",
    "동물": "반려동물",
    "공부": "자기계발",
    "계발": "자기계발"
}

def normalize_category(llm_category):
    """LLM이 뱉은 카테고리를 DB 표준명으로 변환"""
    for key, value in CATEGORY_MAP.items():
        if key in llm_category:
            return value
    return llm_category # 매핑 안 되면 그대로 반환

# ==========================================
# 2. 시각화 함수 (Heatmap Generator)
# ==========================================
def create_matching_report(user_profile, recommendations, filename="matching_report.png"):
    if not recommendations:
        print("❌ 시각화할 추천 데이터가 없습니다.")
        return

    # 데이터 준비
    meeting_titles = [m['info']['title'] for m in recommendations]
    criteria = ["숙련도", "지속성", "분위기", "연령대"]
    
    # 점수 매트릭스 생성 (일치하면 25, 불일치하면 0)
    data = []
    for item in recommendations:
        row = []
        u_tags = user_profile['tags']
        m_tags = item['info']['tags']
        
        for c in criteria:
            if u_tags[c] == m_tags[c]:
                row.append(25) # 일치 점수
            else:
                row.append(0)  # 불일치
        data.append(row)
    
    # 시각화 설정
    plt.rc('font', family='Malgun Gothic') # 윈도우 한글 폰트
    plt.figure(figsize=(10, 6))
    
    # 히트맵 그리기
    sns.heatmap(data, annot=True, fmt="d", cmap="Greens", 
                xticklabels=criteria, yticklabels=meeting_titles,
                vmin=0, vmax=25, cbar_kws={'label': '매칭 점수 (항목당 25점)'})
    
    plt.title(f"[{user_profile['category']}] 유저 성향 vs 추천 모임 매칭 분석", fontsize=15, pad=20)
    plt.xlabel("매칭 기준", fontsize=12)
    plt.ylabel("추천 모임", fontsize=12)
    
    # 결과 저장
    plt.tight_layout()
    plt.savefig(filename)
    print(f"📊 [Report] 시각화 리포트 저장 완료: {filename}")
    # plt.show() # 팝업으로 보고 싶으면 주석 해제

# ==========================================
# 3. 통합 실행 (Main)
# ==========================================
if __name__ == "__main__":
    # 1. DB 준비
    db = generate_dummy_db(100)
    
    # 2. 문제의 Case 3 다시 실행 (육아 용품)
    failed_input = """
    물건목록: [아기 턱받이, 이유식 용기, 유모차 컵홀더]
    물건개수: 3개
    가격: 중가
    촬영빈도: 자주
    """
    
    print("🔄 [Re-Test] Case 3 재도전 (매핑 레이어 적용)...")
    
    # 분석
    profile = analyze_user_profile(failed_input)
    
    if profile is None:
        print("❌ analyze_user_profile()가 None을 반환했습니다. 입력값 또는 함수 구현을 확인하세요.")
        exit(1)
    
    # [핵심 수정] 카테고리 정규화 적용!
    original_cat = profile['category']
    profile['category'] = normalize_category(original_cat)
    print(f"   🛠️ [Fix] 카테고리 보정: '{original_cat}' -> '{profile['category']}'")
    
    # 매칭
    recommendations = run_matching_system(profile, db)
    
    # 결과 출력
    if recommendations:
        print(f"\n✅ 드디어 찾았습니다! ({len(recommendations)}개)")
        for rank, item in enumerate(recommendations):
            print(f"   🥇 {rank+1}위: {item['info']['title']} (점수: {item['score']})")
            
        # 3. 시각화 리포트 생성
        create_matching_report(profile, recommendations)
    else:
        print("❌ 여전히 매칭되는 모임이 없습니다. (DB에 해당 태그 조합이 없을 수 있음)")