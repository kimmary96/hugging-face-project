import webbrowser
import os
import random
from final_rule_matching import generate_dummy_db, analyze_user_profile, run_matching_system

# ==========================================
# 1. 현실적인 더미 데이터 생성기 (Rich Data Generator)
# ==========================================
REALISTIC_TITLES = {
    "운동": ["보라매공원 러닝 크루", "관악산 주말 등산", "한강 라이딩 (초보환영)", "퇴근 후 배드민턴", "아침 풋살 모임", "실내 클라이밍 체험"],
    "가족/육아": ["봉천동 육아 해방", "공동육아 품앗이", "키즈카페 대관 모임", "유모차 산책 (날씨좋을때)", "우리아이 친구 만들기", "초등맘 정보공유"],
    "취미/오락": ["동네 맛집 도장깨기", "보드게임 한판 (전략)", "주말 영화 관람", "퇴근길 맥주 한잔", "카페 투어 & 수다", "원데이 클래스 공구"],
    "반려동물": ["댕댕이 산책 친구", "반려견 사회화 훈련", "고양이 집사 수다방", "관악산 강아지 등산", "펫카페 정모"],
    "자기계발": ["미라클모닝 챌린지", "영어회화 스터디", "주식/재테크 공부", "독서모임 (한달1권)", "직장인 사이드프로젝트"]
}

LOCATIONS = ["봉천동", "신림동", "서울대입구", "낙성대", "사당동"]
TIMES = ["이번 주 토", "매주 금요일", "내일 저녁", "주말 오전", "평일 저녁"]

def normalize_category(category):
    """사용자 프로필 카테고리를 DB 카테고리로 매핑"""
    category_mapping = {
        "육아": "가족/육아",
        "가족": "가족/육아",
        "운동": "운동",
        "취미": "취미/오락",
        "오락": "취미/오락",
        "반려동물": "반려동물",
        "자기계발": "자기계발",
        "학습": "자기계발"
    }
    return category_mapping.get(category, category)

def generate_rich_db(count=50):
    """스크린샷처럼 리얼한 데이터를 가진 DB 생성"""
    db = []
    # 기본 태그 풀 (기존 로직 활용)
    TAG_POOLS = {
        "숙련도": ["#초보환영", "#고인물"],
        "지속성": ["#정기모임", "#번개"],
        "분위기": ["#가벼움", "#진지함"],
        "연령대": ["#또래중심", "#전연령"]
    }
    
    categories = list(REALISTIC_TITLES.keys())
    
    for i in range(count):
        cat = random.choice(categories)
        title = random.choice(REALISTIC_TITLES[cat])
        
        # 태그 랜덤 생성
        tags = {k: random.choice(v) for k, v in TAG_POOLS.items()}
        
        # 메타 데이터 생성 (장소, 시간, 인원)
        location = random.choice(LOCATIONS)
        time = random.choice(TIMES)
        cur_people = random.randint(3, 10)
        max_people = cur_people + random.randint(2, 10)
        
        # 이미지 플레이스홀더 (카테고리별 색상)
        colors = {"운동": "e3f2fd", "가족/육아": "fce4ec", "취미/오락": "fff3e0", "반려동물": "e8f5e9", "자기계발": "f3e5f5"}
        img_color = colors.get(cat, "eee")
        
        db.append({
            "id": i,
            "category": cat,
            "title": title,
            "tags": tags,
            "meta": {
                "location": location,
                "time": time,
                "people": f"{cur_people}/{max_people}"
            },
            "img_color": img_color
        })
    return db

# ==========================================
# 2. 당근마켓 스타일 HTML 템플릿
# ==========================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>당근모임 추천</title>
    <style>
        @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
        
        body {{
            font-family: Pretendard, -apple-system, BlinkMacSystemFont, system-ui, Roboto, sans-serif;
            background-color: #f2f3f6; /* 당근 배경색 */
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
        }}
        .phone-frame {{
            width: 375px;
            height: 812px;
            background-color: white;
            border-radius: 40px;
            box-shadow: 0 30px 60px rgba(0,0,0,0.12);
            overflow: hidden;
            position: relative;
            border: 8px solid #1a1a1a;
        }}
        /* 헤더 영역 */
        .header {{
            background-color: white;
            padding: 40px 20px 10px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .location-title {{ font-size: 18px; font-weight: 700; color: #212124; }}
        .header-icons span {{ font-size: 20px; margin-left: 15px; cursor: pointer; }}
        
        /* 탭 메뉴 */
        .tab-menu {{
            display: flex;
            padding: 0 20px;
            border-bottom: 1px solid #ececec;
            background: white;
        }}
        .tab {{
            padding: 12px 0;
            margin-right: 20px;
            font-size: 15px;
            color: #868b94;
            font-weight: 600;
            cursor: pointer;
            position: relative;
        }}
        .tab.active {{ color: #212124; }}
        .tab.active::after {{
            content: '';
            position: absolute;
            bottom: 0;
            left: 0;
            width: 100%;
            height: 2px;
            background-color: #212124;
        }}

        /* 메인 컨텐츠 */
        .content {{
            padding: 20px;
            height: 640px;
            overflow-y: auto;
            background-color: #f2f3f6;
        }}
        
        .section-header {{ margin-bottom: 16px; }}
        .section-title {{ font-size: 19px; font-weight: 700; color: #212124; margin-bottom: 4px; }}
        .section-subtitle {{ font-size: 13px; color: #868b94; }}
        .view-all {{ float: right; font-size: 13px; color: #868b94; margin-top: 5px; }}

        /* 모임 카드 스타일 */
        .card {{
            background: white;
            border-radius: 16px;
            padding: 16px;
            margin-bottom: 12px;
            display: flex;
            gap: 16px;
            position: relative;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
            transition: transform 0.1s;
        }}
        .card:active {{ transform: scale(0.98); }}
        
        .card-img {{
            width: 80px;
            height: 80px;
            border-radius: 12px;
            background-size: cover;
            background-position: center;
            flex-shrink: 0;
        }}
        
        .card-info {{ flex-grow: 1; display: flex; flex-direction: column; justify-content: space-between; }}
        
        .card-title {{
            font-size: 16px;
            font-weight: 700;
            color: #212124;
            margin-bottom: 4px;
            line-height: 1.3;
        }}
        
        .card-desc {{
            font-size: 12px;
            color: #868b94;
            margin-bottom: 8px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}
        
        .card-meta {{
            font-size: 11px;
            color: #868b94;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        /* 매칭 점수 뱃지 (우측 상단) */
        .match-badge {{
            position: absolute;
            top: 16px;
            right: 16px;
            border: 1px solid #ff6f0f;
            color: #ff6f0f;
            font-size: 11px;
            font-weight: 700;
            padding: 3px 6px;
            border-radius: 6px;
            background: #fff;
        }}

        /* 추천 이유 팝업 (AI Bubble) */
        .ai-bubble {{
            position: absolute;
            top: 130px; /* 첫 번째 카드 위 */
            left: 50%;
            transform: translateX(-50%);
            width: 85%;
            background: white;
            padding: 16px;
            border-radius: 16px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.15);
            z-index: 10;
            border: 1px solid #eee;
            animation: popIn 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        }}
        .ai-bubble::after {{
            content: '';
            position: absolute;
            bottom: -8px;
            left: 50%;
            transform: translateX(-50%);
            border-width: 8px 8px 0;
            border-style: solid;
            border-color: white transparent transparent transparent;
        }}
        .ai-title {{ font-size: 13px; font-weight: 700; color: #212124; margin-bottom: 6px; display: flex; align-items: center; gap: 5px;}}
        .ai-text {{ font-size: 13px; color: #4d5159; line-height: 1.5; }}
        
        @keyframes popIn {{
            from {{ opacity: 0; transform: translate(-50%, 10px); }}
            to {{ opacity: 1; transform: translate(-50%, 0); }}
        }}

        /* 하단 네비게이션 */
        .bottom-nav {{
            position: absolute;
            bottom: 0;
            width: 100%;
            height: 60px;
            background: white;
            border-top: 1px solid #eee;
            display: flex;
            justify-content: space-around;
            align-items: center;
            padding-bottom: 10px;
        }}
        .nav-item {{ text-align: center; color: #212124; font-size: 10px; display: flex; flex-direction: column; align-items: center; gap: 4px;}}
        .nav-icon {{ font-size: 20px; }}
        .nav-item.active {{ color: #ff6f0f; }}
    </style>
</head>
<body>
    <div class="phone-frame">
        <div class="header">
            <div class="location-title">봉천동</div>
            <div class="header-icons">
                <span>🔍</span><span>🔔</span><span>☰</span>
            </div>
        </div>
        
        <div class="tab-menu">
            <div class="tab">동네생활</div>
            <div class="tab active">모임</div>
            <div class="tab">카페</div>
        </div>

        <div class="content">
            <div class="ai-bubble">
                <div class="ai-title">
                    <span style="background:#ff6f0f; color:white; padding:2px 6px; border-radius:4px; font-size:10px;">AI</span>
                    추천 이유
                </div>
                <div class="ai-text">
                    {reasoning_text}
                </div>
            </div>

            <div class="section-header">
                <div class="view-all">전체보기 ></div>
                <div class="section-title">개인화 추천 TOP 3</div>
                <div class="section-subtitle">내 활동/취향 기반으로 모임을 골랐어요</div>
            </div>

            {cards_html}

            <div style="height: 100px;"></div>
        </div>

        <div class="bottom-nav">
            <div class="nav-item"><span class="nav-icon">🏠</span>홈</div>
            <div class="nav-item"><span class="nav-icon">📍</span>동네생활</div>
            <div class="nav-item"><span class="nav-icon">📍</span>동네지도</div>
            <div class="nav-item"><span class="nav-icon">💬</span>채팅</div>
            <div class="nav-item active"><span class="nav-icon">👤</span>나의 당근</div>
        </div>
    </div>
</body>
</html>
"""

# ==========================================
# 3. 로직 실행 및 HTML 렌더링
# ==========================================
def create_final_ui():
    # 1. 리얼한 데이터셋 생성
    db = generate_rich_db(100)
    
    # 2. Case 3 (육아용품) 데이터로 테스트
    user_input = """
    물건목록: [아기 턱받이, 이유식 용기, 유모차 컵홀더]
    물건개수: 3개
    가격: 중가
    촬영빈도: 자주
    """
    
    print("🔄 [System] 데이터 분석 및 매칭 실행 중...")
    
    # 분석 & 매칭
    profile = analyze_user_profile(user_input)
    if profile is None:
        raise ValueError("analyze_user_profile() returned None. Please check the implementation and input format.")
    profile['category'] = normalize_category(profile['category']) # 매핑 적용
    recommendations = run_matching_system(profile, db)
    
    # 3. AI 추천 사유 생성 (동적)
    # 실제로는 LLM이 생성하지만, 여기서는 룰 기반으로 자연스러운 문장 조합
    cat_name = profile['category']
    tag_keywords = [v.replace("#", "") for k, v in profile['tags'].items()]
    reasoning_text = f"최근 <b>{cat_name}</b> 관련 물품을 구매하셨네요! 성향상 <b>{tag_keywords[0]}</b>이시면서 <b>{tag_keywords[1]}</b>을 선호하시는 것 같아, 꼭 맞는 모임을 찾아왔어요."

    # 4. 카드 HTML 생성
    cards_html = ""
    for item in recommendations:
        info = item['info']
        score = item['score']
        
        # 태그 문자열
        tag_str = f"{info['tags']['숙련도']} · {info['tags']['분위기']} · {info['tags']['연령대']}"
        
        cards_html += f"""
        <div class="card">
            <div class="card-img" style="background-color: #{info['img_color']}; background-image: url('https://placehold.co/80x80/{info['img_color']}/555?text={info['category'][:2]}');"></div>
            <div class="card-info">
                <div>
                    <div class="card-title">{info['title']}</div>
                    <div class="card-desc">{tag_str}</div>
                </div>
                <div class="card-meta">
                    <span>📅 {info['meta']['time']}</span>
                    <span>📍 {info['meta']['location']}</span>
                    <span>👥 {info['meta']['people']} 참여중</span>
                </div>
            </div>
            <div class="match-badge">✨ 나랑 {score}점</div>
        </div>
        """

    # 5. 최종 HTML 저장
    final_html = HTML_TEMPLATE.format(
        reasoning_text=reasoning_text,
        cards_html=cards_html
    )
    
    output_file = "danggeun_ui_final.html"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(final_html)
        
    print(f"\n📱 [Mobile] 당근마켓 UI 생성 완료: {output_file}")
    webbrowser.open('file://' + os.path.realpath(output_file))

if __name__ == "__main__":
    create_final_ui()