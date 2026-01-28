import webbrowser
import os
import random
from final_rule_matching import analyze_user_profile, run_matching_system, generate_dummy_db

# ==========================================
# 1. 현실적인 더미 데이터 (자기계발 집중)
# ==========================================
# 유저가 제공한 실제 카테고리 데이터 반영
REALISTIC_TITLES = {
    "자기계발": [
        "강남역 파이썬 딥러닝 스터디", "주말 아침 영어회화 (중급)", "직장인 주식/부동산 투자 토론", 
        "판교 IT 개발자 네트워킹", "미라클모닝 챌린지 (기상인증)", "노션으로 일잘러 되기", 
        "매일 글쓰기 습관 만들기", "사이드 프로젝트 팀원 모집", "재테크 독서 모임 (돈의 속성)", "토익/오픽 단기 완성반"
    ],
    # (타 카테고리는 구색 맞추기용)
    "운동": ["퇴근 후 러닝 크루", "주말 등산"],
    "취미/오락": ["맛집 탐방", "보드게임"],
    "가족/육아": ["육아 소통", "키즈카페"],
    "반려동물": ["강아지 산책", "냥이 집사"]
}

LOCATIONS = ["강남역", "판교", "성수동", "역삼동", "을지로"]
TIMES = ["매주 토 오전 10시", "매주 수 저녁 7시", "이번 주 일요일", "평일 새벽 6시", "격주 목요일"]

def normalize_category(category):
    """Normalize category name to match REALISTIC_TITLES keys"""
    category_map = {
        "자기계발": "자기계발",
        "운동": "운동",
        "취미/오락": "취미/오락",
        "가족/육아": "가족/육아",
        "반려동물": "반려동물"
    }
    return category_map.get(category, "자기계발")

def generate_rich_db(count=50):
    db = []
    TAG_POOLS = {
        "숙련도": ["#초보환영", "#고인물"],
        "지속성": ["#정기모임", "#번개"],
        "분위기": ["#가벼움", "#진지함"],
        "연령대": ["#또래중심", "#전연령"]
    }
    categories = list(REALISTIC_TITLES.keys())
    
    for i in range(count):
        # 자기계발 비중을 높임 (데모용)
        if i < 30: cat = "자기계발"
        else: cat = random.choice(categories)
            
        title = random.choice(REALISTIC_TITLES[cat])
        tags = {k: random.choice(v) for k, v in TAG_POOLS.items()}
        
        # 이미지 컬러 (자기계발은 보라/블루 계열)
        colors = {"자기계발": "ede7f6", "운동": "e3f2fd", "취미/오락": "fff3e0", "가족/육아": "fce4ec", "반려동물": "e8f5e9"}
        img_color = colors.get(cat, "eee")
        
        db.append({
            "id": i,
            "category": cat,
            "title": title,
            "tags": tags,
            "meta": {
                "location": random.choice(LOCATIONS),
                "time": random.choice(TIMES),
                "people": f"{random.randint(4,8)}/{random.randint(10,20)}"
            },
            "img_color": img_color
        })
    return db

# ==========================================
# 2. HTML 템플릿 (해쉬태그 UI 개선)
# ==========================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>당근모임 AI 추천</title>
    <style>
        @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
        
        body {{
            font-family: Pretendard, -apple-system, system-ui, Roboto, sans-serif;
            background-color: #f8f9fa;
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
            box-shadow: 0 30px 60px rgba(0,0,0,0.15);
            overflow: hidden;
            position: relative;
            border: 8px solid #111;
        }}
        /* 헤더 */
        .header {{
            padding: 50px 20px 10px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: white;
        }}
        .header-title {{ font-size: 18px; font-weight: 700; }}
        .header-icons {{ font-size: 20px; letter-spacing: 10px; }}

        /* 탭 */
        .tabs {{
            display: flex;
            padding: 0 20px;
            border-bottom: 1px solid #eee;
            margin-bottom: 10px;
        }}
        .tab {{
            padding: 12px 0;
            margin-right: 20px;
            font-size: 15px;
            font-weight: 600;
            color: #888;
            cursor: pointer;
        }}
        .tab.active {{
            color: #222;
            border-bottom: 2px solid #222;
        }}

        /* 컨텐츠 */
        .content {{
            padding: 16px;
            height: 600px;
            overflow-y: auto;
            background-color: #f8f9fa;
        }}

        /* AI 추천 버블 (개선됨) */
        .ai-card {{
            background: white;
            border-radius: 16px;
            padding: 20px;
            margin-bottom: 24px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            position: relative;
            border: 1px solid #ffe0b2;
        }}
        .ai-badge {{
            display: inline-block;
            background: #ff6f0f;
            color: white;
            font-size: 11px;
            font-weight: 800;
            padding: 4px 8px;
            border-radius: 6px;
            margin-bottom: 8px;
        }}
        .ai-title {{
            font-size: 15px;
            font-weight: 700;
            color: #222;
            margin-bottom: 6px;
        }}
        .ai-text {{
            font-size: 13px;
            color: #555;
            line-height: 1.5;
        }}

        /* 모임 리스트 섹션 */
        .section-title {{
            font-size: 18px;
            font-weight: 700;
            margin-bottom: 12px;
            padding-left: 4px;
        }}

        /* 모임 카드 (해쉬태그 스타일 적용) */
        .card {{
            background: white;
            border-radius: 16px;
            padding: 16px;
            margin-bottom: 12px;
            display: flex;
            gap: 14px;
            position: relative;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
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
        
        .card-info {{
            flex-grow: 1;
            display: flex;
            flex-direction: column;
        }}
        
        .card-title {{
            font-size: 15px;
            font-weight: 700;
            color: #222;
            margin-bottom: 6px;
            line-height: 1.3;
        }}
        
        .card-meta {{
            font-size: 11px;
            color: #888;
            margin-bottom: 8px;
        }}

        /* 해쉬태그 스타일 */
        .tags-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 4px;
        }}
        .hash-tag {{
            background-color: #f1f3f5;
            color: #495057;
            font-size: 10px;
            padding: 3px 6px;
            border-radius: 4px;
            font-weight: 500;
        }}

        .match-score {{
            position: absolute;
            top: 16px;
            right: 16px;
            color: #ff6f0f;
            font-size: 11px;
            font-weight: 700;
            background: #fff0e6;
            padding: 4px 8px;
            border-radius: 12px;
        }}

        /* 하단 탭바 */
        .bottom-nav {{
            position: absolute;
            bottom: 0;
            width: 100%;
            height: 65px;
            background: white;
            border-top: 1px solid #eee;
            display: flex;
            justify-content: space-around;
            align-items: center;
            padding-bottom: 15px;
        }}
        .nav-item {{
            text-align: center;
            font-size: 10px;
            color: #888;
            display: flex;
            flex-direction: column;
            gap: 4px;
        }}
        .nav-icon {{ font-size: 22px; margin-bottom: 2px; }}
        .nav-item.active {{ color: #222; font-weight: 600; }}

    </style>
</head>
<body>
    <div class="phone-frame">
        <div class="header">
            <div class="header-title">강남역</div>
            <div class="header-icons">🔍 🔔 ☰</div>
        </div>
        
        <div class="tabs">
            <div class="tab">동네생활</div>
            <div class="tab active">모임</div>
            <div class="tab">동네맛집</div>
        </div>

        <div class="content">
            <div class="ai-card">
                <div class="ai-badge">AI 추천 이유</div>
                <div class="ai-title">"성장을 위한 준비가 완벽하시네요!" 🚀</div>
                <div class="ai-text">
                    {reasoning_text}
                </div>
            </div>

            <div class="section-title">개인화 추천 TOP 3</div>
            
            {cards_html}

            <div style="height: 100px;"></div>
        </div>

        <div class="bottom-nav">
            <div class="nav-item"><span class="nav-icon">🏠</span>홈</div>
            <div class="nav-item"><span class="nav-icon">📍</span>동네생활</div>
            <div class="nav-item"><span class="nav-icon">🗺️</span>동네지도</div>
            <div class="nav-item"><span class="nav-icon">💬</span>채팅</div>
            <div class="nav-item active"><span class="nav-icon">👤</span>나의 당근</div>
        </div>
    </div>
</body>
</html>
"""

# ==========================================
# 3. 로직 실행 (자기계발 시나리오)
# ==========================================
def create_final_ui():
    # 1. DB 생성
    db = generate_rich_db(100)
    
    # 2. 자기계발 고인물 시나리오 (Case B 변형)
    user_input = """
    물건목록: [파이썬 딥러닝 교과서, 기계식 키보드, 노이즈캔슬링 헤드셋, 스탠딩 데스크, 스타벅스 텀블러]
    물건개수: 5개
    가격: 고가
    촬영빈도: 자주
    """
    
    print("🔄 [System] 자기계발(개발자) 시나리오 분석 중...")
    
    # 분석 & 매칭
    profile = analyze_user_profile(user_input)
    if profile is None:
        print("❌ [Error] 프로필 분석 실패")
        return
    profile['category'] = normalize_category(profile['category'])
    recommendations = run_matching_system(profile, db)
    
    # 3. AI 추천 사유 생성 (규칙 기반 자연어 생성)
    # 태그를 직접 말하지 않고, '해석'해서 말하기
    
    # 추론된 태그 가져오기
    is_expert = profile['tags']['숙련도'] == "#고인물"
    is_serious = profile['tags']['분위기'] == "#진지함"
    
    reasoning_text = ""
    if is_expert and is_serious:
        reasoning_text = "전문적인 장비들과 서적을 구매하신 걸 보니, <b>깊이 있는 성장을 추구하는 전문가</b>이시군요! 가벼운 취미보다는, 실력을 확실히 키울 수 있는 <b>밀도 높은 스터디</b>를 찾아왔어요."
    elif is_expert and not is_serious:
        reasoning_text = "실력은 이미 충분하시지만, 너무 딱딱한 건 싫으신가 봐요! <b>능력자들과 즐겁게 네트워킹</b>할 수 있는 모임을 추천해 드려요."
    else:
        reasoning_text = "새로운 배움을 시작하시는군요! <b>기초부터 차근차근, 부담 없이</b> 시작할 수 있는 따뜻한 모임들만 골라봤어요."

    # 4. 카드 HTML 생성 (해쉬태그 방식)
    cards_html = ""
    for item in recommendations:
        info = item['info']
        score = item['score']
        
        # 해쉬태그 HTML 생성
        tags_html = ""
        tag_list = [info['tags']['숙련도'], info['tags']['지속성'], info['tags']['분위기'], info['tags']['연령대']]
        for t in tag_list:
            tags_html += f'<span class="hash-tag">{t}</span> '
        
        cards_html += f"""
        <div class="card">
            <div class="card-img" style="background-color: #{info['img_color']}; background-image: url('https://placehold.co/80x80/{info['img_color']}/555?text={info['category'][:2]}');"></div>
            <div class="card-info">
                <div class="card-title">{info['title']}</div>
                <div class="card-meta">
                    {info['meta']['time']} · {info['meta']['location']} · {info['meta']['people']}
                </div>
                <div class="tags-row">
                    {tags_html}
                </div>
            </div>
            <div class="match-score">⚡ {score}% 매칭</div>
        </div>
        """

    # 5. 최종 HTML 저장
    final_html = HTML_TEMPLATE.format(
        reasoning_text=reasoning_text,
        cards_html=cards_html
    )
    
    output_file = "danggeun_self_dev_ui.html"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(final_html)
        
    print(f"\n📱 [Mobile] 자기계발 테마 UI 생성 완료: {output_file}")
    webbrowser.open('file://' + os.path.realpath(output_file))

if __name__ == "__main__":
    create_final_ui()