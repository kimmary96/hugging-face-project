"""
Central prompt templates for the project.
All system prompts, instructions, and templates are defined here.
"""

# ==========================================
# Alpaca Format Template (for Training)
# ==========================================
ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

# ==========================================
# Inference Instructions
# ==========================================
INSTRUCTION_CATEGORY_CLASSIFICATION = """물건 목록과 구매 패턴을 분석하여 적합한 모임 카테고리를 추천하세요.

카테고리는 다음 5개 중 하나를 선택하세요:
- 가족/육아
- 반려동물
- 운동
- 자기계발
- 취미/오락

반드시 아래 JSON 형식으로만 응답하세요:
{"category": "선택한 카테고리"}"""

INSTRUCTION_SIMPLE = "물건 목록과 구매 패턴을 분석하여 적합한 모임 카테고리를 추천하세요."

INSTRUCTION_FULL = "유저의 촬영 물건과 패턴을 분석하여 모임 카테고리와 분위기를 추천하세요."

# ==========================================
# System Prompts
# ==========================================
SYSTEM_PROMPT_INFERENCE = """당신은 사용자 구매 데이터를 기반으로 관심사를 분석하는 AI입니다.
물건 목록을 보고 사용자의 라이프스타일과 관심사를 파악하세요."""

SYSTEM_PROMPT_MATCHING = """당신은 사용자 구매 데이터를 기반으로 적합한 모임을 추천하는 AI입니다.
사용자의 물건 목록과 구매 패턴을 분석하여 가장 적합한 모임 카테고리를 선택하세요.

카테고리: 가족/육아, 반려동물, 운동, 자기계발, 취미/오락"""

SYSTEM_PROMPT_THINKING_MODE = """You are an expert Data Analyst AI specializing in consumer behavior analysis.
Analyze the user's purchase history and determine their interests and lifestyle."""

# ==========================================
# Data Generation Prompts
# ==========================================
SYSTEM_PROMPT_DATA_GENERATION = """당신은 '당근마켓 구매 패턴 생성기'입니다.
현재 AI 모델이 '취미', '자기계발', '육아', '반려동물'을 서로 헷갈려하고 있습니다.
이 경계를 명확히 구분할 수 있는 **'어려운(Hard)' 케이스**를 생성해야 합니다.

다음 3가지 유형의 데이터를 균형 있게 생성하세요 (JSON 형식):

1. **[유형 A: 자기계발 vs 취미]**
   - 겉보기엔 취미 같지만, 실제로는 '자격증', '취업', '부업', '대회' 목적이 뚜렷한 물건 목록.

2. **[유형 B: 육아 vs 취미/살림]**
   - 요리나 만들기를 하지만, 목적이 철저히 '아이를 위한' 것인 목록.

3. **[유형 C: 특수 반려동물]**
   - 개/고양이가 아닌 곤충, 파충류, 소동물, 물고기 관련 용품."""

# ==========================================
# Output Format Templates
# ==========================================
OUTPUT_FORMAT_CATEGORY_ONLY = '{"category": "카테고리명"}'

OUTPUT_FORMAT_FULL = '''{
    "category": "카테고리명",
    "tags": {
        "숙련도": "#초보환영 또는 #고인물",
        "지속성": "#정기모임 또는 #번개",
        "분위기": "#가벼움 또는 #진지함",
        "연령대": "#또래중심 또는 #전연령"
    },
    "reasoning": "추천 이유"
}'''
