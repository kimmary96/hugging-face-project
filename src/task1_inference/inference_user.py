import json
import torch
from transpipformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

# --- 설정 ---
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct" # 14B가 무거우면 7B 사용
EMBED_MODEL_ID = "BAAI/bge-m3"
DATA_PATH = "./data/dummy_users.json"

# 1. LLM으로 관심사 추론 (Few-shot CoT)
def generate_interests(items):
    # 당근 렌즈의 핵심: 물건들의 '집합'에서 '페르소나'를 읽어내는 프롬프트
    prompt = f"""
    당신은 중고거래 데이터를 분석하는 AI 전문가입니다.
    사용자가 구매하거나 판매한 물건 목록을 보고, 이 사용자의 '관심사 키워드' 3가지를 추론하세요.
    물건들이 함께 사용되는 맥락(Context)을 고려하세요.

    예시:
    물건: 요가 매트, 폼롤러, 단백질 쉐이크
    생각: 이 물건들은 주로 집에서 운동할 때 사용된다. 건강 관리와 관련이 깊다.
    관심사: 홈트레이닝, 다이어트, 건강

    물건: {items}
    생각:
    관심사:
    """
    # 실제 구현 시에는 토크나이저/모델 로드 후 generate 함수 호출 (생략)
    # 여기서는 프롬프트 구조만 확인
    return prompt

# 2. 임베딩 벡터 생성
def create_embedding(text_list):
    print(">>> 임베딩 모델 로드 중...")
    model = SentenceTransformer(EMBED_MODEL_ID)
    embeddings = model.encode(text_list, normalize_embeddings=True)
    return embeddings

# --- 실행 로직 (메인) ---
# 실제 개발 시:
# 1. JSON 로드
# 2. LLM 로드 -> 관심사 텍스트 생성 -> JSON 업데이트 -> LLM 메모리 해제(del model, torch.cuda.empty_cache())
# 3. 임베딩 모델 로드 -> 생성된 관심사를 벡터화 -> 저장