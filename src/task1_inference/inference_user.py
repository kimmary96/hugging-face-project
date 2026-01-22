import json
import torch
from unsloth import FastLanguageModel
from sentence_transformers import SentenceTransformer
from tqdm import tqdm # 진행률 바 표시

# --- 설정 (Configuration) ---
# [변경] 16GB VRAM 성능 극대화를 위해 14B 모델 사용
LLM_MODEL_ID = "unsloth/Qwen2.5-14B-Instruct-bnb-4bit"
EMBED_MODEL_ID = "BAAI/bge-m3"

# 파일 경로
INPUT_FILE = "./data/raw/dummy_users.json"
OUTPUT_FILE = "./data/processed/user_profiles_vectorized.json"

def main():
    # 1. LLM 로드 (Unsloth 최적화)
    print(f">>> [1/3] LLM 로드 중... ({LLM_MODEL_ID})")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = LLM_MODEL_ID,
        max_seq_length = 2048, # 넉넉하게 설정
        dtype = None,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model) # 추론 속도 2배 향상

    # 2. 임베딩 모델 로드
    print(f">>> [2/3] 임베딩 모델 로드 중... ({EMBED_MODEL_ID})")
    # LLM이 9.3GB, 임베딩이 1.5GB 정도 쓰므로 16GB VRAM에서 동시 실행 가능합니다.
    try:
        embed_model = SentenceTransformer(
            EMBED_MODEL_ID,
            device="cuda",
            model_kwargs={"use_safetensors": True},
        )
    except OSError as exc:
        # Fallback for models without safetensors artifacts.
        if "safetensors" not in str(exc):
            raise
        embed_model = SentenceTransformer(EMBED_MODEL_ID, device="cuda")

    # 3. 데이터 로드
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            users = json.load(f)
    except FileNotFoundError:
        print(f"❌ 오류: {INPUT_FILE} 파일이 없습니다. 더미 데이터를 먼저 생성해주세요.")
        return

    print(f">>> [3/3] 총 {len(users)}명의 유저 데이터 분석 시작...")
    processed_data = []
    
    # Qwen 시스템 프롬프트 (페르소나 부여)
    system_prompt = """당신은 e-커머스 데이터 분석 전문가입니다. 
    유저의 판매 예정 물품 목록을 보고, 그들의 **핵심 관심사 키워드 3~5개**를 추론하세요.
    
    [규칙]
    1. 설명하지 말고 키워드만 쉼표(,)로 구분해서 출력할 것.
    2. 물건의 표면적 이름보다 '맥락(Context)'을 읽을 것. (예: 아기띠 -> 육아)
    """

    for user in tqdm(users):
        items = user['items']
        
        # (A) LLM 추론: Chat Template 적용
        user_msg = f"판매 예정 물품: {items}\n\n이 유저의 관심사는?"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg}
        ]
        
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to("cuda")

        # 생성
        outputs = model.generate(
            input_ids=inputs, 
            max_new_tokens=128,
            use_cache=True,
            temperature=0.3, # 창의성 억제, 정확도 위주
        )
        
        # 결과 텍스트 추출
        decoded_output = tokenizer.batch_decode(outputs[:, inputs.shape[1]:], skip_special_tokens=True)[0]
        interests_text = decoded_output.strip()
        
        # (B) 임베딩 변환
        # 추론된 텍스트("육아, 건강, 이유식")를 벡터로 변환
        vector = embed_model.encode(interests_text).tolist()

        # 결과 저장
        user_result = {
            "user_id": user["user_id"],
            "items": items,
            "inferred_interests": interests_text, # LLM의 결과
            "embedding": vector # 3차 과제 매칭용 벡터
        }
        processed_data.append(user_result)

    # 4. 파일 저장
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)

    print("\n" + "="*50)
    print(f">>> ✅ 처리 완료! 결과 저장됨: {OUTPUT_FILE}")
    print(f">>> [샘플 확인] ID: {processed_data[0]['user_id']}")
    print(f">>> 구매 목록: {processed_data[0]['items']}")
    print(f">>> AI 추론값: {processed_data[0]['inferred_interests']}")
    print("="*50)

if __name__ == "__main__":
    main()
