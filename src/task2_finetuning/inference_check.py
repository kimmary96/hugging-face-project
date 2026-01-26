from unsloth import FastLanguageModel
import torch
import json

# ==========================================
# 1. 설정
# ==========================================
# 방금 학습 끝난 어댑터가 저장된 폴더 경로
ADAPTER_PATH = "outputs_checkpoint" 
MAX_SEQ_LENGTH = 2048

# ==========================================
# 2. 학습된 모델 로드 (Base + Adapter)
# ==========================================
print(f"🔄 학습된 모델 로드 중... ({ADAPTER_PATH})")

# Unsloth는 저장된 폴더 경로를 넣으면 자동으로 Base 모델 + LoRA를 합쳐서 불러옵니다.
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = ADAPTER_PATH, 
    max_seq_length = MAX_SEQ_LENGTH,
    dtype = None,
    load_in_4bit = True,
)

# 추론 모드로 전환 (속도 2배 향상)
FastLanguageModel.for_inference(model)

# ==========================================
# 3. 테스트할 가상의 유저들 (New Personas)
# ==========================================
test_users = [
    # Case 1: 30대 개발자 (학습 데이터에 없던 조합)
    "특징: 30대 판교 개발자, 관심사: 최신 IT 기기/얼리어답터",
    
    # Case 2: 은퇴한 60대 (새로운 연령대)
    "특징: 은퇴한 60대 남성, 관심사: 등산/막걸리",
    
    # Case 3: 육아맘 (복합 관심사)
    "특징: 30대 육아맘, 관심사: 육아용품/중고거래"
]

# ==========================================
# 4. 추론 실행
# ==========================================
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
유저 정보를 바탕으로 적절한 모임을 추천하고 사유를 설명하세요.

### Input:
{}

### Response:
"""

print("\n🚀 추천 결과 테스트 시작!\n" + "="*50)

for user_profile in test_users:
    inputs = tokenizer(
        [alpaca_prompt.format(user_profile)],
        return_tensors = "pt",
    ).to("cuda")

    outputs = model.generate(
        **inputs, 
        max_new_tokens = 256, # JSON만 나오면 되니까 길지 않아도 됨
        use_cache = True,
        temperature = 0.1, # 정답에 가까운 추천을 위해 창의성 낮춤
    )
    
    # 결과 디코딩
    result = tokenizer.batch_decode(outputs)
    final_output = result[0].split("### Response:")[-1].replace(tokenizer.eos_token, "").strip()
    
    print(f"\n👤 [User]: {user_profile}")
    print(f"🤖 [AI 추천]: {final_output}")
    print("-" * 50)

print("\n✅ 테스트 완료.")