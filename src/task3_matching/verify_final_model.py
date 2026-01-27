import torch
from unsloth import FastLanguageModel
import os

# ==========================================
# 1. 환경 및 경로 설정
# ==========================================
max_seq_length = 2048 # 매칭 단계에서는 긴 컨텍스트가 필요 없으므로 효율성 위해 조절
dtype = None 
load_in_4bit = True 

# 최종 학습된 모델 경로 (Day 2 밤샘 작업 결과물)
# 만약 경로 에러 발생 시 절대 경로로 수정 필요 (예: "C:/hugging-face-project/outputs_final_1000")
model_path = "C:\\Users\\User\\Documents\\dev\\hugging-face-project\\outputs_experiments\\Exp2_Overfitting\\checkpoint-49"

print(f"🔄 [System] 모델 로드를 시작합니다... 경로: {model_path}")

# ==========================================
# 2. 모델 및 토크나이저 로드
# ==========================================
try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path, # 여기서 학습된 어댑터를 자동으로 로드합니다.
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        # device_map="auto" # Unsloth는 기본적으로 GPU 0을 사용하므로 생략 가능
    )
    FastLanguageModel.for_inference(model) # 추론 속도 2배 향상 (Native 윈도우 필수)
    print("✅ [Success] 모델 로드 완료! (RTX 4070 Ti Super / 16GB VRAM Optimized)")

except Exception as e:
    print(f"❌ [Error] 모델 로드 실패: {e}")
    print("Tip: \"outputs_experiment/checkpoint-125\" 폴더 안에 'adapter_model.bin' 혹은 'safetensors'가 있는지 확인하세요.")
    exit()

# ==========================================
# 3. 추론 테스트 (Inference Test)
# ==========================================
# 시나리오: Day 2 Baseline보다 복잡한, 숨겨진 의도를 파악해야 하는 유저
test_input = "최근에 '아기띠'랑 '방수 식탁보'를 샀고, 어제는 '손목 보호대'를 검색했어. 나한테 맞는 모임 없을까?"

# 프롬프트 포맷 (Alpaca 스타일 - 학습 때와 동일하게 유지)
prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
사용자의 구매 이력을 기반으로 숨겨진 관심사와 페르소나를 추론하세요.

### Input:
{test_input}

### Response:
"""

print("\n🤖 [AI Thinking] 추론을 시작합니다...\n")

inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")

# 생성 옵션 설정
outputs = model.generate(
    **inputs, 
    max_new_tokens = 256, 
    use_cache = True,
    temperature = 0.7, # 창의성과 정확성의 균형
)

# 결과 디코딩
decoded_output = tokenizer.batch_decode(outputs, skip_special_tokens = True)
result = decoded_output[0].split("### Response:")[-1].strip()

print("="*50)
print(f"📝 [User Input]: {test_input}")
print("-" * 50)
print(f"💡 [Model Output (Persona Analysis)]:\n{result}")
print("="*50)