from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from trl.trainer.sft_config import SFTConfig

# ==========================================
# 1. 설정 (RTX 4070 Ti S 맞춤형)
# ==========================================
# 학습 타겟: 16GB VRAM에서 학습 가능한 가장 똑똑한 모델 (Qwen 3 4B)
# ※ 14B는 학습 시 OOM 발생하므로 4B를 Student로 사용합니다.
MODEL_NAME = "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit" 
OUTPUT_DIR = "outputs_checkpoint"
DATA_FILE = "cleaned_train_data.jsonl" # 방금 정제한 파일

# 4070 Ti는 bfloat16을 지원하므로 True로 설정 (속도/성능 향상)
USE_BFLOAT16 = True 
MAX_SEQ_LENGTH = 2048 # 7B 모델은 16GB에서 2048 길이도 넉넉함

# ==========================================
# 2. 모델 및 토크나이저 로드
# ==========================================
print(f"🔄 모델 로드 중... ({MODEL_NAME})")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = MAX_SEQ_LENGTH,
    dtype = None, # Auto detection (bf16)
    load_in_4bit = True,
)

# LoRA 어댑터 설정 (모델의 전체를 튜닝하는 대신 일부만 효율적으로 학습)
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Rank: 높을수록 똑똑해지지만 메모리 더 씀 (16 추천)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, 
    bias = "none",   
    use_gradient_checkpointing = "unsloth", # 메모리 절약 핵심 기술
    random_state = 3407,
)

# ==========================================
# 3. 데이터셋 로드 및 포맷팅
# ==========================================
print("📂 데이터셋 로딩 중...")
dataset = load_dataset("json", data_files=DATA_FILE, split="train")

# Alpaca 스타일 프롬프트 포맷
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token 

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # EOS 토큰을 붙여야 모델이 생성을 멈추는 법을 배움
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

dataset = dataset.map(formatting_prompts_func, batched = True)

# ==========================================
# 4. 트레이너 설정 (Hyperparameters)
# ==========================================
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    args = SFTConfig(
        dataset_text_field = "text",
        max_length = MAX_SEQ_LENGTH,
        dataset_num_proc = 2,
        packing = False,
        per_device_train_batch_size = 4, # 16GB VRAM 기준 4~8 가능
        gradient_accumulation_steps = 4, # 4 * 4 = 실제 배치 사이즈 16 효과
        warmup_steps = 10,
        max_steps = 100, # 데이터 132개이므로 100 스텝이면 충분히 학습됨
        learning_rate = 2e-4, # 학습률 (LoRA 표준)
        fp16 = not USE_BFLOAT16,
        bf16 = USE_BFLOAT16,
        logging_steps = 1, # 매 스텝마다 로그 출력
        optim = "adamw_8bit", # 메모리 절약형 옵티마이저
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = OUTPUT_DIR,
        report_to = "none", # 완드비(WandB) 사용 안함
    ),
)

# ==========================================
# 5. 학습 실행
# ==========================================
print("🚀 학습 시작! (예상 소요 시간: 5~10분)")
trainer_stats = trainer.train()

print("🎉 학습 완료! 모델 저장 중...")
model.save_pretrained(OUTPUT_DIR) # LoRA 어댑터만 저장됨 (용량 작음)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"💾 저장 경로: {OUTPUT_DIR}")