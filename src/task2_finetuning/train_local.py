from unsloth import FastLanguageModel
from datasets import load_dataset
from trl.trainer.sft_trainer import SFTTrainer
from trl.trainer.sft_config import SFTConfig

# ==========================================
# 설정 (RTX 4070 Ti S 맞춤형)
# ==========================================
MODEL_NAME = "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit"
OUTPUT_DIR = "outputs_checkpoint"
DATA_FILE = "src/task2_finetuning/cleaned_train_data.jsonl"
USE_BFLOAT16 = True
MAX_SEQ_LENGTH = 2048

# Alpaca 스타일 프롬프트 포맷
ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""


def main():
    # ==========================================
    # 1. 모델 및 토크나이저 로드
    # ==========================================
    print(f"🔄 모델 로드 중... ({MODEL_NAME})")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    # LoRA 어댑터 설정
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # ==========================================
    # 2. 데이터셋 로드 및 포맷팅
    # ==========================================
    print("📂 데이터셋 로딩 중...")
    dataset = load_dataset("json", data_files=DATA_FILE, split="train")

    eos_token = tokenizer.eos_token

    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        for instruction, inp, output in zip(instructions, inputs, outputs):
            text = ALPACA_PROMPT.format(instruction, inp, output) + eos_token
            texts.append(text)
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)

    # ==========================================
    # 3. 트레이너 설정
    # ==========================================
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            dataset_text_field="text",
            max_length=MAX_SEQ_LENGTH,
            dataset_num_proc=1,
            packing=False,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            warmup_steps=10,
            max_steps=100,
            learning_rate=2e-4,
            fp16=not USE_BFLOAT16,
            bf16=USE_BFLOAT16,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=OUTPUT_DIR,
            report_to="none",
        ),
    )

    # ==========================================
    # 4. 학습 실행
    # ==========================================
    print("🚀 학습 시작!")
    trainer.train()

    print("🎉 학습 완료! 모델 저장 중...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"💾 저장 경로: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
