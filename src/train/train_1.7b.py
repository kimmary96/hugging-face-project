"""
Qwen3-1.7B 파인튜닝 스크립트 (train_1.7b.py)

당근마켓 콜드스타트 해결을 위한 SLM 파인튜닝
- Model: unsloth/Qwen3-1.7B-unsloth-bnb-4bit
- Strategy: Hybrid (Epoch 3, Rank 64)
- Mode: Non-thinking mode (enable_thinking=False)

Hardware: RTX 4070 Ti Super (16GB VRAM)
"""

import os
import re
from pathlib import Path
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl.trainer.sft_trainer import SFTTrainer
from trl.trainer.sft_config import SFTConfig
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 하이퍼파라미터 설정 (Strict)
# ==========================================
MODEL_NAME = "unsloth/Qwen3-1.7B-unsloth-bnb-4bit"
LORA_RANK = 64                    # 복잡한 논리를 담기 위해 높게 설정
LORA_ALPHA = 64                   # 일반적으로 rank와 동일하게 설정
NUM_TRAIN_EPOCHS = 3              # 데이터 1000개, 과적합 방지
LEARNING_RATE = 2e-4
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4   # Effective Batch Size = 16
MAX_SEQ_LENGTH = 2048

# Target Modules (All-linear)
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]

# ==========================================
# Non-Thinking Mode 추론 설정 (Best Practices)
# ==========================================
INFERENCE_CONFIG = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0,
}

# ==========================================
# 경로 설정
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_FILE_BASE = PROJECT_ROOT / "data/train_final.jsonl"
DATA_FILE_ADD = PROJECT_ROOT / "data/train_add_300.jsonl"
USE_COMBINED_DATA = True
DATA_FILES = [DATA_FILE_BASE, DATA_FILE_ADD] if USE_COMBINED_DATA else [DATA_FILE_BASE]
OUTPUT_DIR = PROJECT_ROOT / "outputs_1.7b"
RESUME_FROM_CHECKPOINT = True
EXP_NAME = "Qwen3-1.7B_Rank64_Epoch3"

def find_latest_checkpoint(output_dir: Path):
    if not output_dir.exists():
        return None
    checkpoints = []
    for child in output_dir.iterdir():
        if child.is_dir() and child.name.startswith("checkpoint-"):
            m = re.match(r"checkpoint-(\d+)$", child.name)
            if m:
                checkpoints.append((int(m.group(1)), child))
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints[-1][1]


def main():
    print("=" * 60)
    print(f"Qwen3-1.7B 파인튜닝 시작")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"LoRA Rank: {LORA_RANK}")
    print(f"Epochs: {NUM_TRAIN_EPOCHS}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Effective Batch Size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"Max Seq Length: {MAX_SEQ_LENGTH}")
    print("=" * 60)

    # 데이터 파일 존재 확인
    missing = [p for p in DATA_FILES if not p.exists()]
    if missing:
        for p in missing:
            print(f"[ERROR] Data file not found: {p}")
        print("Run: python src/utils/clean_jsonl.py")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ==========================================
    # 1. 모델 로드
    # ==========================================
    print("\n[1/5] 모델 로드 중...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # 자동 감지
        load_in_4bit=True,
    )

    # ==========================================
    # 2. LoRA 설정
    # ==========================================
    print("[2/5] LoRA 어댑터 설정 중...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,  # Unsloth 권장: 0
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # ==========================================
    # 3. 데이터셋 로드 및 분할
    # ==========================================
    print("[3/5] 데이터셋 로딩 중...")
    dataset = load_dataset("json", data_files=[str(p) for p in DATA_FILES], split="train")

    # Train/Eval 분할 (80/20)
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    print(f"  Train: {len(train_dataset)}")
    print(f"  Eval: {len(eval_dataset)}")

    # Qwen3 Chat Template 포맷팅 (Non-thinking mode)
    def formatting_prompts_func(examples):
        texts = []
        for instruction, inp, output in zip(
            examples["instruction"], examples["input"], examples["output"]
        ):
            # Qwen3 Chat Template 구성
            messages = [
                {"role": "system", "content": instruction},
                {"role": "user", "content": inp}
            ]

            # Non-thinking mode로 프롬프트 생성
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False  # Non-thinking mode
            )

            # 프롬프트 + 응답 + EOS 토큰
            full_text = prompt + output + tokenizer.eos_token
            texts.append(full_text)
        return {"text": texts}

    train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
    eval_dataset = eval_dataset.map(formatting_prompts_func, batched=True)

    # ==========================================
    # 4. 트레이너 설정 및 학습
    # ==========================================
    print("[4/5] 학습 시작...")
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(
            dataset_text_field="text",
            max_length=MAX_SEQ_LENGTH,
            dataset_num_proc=1,
            packing=False,

            # 배치 설정
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,

            # 학습 설정
            num_train_epochs=NUM_TRAIN_EPOCHS,
            learning_rate=LEARNING_RATE,
            warmup_ratio=0.1,
            weight_decay=0.01,
            lr_scheduler_type="cosine",

            # 정밀도 설정
            fp16=False,
            bf16=True,
            optim="adamw_8bit",

            # 로깅 및 저장
            logging_steps=5,
            eval_strategy="steps",
            eval_steps=10,
            save_strategy="steps",
            save_steps=50,
            save_total_limit=3,
            load_best_model_at_end=True,

            # 기타
            seed=3407,
            output_dir=str(OUTPUT_DIR),
            report_to="none",
        ),
    )

    # 학습 실행
    trainer.train()

    # ==========================================
    # 5. 모델 저장 및 시각화
    # ==========================================
    print("[5/5] 모델 저장 및 시각화...")

    # 모델 저장
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"  모델 저장됨: {OUTPUT_DIR}")

    # 추론 설정 저장 (Non-thinking mode best practices)
    import json
    config_path = OUTPUT_DIR / "inference_config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump({
            "mode": "non-thinking",
            "enable_thinking": False,
            **INFERENCE_CONFIG
        }, f, indent=2)
    print(f"  추론 설정 저장됨: {config_path}")

    # 학습 곡선 시각화
    history = trainer.state.log_history
    train_loss = []
    train_epochs = []
    eval_loss = []
    eval_epochs = []

    for entry in history:
        if 'loss' in entry:
            train_loss.append(entry['loss'])
            train_epochs.append(entry['epoch'])
        elif 'eval_loss' in entry:
            eval_loss.append(entry['eval_loss'])
            eval_epochs.append(entry['epoch'])

    # 최적점 계산
    if len(eval_loss) > 0:
        min_loss_idx = np.argmin(eval_loss)
        min_val_loss = eval_loss[min_loss_idx]
        best_epoch = eval_epochs[min_loss_idx]
    else:
        best_epoch = 0
        min_val_loss = 0

    # 그래프 생성
    plt.figure(figsize=(12, 7))

    plt.plot(train_epochs, train_loss,
             label='Training Loss',
             marker='o', color='blue', linestyle='-', linewidth=1.5, markersize=4)

    if eval_loss:
        plt.plot(eval_epochs, eval_loss,
                 label='Validation Loss',
                 marker='x', color='red', linestyle='--', linewidth=2, markersize=5)

        plt.axvline(x=best_epoch, color='green', linestyle=':', linewidth=2)
        text_y_pos = (max(train_loss) + min(train_loss)) / 2
        plt.text(best_epoch + 0.05, text_y_pos,
                 f'Best\n(Epoch {best_epoch:.2f}\nLoss {min_val_loss:.4f})',
                 color='green', fontweight='bold', fontsize=9,
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    plt.title(f'Learning Curve: {EXP_NAME}', fontsize=14, pad=15)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()

    graph_path = OUTPUT_DIR / f"loss_{EXP_NAME}.png"
    plt.savefig(graph_path, dpi=300, bbox_inches='tight')
    print(f"  그래프 저장됨: {graph_path}")
    plt.close()

    print("\n" + "=" * 60)
    print("학습 완료!")
    print("=" * 60)
    print(f"최종 모델: {OUTPUT_DIR}")
    print(f"Best Validation Loss: {min_val_loss:.4f} (Epoch {best_epoch:.2f})")


if __name__ == '__main__':
    main()
