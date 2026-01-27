import os
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl.trainer.sft_trainer import SFTTrainer
from trl.trainer.sft_config import SFTConfig
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 실험 하이퍼파라미터 (이 부분을 바꿔가며 실험하세요)
# ==========================================
EXP_NAME = "Exp6_Dataset500"  # 실험 이름 (저장 폴더명)
LEARNING_RATE = 2e-4           # 학습률 (2e-4, 2e-5, 2e-3 등 실험)
NUM_TRAIN_EPOCHS = 5          # 에폭 (1, 5, 10 등 실험)
LORA_RANK = 64                 # LoRA Rank (8, 16, 64 등 실험)

# ==========================================
# 설정
# ==========================================
MODEL_NAME = "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit"
DATA_FILE = "src/task2_finetuning/cleaned_train_data.jsonl"
OUTPUT_DIR = f"outputs_experiments/{EXP_NAME}"
MAX_SEQ_LENGTH = 2048

ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""


def main():
    # 출력 디렉토리 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ==========================================
    # 1. 모델 로드
    # ==========================================
    print(f"🔄 모델 로드 중... ({MODEL_NAME})")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # ==========================================
    # 2. 데이터셋 로드 및 분할
    # ==========================================
    print("📂 데이터셋 로딩 중...")
    dataset = load_dataset("json", data_files=DATA_FILE, split="train")
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    print(f"📊 데이터 분할: Train({len(train_dataset)}개) / Eval({len(eval_dataset)}개)")

    eos_token = tokenizer.eos_token

    def formatting_prompts_func(examples):
        texts = []
        for instruction, inp, output in zip(
            examples["instruction"], examples["input"], examples["output"]
        ):
            text = ALPACA_PROMPT.format(instruction, inp, output) + eos_token
            texts.append(text)
        return {"text": texts}

    train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
    eval_dataset = eval_dataset.map(formatting_prompts_func, batched=True)

    # ==========================================
    # 3. 트레이너 설정
    # ==========================================
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
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=4,
            num_train_epochs=NUM_TRAIN_EPOCHS,
            learning_rate=LEARNING_RATE,
            fp16=False,
            bf16=True,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=OUTPUT_DIR,
            eval_strategy="steps",
            eval_steps=5,
            logging_steps=5,
            save_strategy="steps",
            save_steps=20,
            save_total_limit=2,
            load_best_model_at_end=True,
            report_to="none",
        ),
    )

    # ==========================================
    # 4. 학습 실행
    # ==========================================
    print(f"🚀 실험 시작: {EXP_NAME} (LR={LEARNING_RATE}, Epoch={NUM_TRAIN_EPOCHS}, Rank={LORA_RANK})")
    trainer.train()

    print("🎉 학습 완료! 모델 저장 중...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"💾 저장 경로: {OUTPUT_DIR}")

    # ==========================================
    # 5. 학습 결과 시각화
    # ==========================================
    print("📊 그래프 그리는 중...")

        # 1. 로그 데이터 추출 (Steps 대신 Epoch 사용)
    history = trainer.state.log_history
    train_loss = []
    train_epochs = []
    eval_loss = []
    eval_epochs = []

    for entry in history:
        if 'loss' in entry: # Training Log
            train_loss.append(entry['loss'])
            train_epochs.append(entry['epoch'])
        elif 'eval_loss' in entry: # Validation Log
            eval_loss.append(entry['eval_loss'])
            eval_epochs.append(entry['epoch'])

    # 2. 최적점(Saturation Point) 자동 계산
    # Validation Loss가 가장 낮은 지점을 찾습니다.
    if len(eval_loss) > 0:
        min_loss_idx = np.argmin(eval_loss)
        min_val_loss = eval_loss[min_loss_idx]
        best_epoch = eval_epochs[min_loss_idx]
    else:
        # 혹시 Eval 데이터가 없을 경우를 대비한 예외처리
        best_epoch = 0
        min_val_loss = 0

    # 3. 그래프 스타일 설정
    plt.figure(figsize=(12, 7)) # 크기를 조금 더 키움

    # Train Loss (파란색 실선 + 원형 마커)
    plt.plot(train_epochs, train_loss, 
            label='Training Loss', 
            marker='o', color='blue', linestyle='-', linewidth=1.5, markersize=5)

    # Eval Loss (빨간색 점선 + X 마커)
    plt.plot(eval_epochs, eval_loss, 
            label='Validation Loss', 
            marker='x', color='red', linestyle='--', linewidth=2, markersize=6)

    # 4. 꾸미기 요소 (Grid, Title)
    plt.title(f'Learning Curve: {EXP_NAME}', fontsize=14, pad=15)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # 5. 최적점(Optimal Point) 시각화
    if len(eval_loss) > 0:
        plt.axvline(x=best_epoch, color='green', linestyle=':', linewidth=2, label='Optimal Point')
        
        # 텍스트 위치 자동 조정 (그래프 중간 높이)
        text_y_pos = (max(train_loss) + min(train_loss)) / 2
        
        plt.text(best_epoch + 0.1, text_y_pos, 
                f'Saturation Point\n(Epoch {best_epoch:.2f}, Loss {min_val_loss:.4f})', 
                color='green', fontweight='bold', fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    plt.tight_layout()

    # 하이퍼파라미터를 포함한 고유 파일명
    graph_filename = f"loss_LR{LEARNING_RATE}_EP{NUM_TRAIN_EPOCHS}_R{LORA_RANK}.png"
    graph_path = os.path.join(OUTPUT_DIR, graph_filename)
    plt.savefig(graph_path, dpi=300, bbox_inches='tight')
    print(f"📈 그래프 저장됨: {graph_path}")
    plt.close()


if __name__ == '__main__':
    main()
