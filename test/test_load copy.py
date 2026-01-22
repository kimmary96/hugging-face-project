# -*- coding: utf-8 -*-
from unsloth import FastLanguageModel
import torch

# 1. 모델 설정
# 4B 모델로 우선 안정성/속도 확인 (16GB VRAM 기준 추천)
model_name = "unsloth/Qwen3-4B-Thinking-2507-unsloth-bnb-4bit"
max_seq_length = 2048
dtype = None  # None이면 자동 설정 (float16 등)
load_in_4bit = True  # 4bit 로딩 활성화
load_in_8bit = False  # 8bit는 2x 메모리 사용
full_finetuning = False

# 14B로 올릴 경우 권장 예시 (VRAM 여유가 적을 수 있음)
# model_name = "unsloth/Qwen3-14B-unsloth-bnb-4bit"
# max_seq_length = 1024

# 2. 모델 로드 테스트 (여기서 에러가 나면 실패)
print(">>> 모델 로딩 시작...")
try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        full_finetuning=full_finetuning,
    )
    print(">>> ✅ 성공! 모델이 정상적으로 로드되었습니다.")
    print(f">>> 사용 중인 VRAM: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
except Exception as e:
    print(f">>> ❌ 실패! 에러 메시지를 확인하세요.\n{e}")
