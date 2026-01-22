# -*- coding: utf-8 -*-
from unsloth import FastLanguageModel
import torch

# 1. 모델 설정
model_name = "unsloth/Qwen2.5-14B-Instruct-bnb-4bit"  # 4bit 미리 저장된 모델 사용
max_seq_length = 1024
dtype = None  # None이면 자동 설정 (float16 등)
load_in_4bit = True  # 4bit 로딩 활성화

# 2. 모델 로드 테스트 (여기서 에러가 나면 실패)
print(">>> 모델 로딩 시작...")
try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    print(">>> ✅ 성공! 모델이 정상적으로 로드되었습니다.")
    print(f">>> 사용 중인 VRAM: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
except Exception as e:
    print(f">>> ❌ 실패! 에러 메시지를 확인하세요.\n{e}")
