# 당근마켓 사용자-모임 매칭 AI 프로젝트

LLM과 임베딩을 활용하여 사용자의 관심사를 추론하고, 적합한 모임을 추천하는 AI 시스템입니다.

## 프로젝트 개요

이 프로젝트는 3단계로 구성됩니다:

1. **1차 과제**: 사용자의 상품 목록을 기반으로 관심사를 추론하고 임베딩 벡터 생성
2. **2차 과제**: 모임 분류를 위한 LLM 파인튜닝 (Unsloth + LoRA)
3. **3차 과제**: 코사인 유사도 기반 사용자-모임 매칭 및 시각화

## 폴더 구조

```
hugging-face-project/
│
├── .gitignore              # Git 제외 파일 목록
├── requirements.txt        # Python 의존성 목록
├── README.md               # 프로젝트 설명서 (현재 파일)
│
├── data/                   # 데이터 저장소
│   ├── raw/                # 원본 데이터
│   │   ├── dummy_users.json
│   │   └── carrot_meetings_100.csv (예정)
│   └── processed/          # 가공된 데이터
│       └── train_dataset.jsonl
│
├── src/                    # 소스 코드
│   ├── task1_inference/    # [1차 과제] 관심사 추론 & 임베딩
│   │   ├── generate_dummy.py   # 더미 데이터 생성
│   │   └── inference_user.py   # LLM 추론 및 임베딩 생성
│   │
│   ├── task2_finetuning/   # [2차 과제] 모임 분류 파인튜닝
│   │   ├── preprocess.py       # 데이터 전처리 (CSV → JSONL)
│   │   └── train_lora.py       # Unsloth LoRA 파인튜닝
│   │
│   └── task3_matching/     # [3차 과제] 매칭 및 시각화
│       ├── match_logic.py      # 코사인 유사도 매칭
│       └── visualize.py        # 결과 시각화 (히트맵 등)
│
├── saved_models/           # [Git 제외] 학습된 모델 저장
│   └── unsloth_lora_model/
│
└── outputs/                # 실험 결과물
    ├── user_vectors.pkl
    └── loss_curve.png
```

## 기술 스택

- **LLM**: Qwen2.5-14B-Instruct (4bit 양자화)
- **최적화**: Unsloth (추론 속도 2배 향상)
- **임베딩**: BAAI/bge-m3
- **파인튜닝**: LoRA (Low-Rank Adaptation)
- **프레임워크**: PyTorch, Transformers, sentence-transformers

## 설치 방법

```bash
# 저장소 클론
git clone https://github.com/your-username/hugging-face-project.git
cd hugging-face-project

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

## 실행 방법

### 1차 과제: 관심사 추론 및 임베딩

```bash
# 더미 데이터 생성 (선택)
python src/task1_inference/generate_dummy.py

# LLM 추론 및 임베딩 생성
python src/task1_inference/inference_user.py
```

### 2차 과제: 파인튜닝

```bash
# 데이터 전처리
python src/task2_finetuning/preprocess.py

# LoRA 파인튜닝
python src/task2_finetuning/train_lora.py
```

### 3차 과제: 매칭 및 시각화

```bash
# 매칭 실행
python src/task3_matching/match_logic.py

# 결과 시각화
python src/task3_matching/visualize.py
```

## 시스템 요구사항

- Python 3.10+
- CUDA 12.1+ (GPU 사용 시)
- RAM: 16GB 이상
- VRAM: 12GB 이상 (4bit 양자화 기준)

## 라이선스

MIT License
