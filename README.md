# 당근마켓 사용자-모임 매칭 AI 프로젝트

LLM과 임베딩을 활용하여 사용자의 관심사를 추론하고, 적합한 모임을 추천하는 AI 시스템입니다.

## 프로젝트 개요

이 프로젝트는 3단계로 구성됩니다:

1. **1차 과제**: 사용자의 상품 목록을 기반으로 관심사를 추론하고 임베딩 벡터 생성
2. **2차 과제**: 모임 분류를 위한 LLM 파인튜닝 (Unsloth + LoRA)
3. **3차 과제**: 3단계 필터링 기반 사용자-모임 매칭 및 시각화

## 모델 성능

| 모델 | 정확도 | 비고 |
|------|--------|------|
| 원본 Qwen3-1.7B | 56.5% | Instruction 명시 필요 |
| 파인튜닝 Qwen3-1.7B | 73.9% | +17.4%p 향상 |

### 카테고리별 정확도 (파인튜닝 모델)
- 운동: 87.5%
- 반려동물: 85.7%
- 취미/오락: 84.6%
- 자기계발: 69.2%
- 가족/육아: 20.0%

## 폴더 구조

```
hugging-face-project/
│
├── src/                        # 소스 코드
│   ├── config/                 # 설정 모듈
│   │   ├── settings.py         # 모델, 경로, 하이퍼파라미터
│   │   └── prompts.py          # 프롬프트 템플릿
│   │
│   ├── core/                   # 핵심 엔진
│   │   ├── inference_engine.py # LLM 추론 래퍼
│   │   └── matching_engine.py  # 3단계 매칭 알고리즘
│   │
│   ├── data/                   # 데이터 처리
│   │   ├── loader.py           # JSONL/JSON 로드/저장
│   │   └── cleaner.py          # 데이터 정제/검증
│   │
│   ├── training/               # 학습 모듈
│   │   ├── trainer.py          # LoRA 파인튜닝
│   │   └── evaluator.py        # 블라인드 테스트
│   │
│   ├── visualization/          # 시각화
│   │   ├── report.py           # 마크다운 리포트
│   │   └── ui_generator.py     # HTML UI 생성
│   │
│   ├── scripts/                # 실행 스크립트
│   │   ├── run_training.py     # 학습 실행
│   │   └── run_evaluation.py   # 평가 실행
│   │
│   ├── task1_inference/        # [Legacy] 관심사 추론
│   ├── task2_finetuning/       # [Legacy] 파인튜닝
│   ├── task3_matching/         # [Legacy] 매칭
│   ├── train/                  # [Legacy] 1.7B 학습
│   └── utils/                  # 유틸리티
│
├── data/                       # 데이터 저장소
│   ├── raw/                    # 원본 데이터
│   └── processed/              # 가공된 데이터
│
├── outputs_1.7b/               # 파인튜닝된 Qwen3-1.7B 모델
│
└── saved_models/               # 최종 모델 저장
```

## 기술 스택

- **LLM**: Qwen3-1.7B, Qwen3-14B (4bit 양자화)
- **최적화**: Unsloth (추론 속도 2배 향상)
- **임베딩**: BAAI/bge-m3 (1024차원)
- **파인튜닝**: LoRA (Rank=64, Alpha=64)
- **프레임워크**: PyTorch, Transformers, TRL, PEFT

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

### 새로운 모듈 방식 (권장)

```bash
# 모델 평가
python -m src.scripts.run_evaluation --model outputs_1.7b

# 모델 학습
python -m src.scripts.run_training --data data/train_final.jsonl --epochs 3
```

### 직접 모듈 사용

```python
from src.config import Models, CATEGORIES
from src.core import InferenceEngine, MatchingEngine
from src.data import DataLoader, DataCleaner
from src.training import LoRATrainer, ModelEvaluator
from src.visualization import ReportGenerator, UIGenerator

# 추론 엔진 사용
engine = InferenceEngine(model_path="outputs_1.7b")
profile = engine.analyze_user_profile(user_input)

# 매칭 실행
db = generate_dummy_db(100)
matcher = MatchingEngine(db)
results = matcher.match(profile, top_n=3)
```

### Legacy 스크립트

```bash
# 블라인드 테스트
python src/train/blind_test_1.7b.py

# 원본 모델 테스트
python src/train/blind_test_base_1.7b.py
```

## 매칭 알고리즘

3단계 필터링 방식:

1. **Hard Filter**: 카테고리 일치 여부
2. **Soft Score**: 태그 매칭 점수 (4개 × 25점 = 100점)
   - 숙련도: #초보환영 / #고인물
   - 지속성: #정기모임 / #번개
   - 분위기: #가벼움 / #진지함
   - 연령대: #또래중심 / #전연령
3. **Top-N**: 점수순 정렬 후 상위 추출

## 시스템 요구사항

- Python 3.10+
- CUDA 12.1+ (GPU 사용 시)
- RAM: 16GB 이상
- VRAM: 8GB 이상 (1.7B 4bit 기준)

## 라이선스

MIT License
