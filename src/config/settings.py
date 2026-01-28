"""
Central configuration settings for the project.
All model names, paths, and hyperparameters are defined here.
"""
from pathlib import Path

# ==========================================
# Project Paths
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs_1.7b"
SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"

# ==========================================
# Model Configuration
# ==========================================
class Models:
    # Production Models
    QWEN3_1_7B = "unsloth/Qwen3-1.7B-unsloth-bnb-4bit"
    QWEN3_4B = "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit"
    QWEN3_14B = "unsloth/Qwen3-14B-unsloth-bnb-4bit"
    QWEN2_5_14B = "Qwen/Qwen2.5-14B-Instruct"

    # Embedding Model
    EMBEDDING_MODEL = "BAAI/bge-m3"
    EMBEDDING_DIM = 1024

# ==========================================
# Training Configuration
# ==========================================
class TrainingConfig:
    MAX_SEQ_LENGTH = 2048
    LOAD_IN_4BIT = True

    # LoRA Settings
    LORA_RANK = 64
    LORA_ALPHA = 64
    LORA_DROPOUT = 0

    # Training Hyperparameters
    LEARNING_RATE = 2e-4
    BATCH_SIZE = 4
    GRADIENT_ACCUMULATION_STEPS = 4
    NUM_EPOCHS = 3
    WARMUP_STEPS = 5

    # Generation Settings
    TEMPERATURE = 0.7
    TOP_P = 0.8
    TOP_K = 20
    MAX_NEW_TOKENS = 512

# ==========================================
# Data Files
# ==========================================
class DataFiles:
    # Training Data
    TRAIN_FINAL = DATA_DIR / "train_final.jsonl"
    TRAIN_ADD_300 = DATA_DIR / "train_add_300.jsonl"

    # Test Data
    TEST_DATA = RAW_DATA_DIR / "train_final1.jsonl"

    # Raw Data
    RAW_DATA_1000 = RAW_DATA_DIR / "raw_data_1000.jsonl"
    SYNTHETIC_500 = RAW_DATA_DIR / "synthetic_train_data_500.jsonl"
    DATA_300 = RAW_DATA_DIR / "data_300.jsonl"

    # Output Files
    BLIND_TEST_RESULT = DATA_DIR / "blind_test_result_all.md"
    BLIND_TEST_BASE_RESULT = DATA_DIR / "blind_test_result_base.md"

# ==========================================
# Category Configuration
# ==========================================
CATEGORIES = [
    "가족/육아",
    "반려동물",
    "운동",
    "자기계발",
    "취미/오락",
]

TAG_DIMENSIONS = {
    "숙련도": ["#초보환영", "#고인물"],
    "지속성": ["#정기모임", "#번개"],
    "분위기": ["#가벼움", "#진지함"],
    "연령대": ["#또래중심", "#전연령"],
}
