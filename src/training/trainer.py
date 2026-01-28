"""
LoRA fine-tuning trainer for Qwen models.
"""
from pathlib import Path
from typing import Optional, Dict, Any

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import Models, TrainingConfig, OUTPUTS_DIR
from config.prompts import ALPACA_PROMPT


class LoRATrainer:
    """
    LoRA fine-tuning trainer for Qwen models using Unsloth.

    Supports Qwen3-1.7B, Qwen3-4B, and Qwen3-14B models.
    """

    def __init__(
        self,
        model_name: str = Models.QWEN3_1_7B,
        output_dir: Path = OUTPUTS_DIR,
        lora_rank: int = TrainingConfig.LORA_RANK,
        lora_alpha: int = TrainingConfig.LORA_ALPHA,
        max_seq_length: int = TrainingConfig.MAX_SEQ_LENGTH,
    ):
        """
        Initialize trainer with configuration.

        Args:
            model_name: Base model to fine-tune
            output_dir: Directory to save checkpoints
            lora_rank: LoRA rank (higher = more parameters)
            lora_alpha: LoRA alpha scaling
            max_seq_length: Maximum sequence length
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.max_seq_length = max_seq_length
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load base model with LoRA adapters."""
        from unsloth import FastLanguageModel

        print(f"Loading model: {self.model_name}")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            dtype=None,
            load_in_4bit=TrainingConfig.LOAD_IN_4BIT,
        )

        # Add LoRA adapters
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.lora_rank,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            lora_alpha=self.lora_alpha,
            lora_dropout=TrainingConfig.LORA_DROPOUT,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
        print("Model loaded with LoRA adapters!")

    def prepare_dataset(self, data_path: Path):
        """
        Prepare dataset for training.

        Args:
            data_path: Path to JSONL training data
        """
        from datasets import load_dataset

        dataset = load_dataset('json', data_files=str(data_path), split='train')

        def format_prompt(example):
            return {
                "text": ALPACA_PROMPT.format(
                    instruction=example['instruction'],
                    input=example['input'],
                    output=example['output']
                ) + self.tokenizer.eos_token
            }

        self.dataset = dataset.map(format_prompt)
        print(f"Dataset prepared: {len(self.dataset)} samples")

    def train(
        self,
        num_epochs: int = TrainingConfig.NUM_EPOCHS,
        learning_rate: float = TrainingConfig.LEARNING_RATE,
        batch_size: int = TrainingConfig.BATCH_SIZE,
        gradient_accumulation_steps: int = TrainingConfig.GRADIENT_ACCUMULATION_STEPS,
    ):
        """
        Run training loop.

        Args:
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Per-device batch size
            gradient_accumulation_steps: Gradient accumulation steps
        """
        from trl import SFTTrainer
        from transformers import TrainingArguments

        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        if not hasattr(self, 'dataset'):
            raise ValueError("Dataset not prepared. Call prepare_dataset() first.")

        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=TrainingConfig.WARMUP_STEPS,
            logging_steps=10,
            save_steps=50,
            save_total_limit=3,
            fp16=True,
            optim="adamw_8bit",
        )

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.dataset,
            dataset_text_field="text",
            max_seq_length=self.max_seq_length,
            args=training_args,
        )

        print("Starting training...")
        trainer.train()
        print("Training completed!")

        # Save final model
        self.save_model()

    def save_model(self, path: Optional[Path] = None):
        """Save trained model."""
        save_path = path or self.output_dir
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"Model saved to: {save_path}")

    def get_training_config(self) -> Dict[str, Any]:
        """Get current training configuration."""
        return {
            "model_name": self.model_name,
            "output_dir": str(self.output_dir),
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "max_seq_length": self.max_seq_length,
            "learning_rate": TrainingConfig.LEARNING_RATE,
            "batch_size": TrainingConfig.BATCH_SIZE,
            "num_epochs": TrainingConfig.NUM_EPOCHS,
        }
