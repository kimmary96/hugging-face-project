"""
Inference engine for LLM-based user profile analysis.
Wraps Unsloth/Qwen models with common inference patterns.
"""
import json
import re
from typing import Dict, Optional, List
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import Models, TrainingConfig, OUTPUTS_DIR
from config.prompts import SYSTEM_PROMPT_MATCHING, INSTRUCTION_CATEGORY_CLASSIFICATION


class InferenceEngine:
    """
    LLM inference engine for user profile analysis.

    Handles model loading, prompt formatting, and response parsing.
    """

    def __init__(
        self,
        model_path: str = None,
        max_seq_length: int = None,
        load_in_4bit: bool = True
    ):
        """
        Initialize inference engine.

        Args:
            model_path: Path to model (default: outputs_1.7b)
            max_seq_length: Maximum sequence length
            load_in_4bit: Whether to use 4-bit quantization
        """
        self.model_path = model_path or str(OUTPUTS_DIR)
        self.max_seq_length = max_seq_length or TrainingConfig.MAX_SEQ_LENGTH
        self.load_in_4bit = load_in_4bit
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load the model and tokenizer."""
        from unsloth import FastLanguageModel

        print(f"Loading model: {self.model_path}")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_path,
            max_seq_length=self.max_seq_length,
            dtype=None,
            load_in_4bit=self.load_in_4bit,
        )
        FastLanguageModel.for_inference(self.model)
        print("Model loaded successfully!")

    def analyze_user_profile(
        self,
        user_input: str,
        system_prompt: str = None,
        temperature: float = 0.1,
        max_new_tokens: int = 512
    ) -> Optional[Dict]:
        """
        Analyze user input and extract profile with category and tags.

        Args:
            user_input: User's item list, price, and frequency
            system_prompt: Custom system prompt (optional)
            temperature: Generation temperature (lower = more deterministic)
            max_new_tokens: Maximum tokens to generate

        Returns:
            Dictionary with category, tags, and reasoning, or None if parsing fails
        """
        if self.model is None:
            self.load_model()

        system_prompt = system_prompt or SYSTEM_PROMPT_MATCHING

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt"
        ).to("cuda")

        outputs = self.model.generate(
            input_ids=inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )

        decoded = self.tokenizer.decode(
            outputs[0][inputs.shape[1]:],
            skip_special_tokens=True
        )

        return self._parse_json_response(decoded)

    def classify_category(
        self,
        user_input: str,
        temperature: float = 0.7
    ) -> Optional[str]:
        """
        Classify user input into one of the 5 categories.

        Args:
            user_input: User's item list description
            temperature: Generation temperature

        Returns:
            Category string or None if parsing fails
        """
        if self.model is None:
            self.load_model()

        messages = [
            {"role": "system", "content": INSTRUCTION_CATEGORY_CLASSIFICATION},
            {"role": "user", "content": user_input}
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt"
        ).to("cuda")

        outputs = self.model.generate(
            input_ids=inputs,
            max_new_tokens=512,
            temperature=temperature,
            top_p=TrainingConfig.TOP_P,
            top_k=TrainingConfig.TOP_K,
        )

        decoded = self.tokenizer.decode(
            outputs[0][inputs.shape[1]:],
            skip_special_tokens=True
        )

        result = self._parse_json_response(decoded)
        return result.get('category') if result else None

    def _parse_json_response(self, text: str) -> Optional[Dict]:
        """
        Extract and parse JSON from model response.

        Args:
            text: Raw model output text

        Returns:
            Parsed dictionary or None if parsing fails
        """
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON block
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end+1])
            except json.JSONDecodeError:
                pass

        # Try regex for category
        match = re.search(r'"category"\s*:\s*"([^"]+)"', text)
        if match:
            return {"category": match.group(1)}

        return None

    def generate_raw(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7
    ) -> str:
        """
        Generate raw text response without JSON parsing.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Generation temperature

        Returns:
            Generated text string
        """
        if self.model is None:
            self.load_model()

        inputs = self.tokenizer(
            [prompt],
            return_tensors="pt"
        ).to("cuda")

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=TrainingConfig.TOP_P,
            top_k=TrainingConfig.TOP_K,
        )

        return self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
