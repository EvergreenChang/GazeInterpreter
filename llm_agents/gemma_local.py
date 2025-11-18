"""Gemma local model inference implementation"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


logger = logging.getLogger(__name__)


def _resolve_torch_dtype(dtype: Optional[str]) -> Optional[torch.dtype]:
    """Convert string representation of dtype to torch.dtype."""
    if dtype is None or dtype == "auto":
        return None

    dtype = dtype.lower()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported torch_dtype: {dtype}")
    return mapping[dtype]


class GemmaLocalLLM:
    """Gemma local model inference class based on Transformers."""

    _MODEL_CACHE: Dict[str, Tuple[AutoTokenizer, AutoModelForCausalLM, str]] = {}

    def __init__(
        self,
        model_path: str,
        system_instruction: Optional[str] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        torch_dtype: Optional[str] = "bfloat16",
        device_map: Optional[str] = "auto",
        attn_implementation: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.05,
        use_chat_template: bool = True,
    ) -> None:
        """Initialize local Gemma model.

        Args:
            model_path: Hugging Face model weight path or repository name.
            system_instruction: Default system prompt.
            generation_config: Generation configuration dictionary.
            torch_dtype: Specify loaded dtype, e.g. "bfloat16", "float16", default is bfloat16.
            device_map: transformers load device_map, default is "auto".
            attn_implementation: Attention implementation, optional "flash_attention_2" etc., most cases keep None.
            max_new_tokens: Default maximum number of tokens to generate.
            temperature: Default temperature.
            top_p: Default top_p sampling value.
            repetition_penalty: Repetition penalty coefficient.
            use_chat_template: Whether to use chat template to wrap conversation.
        """

        self.model_path = model_path
        self.system_instruction = system_instruction
        self.generation_config = generation_config or {}
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.use_chat_template = use_chat_template
        self.attn_implementation = attn_implementation

        resolved_dtype = _resolve_torch_dtype(torch_dtype)
        cache_key = "::".join([
            model_path,
            str(resolved_dtype) if resolved_dtype else "auto",
            device_map or "none",
            attn_implementation or "standard",
        ])

        if cache_key in GemmaLocalLLM._MODEL_CACHE:
            tokenizer, model, model_device = GemmaLocalLLM._MODEL_CACHE[cache_key]
            self.tokenizer = tokenizer
            self.model = model
            self.model_device = model_device
            logger.debug("Reusing cached Gemma model: %s", cache_key)
        else:
            logger.info("Loading Gemma model: %s", model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            model_kwargs = {
                "torch_dtype": resolved_dtype,
                "device_map": device_map,
            }
            if attn_implementation:
                model_kwargs["attn_implementation"] = attn_implementation

            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    **model_kwargs,
                )
            except ValueError as exc:
                logger.warning(
                    "Specified attention implementation %s failed to load, falling back to default implementation. Error: %s",
                    attn_implementation,
                    exc,
                )
                model_kwargs.pop("attn_implementation", None)
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    **model_kwargs,
                )

            model.eval()
            if hasattr(model, "hf_device_map") and isinstance(model.hf_device_map, dict):
                devices = set(model.hf_device_map.values())
                model_device = next(iter(devices)) if len(devices) == 1 else "auto"
            else:
                model_device = next(model.parameters()).device.type

            GemmaLocalLLM._MODEL_CACHE[cache_key] = (tokenizer, model, model_device)
            self.tokenizer = tokenizer
            self.model = model
            self.model_device = model_device
            logger.info("Gemma model loaded, device: %s", model_device)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    def _build_messages(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Construct conversation message list."""
        messages: List[Dict[str, str]] = []
        system_content = system_prompt or self.system_instruction
        if system_content:
            messages.append({"role": "system", "content": system_content})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _prepare_inputs(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
        padding: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Construct model input based on messages."""
        if self.use_chat_template and hasattr(self.tokenizer, "apply_chat_template"):
            if padding:
                inputs = self.tokenizer.apply_chat_template(
                    [messages],
                    return_tensors="pt",
                    add_generation_prompt=add_generation_prompt,
                    padding=True,
                )
                # apply_chat_template returns BatchEncoding for batch input, need to take the first element
                if isinstance(inputs, dict):
                    inputs = {k: v for k, v in inputs.items()}
                    for key in inputs:
                        inputs[key] = inputs[key][0].unsqueeze(0)
                elif isinstance(inputs, torch.Tensor):
                    inputs = inputs[0].unsqueeze(0)
            else:
                inputs = self.tokenizer.apply_chat_template(
                    messages,
                    return_tensors="pt",
                    add_generation_prompt=add_generation_prompt,
                )
        else:
            prompt_text = "\n".join([m["content"] for m in messages]) + "\n"
            inputs = self.tokenizer(
                prompt_text,
                return_tensors="pt",
                padding="longest" if padding else False,
            )

        if isinstance(inputs, torch.Tensor):
            model_inputs = {"input_ids": inputs}
        else:
            model_inputs = inputs

        model_inputs = {k: v.to(self.model.device) for k, v in model_inputs.items()}
        if "attention_mask" not in model_inputs:
            model_inputs["attention_mask"] = torch.ones_like(model_inputs["input_ids"], device=self.model.device)

        if model_inputs["input_ids"].dim() == 1:
            model_inputs["input_ids"] = model_inputs["input_ids"].unsqueeze(0)
        if model_inputs["attention_mask"].dim() == 1:
            model_inputs["attention_mask"] = model_inputs["attention_mask"].unsqueeze(0)
        return model_inputs

    @staticmethod
    def _clean_response(text: str) -> str:
        if not text:
            return ""

        cleaned = text.strip()

        if cleaned.startswith("```"):
            segments = cleaned.split("```")
            cleaned_segments = [segment for segment in segments if segment.strip()]
            if cleaned_segments:
                cleaned = cleaned_segments[-1].strip()

        marker = cleaned.lower().rfind("# output:")
        if marker != -1:
            cleaned = cleaned[marker + len("# output:"):].strip()

        lowered = cleaned.lower()
        for prefix in ("assistant:", "assistant", "model:", "model"):
            if lowered.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
                lowered = cleaned.lower()

        marker_idx = cleaned.lower().rfind("the human")
        if marker_idx != -1:
            cleaned = cleaned[marker_idx:]

        return cleaned.strip()

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        **kwargs,
    ) -> str:
        """生成文本响应."""

        try:
            if isinstance(prompt, list):
                return self.generate_batch(
                    prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    **kwargs,
                )

            messages = self._build_messages(prompt, system_prompt)
            model_inputs = self._prepare_inputs(messages)
            input_ids = model_inputs["input_ids"]
            attention_mask = model_inputs["attention_mask"]

            generation_kwargs = {
                "max_new_tokens": max_tokens or self.generation_config.get("max_new_tokens", self.max_new_tokens),
                "temperature": temperature or self.generation_config.get("temperature", self.temperature),
                "top_p": top_p or self.generation_config.get("top_p", self.top_p),
                "repetition_penalty": repetition_penalty or self.generation_config.get("repetition_penalty", self.repetition_penalty),
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }

            generation_kwargs.update(kwargs)

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **generation_kwargs,
                )

            input_lengths = attention_mask.sum(dim=1)
            generated_ids = outputs[0, input_lengths[0].item():]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            # The result of single batch inference cannot use _clean_response, because the output content of the eval agent is not just a description, using _clean_response will be eliminated
            return text.strip()

        except Exception as exc:
            logger.error("Gemma local inference decoding failed: %s", exc)
            return ""

    def generate_batch(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        **kwargs,
    ) -> List[str]:
        """Batch generate text response."""

        if not prompts:
            return []

        try:
            messages_list = [self._build_messages(prompt, system_prompt) for prompt in prompts]
            model_inputs = self.tokenizer.apply_chat_template(
                messages_list,
                return_tensors="pt",
                add_generation_prompt=True,
                padding=True,
            )

            if isinstance(model_inputs, torch.Tensor):
                input_ids = model_inputs.to(self.model.device)
                if self.tokenizer.pad_token_id is None:
                    attention_mask = torch.ones_like(input_ids, device=self.model.device)
                else:
                    attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long().to(self.model.device)
            else:
                input_ids = model_inputs["input_ids"].to(self.model.device)
                attention_mask = model_inputs["attention_mask"].to(self.model.device)

            generation_kwargs = {
                "max_new_tokens": max_tokens or self.generation_config.get("max_new_tokens", self.max_new_tokens),
                "temperature": temperature or self.generation_config.get("temperature", self.temperature),
                "top_p": top_p or self.generation_config.get("top_p", self.top_p),
                "repetition_penalty": repetition_penalty or self.generation_config.get("repetition_penalty", self.repetition_penalty),
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }

            generation_kwargs.update(kwargs)

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **generation_kwargs,
                )

            input_lengths = attention_mask.sum(dim=1)
            results: List[str] = []
            for i in range(outputs.size(0)):
                start_idx = int(input_lengths[i].item())
                generated_ids = outputs[i, start_idx:]
                text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                results.append(self._clean_response(text))

            return results

        except Exception as exc:
            logger.error("Gemma local batch inference decoding failed: %s", exc)
            # Degrade to single generation
            return [self.generate(prompt, system_prompt, temperature, max_tokens, top_p, repetition_penalty, **kwargs) for prompt in prompts]

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """Generate reply based on multi-turn conversation."""

        try:
            model_inputs = self._prepare_inputs(messages)
            input_ids = model_inputs["input_ids"]
            attention_mask = model_inputs["attention_mask"]

            generation_kwargs = {
                "max_new_tokens": max_tokens or self.generation_config.get("max_new_tokens", self.max_new_tokens),
                "temperature": temperature or self.generation_config.get("temperature", self.temperature),
                "top_p": self.generation_config.get("top_p", self.top_p),
                "repetition_penalty": self.generation_config.get("repetition_penalty", self.repetition_penalty),
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }

            generation_kwargs.update(kwargs)

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **generation_kwargs,
                )

            input_lengths = attention_mask.sum(dim=1)
            generated_ids = outputs[0, input_lengths[0].item():]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            return text.strip()

        except Exception as exc:
            logger.error("Gemma local conversation generation failed: %s", exc)
            return ""
