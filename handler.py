"""
jakma-darija-classifier · Pass-1 inference handler

Wraps a base xlm-roberta-base model with the jak.ma trade × city × confidence
heads. Designed for Hugging Face Inference Endpoints + Spaces.
"""
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Dict, List

CONFIG_FILENAME = "config.json"
HEADS_FILENAME = "heads.pt"
HEAD_NAMES = ("trade", "city", "confidence")


@dataclass(frozen=True)
class ClassifierConfig:
    hidden_size: int
    trade_labels: tuple[str, ...]
    city_labels: tuple[str, ...]
    confidence_labels: tuple[str, ...]

    def labels_for(self, head_name: str) -> tuple[str, ...]:
        if head_name not in HEAD_NAMES:
            raise KeyError(f"unknown classifier head: {head_name}")
        return getattr(self, f"{head_name}_labels")


def _read_positive_int(raw_config: Mapping[str, Any], key: str) -> int:
    value = raw_config.get(key)
    if type(value) is not int or value <= 0:
        raise ValueError(f"config field {key!r} must be a positive integer")
    return value


def _read_labels(raw_config: Mapping[str, Any], head_name: str) -> tuple[str, ...]:
    count_key = f"num_labels_{head_name}"
    labels_key = f"id2label_{head_name}"
    label_count = _read_positive_int(raw_config, count_key)
    id_to_label = raw_config.get(labels_key)
    if not isinstance(id_to_label, Mapping):
        raise ValueError(f"config field {labels_key!r} must be an ID-to-label mapping")

    expected_ids = {str(index) for index in range(label_count)}
    actual_ids = set(id_to_label)
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        unexpected = sorted(actual_ids - expected_ids)
        raise ValueError(
            f"config field {labels_key!r} must contain contiguous IDs 0..{label_count - 1} "
            f"(missing={missing}, unexpected={unexpected})"
        )

    labels = tuple(id_to_label[str(index)] for index in range(label_count))
    for index, label in enumerate(labels):
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"config field {labels_key!r}[{index}] must be a non-empty string")
    if len(set(labels)) != len(labels):
        raise ValueError(f"config field {labels_key!r} must not contain duplicate labels")
    return labels


def _load_classifier_config(model_path: str) -> ClassifierConfig:
    if not model_path:
        raise ValueError("model path is required to load classifier configuration")

    config_path = os.path.join(model_path, CONFIG_FILENAME)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"classifier configuration is missing: expected {config_path}"
        )
    try:
        with open(config_path, encoding="utf-8") as config_file:
            raw_config = json.load(config_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"failed to load classifier configuration from {config_path}") from exc

    if not isinstance(raw_config, Mapping):
        raise ValueError("config.json must contain a JSON object")

    return ClassifierConfig(
        hidden_size=_read_positive_int(raw_config, "hidden_size"),
        trade_labels=_read_labels(raw_config, "trade"),
        city_labels=_read_labels(raw_config, "city"),
        confidence_labels=_read_labels(raw_config, "confidence"),
    )


def _expected_head_shapes(config: ClassifierConfig) -> Dict[str, tuple[int, ...]]:
    """Return the only state-dict shape accepted by this handler."""
    shapes = {}
    for head_name in HEAD_NAMES:
        label_count = len(config.labels_for(head_name))
        shapes[f"{head_name}.weight"] = (label_count, config.hidden_size)
        shapes[f"{head_name}.bias"] = (label_count,)
    return shapes


def _validate_encoder_hidden_size(config: ClassifierConfig, actual_hidden_size: Any) -> None:
    if type(actual_hidden_size) is not int or actual_hidden_size <= 0:
        raise ValueError("encoder hidden_size must be a positive integer")
    if actual_hidden_size != config.hidden_size:
        raise ValueError(
            "encoder hidden_size does not match config.json "
            f"(encoder={actual_hidden_size}, config={config.hidden_size})"
        )


def _load_heads(
    torch: Any,
    model_path: str,
    config: ClassifierConfig,
    device: str,
) -> Any:
    """Load a complete, tensor-only classification-head state dict."""
    if not model_path:
        raise ValueError("model path is required to load trained classification heads")

    heads_path = os.path.join(model_path, HEADS_FILENAME)
    if not os.path.isfile(heads_path):
        raise FileNotFoundError(
            f"trained classification heads are missing: expected {heads_path}"
        )

    try:
        state_dict = torch.load(
            heads_path,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as exc:
        raise RuntimeError(f"failed to load trained heads from {heads_path}") from exc

    if not isinstance(state_dict, Mapping):
        raise ValueError("heads.pt must contain a state dict mapping parameter names to tensors")

    expected_shapes = _expected_head_shapes(config)
    actual_keys = set(state_dict)
    expected_keys = set(expected_shapes)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(actual_keys - expected_keys)
        raise ValueError(
            "heads.pt parameter keys do not match the classifier architecture "
            f"(missing={missing}, unexpected={unexpected})"
        )

    for name, expected_shape in expected_shapes.items():
        tensor = state_dict[name]
        if not torch.is_tensor(tensor):
            raise ValueError(f"heads.pt parameter {name!r} is not a tensor")
        actual_shape = tuple(tensor.shape)
        if actual_shape != expected_shape:
            raise ValueError(
                f"heads.pt parameter {name!r} has shape {actual_shape}; "
                f"expected {expected_shape}"
            )

    heads = torch.nn.ModuleDict({
        head_name: torch.nn.Linear(
            config.hidden_size,
            len(config.labels_for(head_name)),
        )
        for head_name in HEAD_NAMES
    })
    heads.load_state_dict(state_dict, strict=True)
    return heads.to(device).eval()


def _validate_inputs(data: Any) -> List[str]:
    """Normalize one string or a batch of strings into a non-empty batch."""
    if not isinstance(data, Mapping):
        raise TypeError("request body must be a mapping with an 'inputs' field")
    if "inputs" not in data:
        raise ValueError("request body must include an 'inputs' field")

    inputs = data["inputs"]
    if isinstance(inputs, str):
        texts = [inputs]
    elif isinstance(inputs, list):
        if not inputs:
            raise ValueError("'inputs' batch must contain at least one string")
        texts = inputs
    else:
        raise TypeError("'inputs' must be a string or a list of strings")

    for index, text in enumerate(texts):
        if not isinstance(text, str):
            raise TypeError(f"'inputs[{index}]' must be a string")
        if not text.strip():
            raise ValueError(f"'inputs[{index}]' must not be empty or whitespace")

    return texts


class EndpointHandler:
    """
    Hugging Face Inference Endpoints / Spaces handler.

    Loads the base model + classification heads, returns structured
    {trade, city, confidence} predictions.
    """

    def __init__(self, path: str = ""):
        self.classifier_config = _load_classifier_config(path)

        import torch

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.heads = _load_heads(
            torch=torch,
            model_path=path,
            config=self.classifier_config,
            device=self.device,
        )

        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.base = AutoModel.from_pretrained(path).to(self.device).eval()
        _validate_encoder_hidden_size(
            self.classifier_config,
            self.base.config.hidden_size,
        )

    def __call__(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Single or batch inference call.

        Input format:
            {"inputs": "بغيت plombier f Casa daba"}
            {"inputs": ["بغيت plombier", "Need an electrician in Rabat"]}

        Output format:
            [{
                "trade": "plumber",
                "city": "Casablanca",
                "confidence": "high",
                "scores": {
                    "trade": 0.94, "city": 0.91, "confidence": 0.89
                }
            }]
        """
        import torch

        texts = _validate_inputs(data)

        toks = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(self.device)

        with torch.no_grad():
            out = self.base(**toks)
            pooled = out.last_hidden_state[:, 0]  # [CLS]

            trade_logits = self.heads["trade"](pooled)
            city_logits = self.heads["city"](pooled)
            conf_logits = self.heads["confidence"](pooled)

            trade_probs = torch.softmax(trade_logits, dim=-1)
            city_probs = torch.softmax(city_logits, dim=-1)
            conf_probs = torch.softmax(conf_logits, dim=-1)

        predictions = []
        for row in range(len(texts)):
            trade_idx = int(trade_probs[row].argmax().item())
            city_idx = int(city_probs[row].argmax().item())
            conf_idx = int(conf_probs[row].argmax().item())

            predictions.append({
                "trade": self.classifier_config.trade_labels[trade_idx],
                "city": self.classifier_config.city_labels[city_idx],
                "confidence": self.classifier_config.confidence_labels[conf_idx],
                "scores": {
                    "trade": float(trade_probs[row, trade_idx].item()),
                    "city": float(city_probs[row, city_idx].item()),
                    "confidence": float(conf_probs[row, conf_idx].item()),
                },
            })

        return predictions
