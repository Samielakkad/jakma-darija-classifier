"""
jakma-darija-classifier · Pass-1 inference handler

Wraps a base xlm-roberta-base model with the jak.ma trade × city × confidence
heads. Designed for Hugging Face Inference Endpoints + Spaces.
"""
from collections.abc import Mapping
from typing import Any, Dict, List
import os

# Base model — published weights are a thin policy + tokenizer extension.
# Full LoRA-tuned weights for the classification heads are released alongside.
BASE_MODEL = "xlm-roberta-base"

TRADES = [
    "plumber", "electrician", "tiler", "painter", "carpenter", "mason",
    "mechanic", "electronics", "ac_technician", "gardener", "cleaner", "mover",
]
CITIES = [
    "Casablanca", "Rabat", "Sale", "Tangier", "Marrakesh", "Agadir", "Fes",
    "Meknes", "Oujda", "Kenitra", "Tetouan", "Nador", "Beni Mellal",
    "El Jadida", "Mohammedia",
]
CONFIDENCE = ["low", "medium", "high"]
HEADS_FILENAME = "heads.pt"


def _expected_head_shapes(hidden_size: int) -> Dict[str, tuple[int, ...]]:
    """Return the only state-dict shape accepted by this handler."""
    return {
        "trade.weight": (len(TRADES), hidden_size),
        "trade.bias": (len(TRADES),),
        "city.weight": (len(CITIES), hidden_size),
        "city.bias": (len(CITIES),),
        "confidence.weight": (len(CONFIDENCE), hidden_size),
        "confidence.bias": (len(CONFIDENCE),),
    }


def _load_heads(torch: Any, model_path: str, hidden_size: int, device: str) -> Any:
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

    expected_shapes = _expected_head_shapes(hidden_size)
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
        "trade": torch.nn.Linear(hidden_size, len(TRADES)),
        "city": torch.nn.Linear(hidden_size, len(CITIES)),
        "confidence": torch.nn.Linear(hidden_size, len(CONFIDENCE)),
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
        from transformers import AutoTokenizer, AutoModel
        import torch

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(path or BASE_MODEL)
        self.base = AutoModel.from_pretrained(path or BASE_MODEL).to(self.device).eval()

        self.heads = _load_heads(
            torch=torch,
            model_path=path,
            hidden_size=self.base.config.hidden_size,
            device=self.device,
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
                "trade": TRADES[trade_idx],
                "city": CITIES[city_idx],
                "confidence": CONFIDENCE[conf_idx],
                "scores": {
                    "trade": float(trade_probs[row, trade_idx].item()),
                    "city": float(city_probs[row, city_idx].item()),
                    "confidence": float(conf_probs[row, conf_idx].item()),
                },
            })

        return predictions
