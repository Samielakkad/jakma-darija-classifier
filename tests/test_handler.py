import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from handler import CITIES, CONFIDENCE, TRADES, EndpointHandler, _load_heads, _validate_inputs


class FakeTensor:
    def __init__(self, shape):
        self.shape = shape


class FakeLinear:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features


class FakeModuleDict(dict):
    def __init__(self, modules):
        super().__init__(modules)
        self.loaded_state = None
        self.strict = None
        self.device = None
        self.evaluating = False

    def load_state_dict(self, state_dict, strict):
        self.loaded_state = state_dict
        self.strict = strict

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.evaluating = True
        return self


class FakeNN:
    Linear = FakeLinear
    ModuleDict = FakeModuleDict


class FakeTorch:
    nn = FakeNN()

    def __init__(self, state_dict=None, load_error=None):
        self.state_dict = state_dict
        self.load_error = load_error
        self.load_calls = []

    def load(self, path, **kwargs):
        self.load_calls.append((path, kwargs))
        if self.load_error:
            raise self.load_error
        return self.state_dict

    @staticmethod
    def is_tensor(value):
        return isinstance(value, FakeTensor)


class FakeScalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class FakeRow:
    def __init__(self, values):
        self.values = values

    def argmax(self):
        return FakeScalar(max(range(len(self.values)), key=self.values.__getitem__))


class FakeMatrix:
    def __init__(self, rows):
        self.rows = rows

    def __getitem__(self, key):
        if isinstance(key, tuple):
            row, column = key
            return FakeScalar(self.rows[row][column])
        return FakeRow(self.rows[key])


class FakeNoGrad:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class FakeInferenceTorch:
    @staticmethod
    def no_grad():
        return FakeNoGrad()

    @staticmethod
    def softmax(logits, dim):
        if dim != -1:
            raise AssertionError("softmax must operate over the label dimension")
        return logits


class FakeTokens(dict):
    def __init__(self):
        super().__init__({"input_ids": "fake-input-ids"})
        self.device = None

    def to(self, device):
        self.device = device
        return self


class FakeTokenizer:
    def __init__(self):
        self.calls = []
        self.tokens = FakeTokens()

    def __call__(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        return self.tokens


class FakeHiddenState:
    def __getitem__(self, key):
        if key != (slice(None), 0):
            raise AssertionError("handler must pool the first token for every row")
        return "pooled-batch"


class FakeBase:
    def __init__(self):
        self.calls = []

    def __call__(self, **tokens):
        self.calls.append(tokens)
        return SimpleNamespace(last_hidden_state=FakeHiddenState())


class FakeHead:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def __call__(self, pooled):
        self.calls.append(pooled)
        return FakeMatrix(self.rows)


def valid_state_dict(hidden_size=4):
    return {
        "trade.weight": FakeTensor((len(TRADES), hidden_size)),
        "trade.bias": FakeTensor((len(TRADES),)),
        "city.weight": FakeTensor((len(CITIES), hidden_size)),
        "city.bias": FakeTensor((len(CITIES),)),
        "confidence.weight": FakeTensor((len(CONFIDENCE), hidden_size)),
        "confidence.bias": FakeTensor((len(CONFIDENCE),)),
    }


class LoadHeadsTests(unittest.TestCase):
    def test_missing_heads_fail_fast(self):
        with tempfile.TemporaryDirectory() as model_path:
            with self.assertRaisesRegex(FileNotFoundError, "heads.pt"):
                _load_heads(FakeTorch(), model_path, hidden_size=4, device="cpu")

    def test_empty_model_path_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "model path"):
            _load_heads(FakeTorch(), "", hidden_size=4, device="cpu")

    def test_loads_tensor_only_state_dict_strictly_on_cpu(self):
        state_dict = valid_state_dict()
        fake_torch = FakeTorch(state_dict)

        with tempfile.TemporaryDirectory() as model_path:
            heads_path = Path(model_path, "heads.pt")
            heads_path.touch()
            heads = _load_heads(fake_torch, model_path, hidden_size=4, device="cuda")

        self.assertEqual(
            fake_torch.load_calls,
            [(str(heads_path), {"map_location": "cpu", "weights_only": True})],
        )
        self.assertIs(heads.loaded_state, state_dict)
        self.assertTrue(heads.strict)
        self.assertEqual(heads.device, "cuda")
        self.assertTrue(heads.evaluating)

    def test_corrupt_checkpoint_has_actionable_error(self):
        fake_torch = FakeTorch(load_error=OSError("truncated"))

        with tempfile.TemporaryDirectory() as model_path:
            Path(model_path, "heads.pt").touch()
            with self.assertRaisesRegex(RuntimeError, "failed to load trained heads"):
                _load_heads(fake_torch, model_path, hidden_size=4, device="cpu")

    def test_non_mapping_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as model_path:
            Path(model_path, "heads.pt").touch()
            with self.assertRaisesRegex(ValueError, "state dict mapping"):
                _load_heads(FakeTorch([]), model_path, hidden_size=4, device="cpu")

    def test_missing_or_unexpected_parameters_are_rejected(self):
        state_dict = valid_state_dict()
        del state_dict["city.bias"]
        state_dict["other.bias"] = FakeTensor((1,))

        with tempfile.TemporaryDirectory() as model_path:
            Path(model_path, "heads.pt").touch()
            with self.assertRaisesRegex(ValueError, "missing=.*city.bias.*unexpected=.*other.bias"):
                _load_heads(FakeTorch(state_dict), model_path, hidden_size=4, device="cpu")

    def test_wrong_parameter_shape_is_rejected(self):
        state_dict = valid_state_dict()
        state_dict["trade.weight"] = FakeTensor((len(TRADES) + 1, 4))

        with tempfile.TemporaryDirectory() as model_path:
            Path(model_path, "heads.pt").touch()
            with self.assertRaisesRegex(ValueError, "trade.weight.*shape"):
                _load_heads(FakeTorch(state_dict), model_path, hidden_size=4, device="cpu")

    def test_non_tensor_parameter_is_rejected(self):
        state_dict = valid_state_dict()
        state_dict["confidence.bias"] = [0, 0, 0]

        with tempfile.TemporaryDirectory() as model_path:
            Path(model_path, "heads.pt").touch()
            with self.assertRaisesRegex(ValueError, "confidence.bias.*not a tensor"):
                _load_heads(FakeTorch(state_dict), model_path, hidden_size=4, device="cpu")


class ValidateInputsTests(unittest.TestCase):
    def test_single_string_becomes_one_item_batch(self):
        self.assertEqual(_validate_inputs({"inputs": "bghit plombier"}), ["bghit plombier"])

    def test_string_batch_is_preserved(self):
        texts = ["bghit plombier", "need an electrician"]
        self.assertEqual(_validate_inputs({"inputs": texts}), texts)

    def test_request_must_be_a_mapping(self):
        with self.assertRaisesRegex(TypeError, "request body"):
            _validate_inputs("bghit plombier")

    def test_inputs_field_is_required(self):
        with self.assertRaisesRegex(ValueError, "include an 'inputs' field"):
            _validate_inputs({})

    def test_inputs_must_have_supported_type(self):
        for value in (None, 7, ("one", "two"), {"text": "one"}):
            with self.subTest(value=value):
                with self.assertRaisesRegex(TypeError, "string or a list"):
                    _validate_inputs({"inputs": value})

    def test_empty_string_and_batch_are_rejected(self):
        for value in ("", "  \t", []):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    _validate_inputs({"inputs": value})

    def test_invalid_batch_items_report_their_index(self):
        with self.assertRaisesRegex(TypeError, r"inputs\[1\].*string"):
            _validate_inputs({"inputs": ["valid", 4]})
        with self.assertRaisesRegex(ValueError, r"inputs\[1\].*empty"):
            _validate_inputs({"inputs": ["valid", "  "]})


class BatchInferenceTests(unittest.TestCase):
    def test_batch_uses_one_encoder_pass_and_returns_one_prediction_per_input(self):
        handler = EndpointHandler.__new__(EndpointHandler)
        handler.device = "cpu"
        handler.tokenizer = FakeTokenizer()
        handler.base = FakeBase()
        handler.heads = {
            "trade": FakeHead([[0.1, 0.8, 0.1], [0.7, 0.2, 0.1]]),
            "city": FakeHead([[0.6, 0.3, 0.1], [0.1, 0.8, 0.1]]),
            "confidence": FakeHead([[0.1, 0.2, 0.7], [0.6, 0.3, 0.1]]),
        }
        texts = ["need an electrician in Casa", "bghit plombier f Rabat"]

        with patch.dict("sys.modules", {"torch": FakeInferenceTorch()}):
            predictions = handler({"inputs": texts})

        self.assertEqual(
            handler.tokenizer.calls,
            [(
                texts,
                {
                    "return_tensors": "pt",
                    "padding": True,
                    "truncation": True,
                    "max_length": 128,
                },
            )],
        )
        self.assertEqual(handler.tokenizer.tokens.device, "cpu")
        self.assertEqual(len(handler.base.calls), 1)
        self.assertEqual(handler.heads["trade"].calls, ["pooled-batch"])
        self.assertEqual(
            predictions,
            [
                {
                    "trade": "electrician",
                    "city": "Casablanca",
                    "confidence": "high",
                    "scores": {"trade": 0.8, "city": 0.6, "confidence": 0.7},
                },
                {
                    "trade": "plumber",
                    "city": "Rabat",
                    "confidence": "low",
                    "scores": {"trade": 0.7, "city": 0.8, "confidence": 0.6},
                },
            ],
        )

if __name__ == "__main__":
    unittest.main()
