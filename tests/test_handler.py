import json
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from handler import (
    ClassifierConfig,
    EndpointHandler,
    _expected_head_shapes,
    _load_classifier_config,
    _load_heads,
    _validate_encoder_hidden_size,
    _validate_inputs,
)


TEST_CONFIG = ClassifierConfig(
    hidden_size=4,
    trade_labels=("plumber", "electrician", "tiler"),
    city_labels=("Casablanca", "Rabat", "Tangier"),
    confidence_labels=("low", "medium", "high"),
)


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


class FakeCuda:
    @staticmethod
    def is_available():
        return False


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


def valid_state_dict(config=TEST_CONFIG):
    return {
        name: FakeTensor(shape)
        for name, shape in _expected_head_shapes(config).items()
    }


def valid_raw_config():
    return {
        "hidden_size": 4,
        "num_labels_trade": 3,
        "num_labels_city": 3,
        "num_labels_confidence": 3,
        "id2label_trade": {"2": "tiler", "0": "plumber", "1": "electrician"},
        "id2label_city": {"0": "Casablanca", "1": "Rabat", "2": "Tangier"},
        "id2label_confidence": {"0": "low", "1": "medium", "2": "high"},
    }


def write_config(model_path, raw_config):
    with Path(model_path, "config.json").open("w", encoding="utf-8") as config_file:
        json.dump(raw_config, config_file)


class LoadHeadsTests(unittest.TestCase):
    def test_missing_heads_fail_fast(self):
        with tempfile.TemporaryDirectory() as model_path:
            with self.assertRaisesRegex(FileNotFoundError, "heads.pt"):
                _load_heads(FakeTorch(), model_path, config=TEST_CONFIG, device="cpu")

    def test_empty_model_path_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "model path"):
            _load_heads(FakeTorch(), "", config=TEST_CONFIG, device="cpu")

    def test_loads_tensor_only_state_dict_strictly_on_cpu(self):
        state_dict = valid_state_dict()
        fake_torch = FakeTorch(state_dict)

        with tempfile.TemporaryDirectory() as model_path:
            heads_path = Path(model_path, "heads.pt")
            heads_path.touch()
            heads = _load_heads(fake_torch, model_path, config=TEST_CONFIG, device="cuda")

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
                _load_heads(fake_torch, model_path, config=TEST_CONFIG, device="cpu")

    def test_non_mapping_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as model_path:
            Path(model_path, "heads.pt").touch()
            with self.assertRaisesRegex(ValueError, "state dict mapping"):
                _load_heads(FakeTorch([]), model_path, config=TEST_CONFIG, device="cpu")

    def test_missing_or_unexpected_parameters_are_rejected(self):
        state_dict = valid_state_dict()
        del state_dict["city.bias"]
        state_dict["other.bias"] = FakeTensor((1,))

        with tempfile.TemporaryDirectory() as model_path:
            Path(model_path, "heads.pt").touch()
            with self.assertRaisesRegex(ValueError, "missing=.*city.bias.*unexpected=.*other.bias"):
                _load_heads(FakeTorch(state_dict), model_path, config=TEST_CONFIG, device="cpu")

    def test_wrong_parameter_shape_is_rejected(self):
        state_dict = valid_state_dict()
        state_dict["trade.weight"] = FakeTensor((len(TEST_CONFIG.trade_labels) + 1, 4))

        with tempfile.TemporaryDirectory() as model_path:
            Path(model_path, "heads.pt").touch()
            with self.assertRaisesRegex(ValueError, "trade.weight.*shape"):
                _load_heads(FakeTorch(state_dict), model_path, config=TEST_CONFIG, device="cpu")

    def test_non_tensor_parameter_is_rejected(self):
        state_dict = valid_state_dict()
        state_dict["confidence.bias"] = [0, 0, 0]

        with tempfile.TemporaryDirectory() as model_path:
            Path(model_path, "heads.pt").touch()
            with self.assertRaisesRegex(ValueError, "confidence.bias.*not a tensor"):
                _load_heads(FakeTorch(state_dict), model_path, config=TEST_CONFIG, device="cpu")


class ClassifierConfigTests(unittest.TestCase):
    def test_loads_labels_by_numeric_id_not_json_insertion_order(self):
        with tempfile.TemporaryDirectory() as model_path:
            write_config(model_path, valid_raw_config())
            config = _load_classifier_config(model_path)

        self.assertEqual(config, TEST_CONFIG)

    def test_missing_config_fails_fast(self):
        with tempfile.TemporaryDirectory() as model_path:
            with self.assertRaisesRegex(FileNotFoundError, "config.json"):
                _load_classifier_config(model_path)

    def test_malformed_json_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as model_path:
            Path(model_path, "config.json").write_text("{broken", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "failed to load classifier configuration"):
                _load_classifier_config(model_path)

    def test_label_count_and_ids_must_agree(self):
        raw_config = valid_raw_config()
        del raw_config["id2label_city"]["1"]
        raw_config["id2label_city"]["9"] = "Rabat"

        with tempfile.TemporaryDirectory() as model_path:
            write_config(model_path, raw_config)
            with self.assertRaisesRegex(ValueError, "contiguous IDs.*missing=.*1.*unexpected=.*9"):
                _load_classifier_config(model_path)

    def test_labels_must_be_unique_non_empty_strings(self):
        invalid_labels = ("", "  ", None, "plumber")
        for invalid_label in invalid_labels:
            raw_config = valid_raw_config()
            raw_config["id2label_trade"]["1"] = invalid_label
            with self.subTest(label=invalid_label), tempfile.TemporaryDirectory() as model_path:
                write_config(model_path, raw_config)
                with self.assertRaises(ValueError):
                    _load_classifier_config(model_path)

    def test_labels_are_stripped_before_duplicate_validation(self):
        raw_config = valid_raw_config()
        raw_config["id2label_trade"]["0"] = " plumber "

        with tempfile.TemporaryDirectory() as model_path:
            write_config(model_path, raw_config)
            config = _load_classifier_config(model_path)

        self.assertEqual(config.trade_labels[0], "plumber")

        raw_config["id2label_trade"]["1"] = "plumber"
        with tempfile.TemporaryDirectory() as model_path:
            write_config(model_path, raw_config)
            with self.assertRaisesRegex(ValueError, "duplicate labels"):
                _load_classifier_config(model_path)

    def test_head_shapes_come_from_config(self):
        self.assertEqual(
            _expected_head_shapes(TEST_CONFIG),
            {
                "trade.weight": (3, 4),
                "trade.bias": (3,),
                "city.weight": (3, 4),
                "city.bias": (3,),
                "confidence.weight": (3, 4),
                "confidence.bias": (3,),
            },
        )

    def test_encoder_hidden_size_must_match_config(self):
        _validate_encoder_hidden_size(TEST_CONFIG, 4)
        with self.assertRaisesRegex(ValueError, "encoder=8, config=4"):
            _validate_encoder_hidden_size(TEST_CONFIG, 8)

    def test_checked_in_config_is_valid(self):
        repository_root = str(Path(__file__).resolve().parents[1])
        config = _load_classifier_config(repository_root)
        self.assertEqual(config.hidden_size, 768)
        self.assertEqual(
            tuple(map(len, (
                config.trade_labels,
                config.city_labels,
                config.confidence_labels,
            ))),
            (12, 15, 3),
        )


class HandlerInitializationTests(unittest.TestCase):
    def test_missing_heads_fail_before_transformers_load_model_artifacts(self):
        fake_torch = FakeTorch()
        fake_torch.cuda = FakeCuda()
        fake_transformers = ModuleType("transformers")

        class UnexpectedFactory:
            @classmethod
            def from_pretrained(cls, path):
                raise AssertionError(f"Transformers loaded before heads validation: {path}")

        fake_transformers.AutoModel = UnexpectedFactory
        fake_transformers.AutoTokenizer = UnexpectedFactory

        with tempfile.TemporaryDirectory() as model_path:
            write_config(model_path, valid_raw_config())
            with patch.dict(
                "sys.modules",
                {"torch": fake_torch, "transformers": fake_transformers},
            ):
                with self.assertRaisesRegex(FileNotFoundError, "heads.pt"):
                    EndpointHandler(model_path)


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
        handler.classifier_config = TEST_CONFIG
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
