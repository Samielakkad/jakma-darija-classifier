import tempfile
import unittest
from pathlib import Path

from handler import CITIES, CONFIDENCE, TRADES, _load_heads


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


if __name__ == "__main__":
    unittest.main()
