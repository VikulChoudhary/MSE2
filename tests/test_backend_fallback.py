"""Make sure blocked scikit-learn DLLs never break HELIOS."""
import builtins
from helios.training_model import FraudModel


def test_sklearn_and_python_backends_use_same_saved_format(monkeypatch):
    transactions = [[10, 11], [20, 9], [300, 3], [420, 2]]
    labels = [0, 0, 1, 1]
    monkeypatch.setenv("HELIOS_ML_BACKEND", "python")
    model = FraudModel(epochs=30).fit(transactions, labels)
    assert model.backend == "pure_python"
    restored = FraudModel.from_dict(model.to_dict())
    assert restored.predict(transactions) == model.predict(transactions)


def test_falls_back_when_windows_blocks_sklearn(monkeypatch):
    original_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name.startswith("sklearn"):
            raise ImportError("DLL load failed: Application Control policy blocked this file")
        return original_import(name, *args, **kwargs)

    monkeypatch.setenv("HELIOS_ML_BACKEND", "auto")
    monkeypatch.setattr(builtins, "__import__", blocked_import)
    model = FraudModel(epochs=30).fit(
        [[10, 1], [12, 2], [300, 9], [400, 10]], [0, 0, 1, 1]
    )
    assert model.backend == "pure_python"
    assert len(model.predict([[25, 2]])) == 1
