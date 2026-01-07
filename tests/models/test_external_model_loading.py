"""Tests for loading external DRPModel implementations via the --models parameter."""

from types import SimpleNamespace

from drevalpy.models import MODEL_FACTORY
from drevalpy.utils import normalize_models_and_register_externals


def test_external_model_spec_registers_and_normalizes_models() -> None:
    """
    Using a ClassName:/path specification should register the external model and
    normalize args.models to plain model names.
    """
    external_model_spec = "DrugGNN:models/external_model/drug_gnn.py"

    # args mimics the argparse.Namespace passed into main()
    args = SimpleNamespace(models=["NaiveDrugMeanPredictor", external_model_spec])

    # Preserve any pre-existing registration for DrugGNN to avoid side effects
    original_druggnn = MODEL_FACTORY.get("DrugGNN")
    try:
        normalize_models_and_register_externals(args)

        # The plain name should be used after normalization
        assert args.models == ["NaiveDrugMeanPredictor", "DrugGNN"]

        # The external model should now be registered in MODEL_FACTORY
        assert "DrugGNN" in MODEL_FACTORY
        loaded_cls = MODEL_FACTORY["DrugGNN"]

        # It should be a class type
        assert isinstance(loaded_cls, type)

        # The dynamically loaded class should come from the helper's synthetic module
        assert loaded_cls.__module__ == "external_DrugGNN"
    finally:
        # Restore original registration to avoid influencing other tests
        if original_druggnn is not None:
            MODEL_FACTORY["DrugGNN"] = original_druggnn
        elif "DrugGNN" in MODEL_FACTORY:
            del MODEL_FACTORY["DrugGNN"]


