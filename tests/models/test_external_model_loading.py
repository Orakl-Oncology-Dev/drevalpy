"""Tests for loading external DRPModel implementations via the --models parameter."""

from types import SimpleNamespace

from drevalpy.models import MODEL_FACTORY
from drevalpy.utils import normalize_models_and_register_externals


def test_external_model_spec_registers_and_normalizes_models() -> None:
    """
    Using a ClassName:/path specification should register the external model and
    normalize args.models to plain model names.
    """
    external_model_spec = "DrugGNN_External:models/external_model/drug_gnn.py"

    # args mimics the argparse.Namespace passed into main()
    args = SimpleNamespace(models=["NaiveDrugMeanPredictor", external_model_spec])

    normalize_models_and_register_externals(args)

    # The plain name should be used after normalization
    assert args.models == ["NaiveDrugMeanPredictor", "DrugGNN_External"]

    # The external model should now be registered in MODEL_FACTORY
    assert "DrugGNN_External" in MODEL_FACTORY
    loaded_cls = MODEL_FACTORY["DrugGNN_External"]

    # It should be a class type
    assert isinstance(loaded_cls, type)

    # The dynamically loaded class should come from the helper's synthetic module
    assert loaded_cls.__module__ == "external_DrugGNN_External"
