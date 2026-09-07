# ==========================================================
# Neura-X Tests: Modular Composability
# ==========================================================

import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from neura_x.module import Module, FractalLayer
from neura_x.compose import compose, ComposedModel


def test_compose_creation():
    """Test composing a model."""
    brain = FractalLayer(in_features=10, out_features=10)
    module1 = FractalLayer(in_features=10, out_features=10)

    model = compose(brain=brain, modules=[module1])
    assert isinstance(model, ComposedModel)
    print("✅ test_compose_creation passed")


def test_compose_attach_detach():
    """Test hot-swapping modules."""
    brain = FractalLayer(in_features=10, out_features=10)
    model = compose(brain=brain, modules=[])

    # Attach
    new_module = FractalLayer(in_features=10, out_features=10)
    model.attach(new_module)
    assert len(model.list_modules()) == 1

    # Detach
    name = model.list_modules()[0]
    model.detach(name)
    assert len(model.list_modules()) == 0

    print("✅ test_compose_attach_detach passed")


if __name__ == "__main__":
    test_compose_creation()
    test_compose_attach_detach()
    print("\n🎉 All Compose tests passed!")