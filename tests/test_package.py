"""Test suite for dataset_generator package."""

import dataset_generator


def test_version() -> None:
    """Test package version is accessible."""
    assert hasattr(dataset_generator, "__version__")
    assert isinstance(dataset_generator.__version__, str)
    assert len(dataset_generator.__version__) > 0
