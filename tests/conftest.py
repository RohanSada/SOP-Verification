import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

REPO = Path(__file__).resolve().parents[1]


def _has_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    ann = os.environ.get("SOPVERIFY_TEST_ANNOTATIONS") or os.environ.get("SOPVERIFY_ANNOTATIONS")
    skip_data = pytest.mark.skip(reason="IMPACT annotations not available (set SOPVERIFY_TEST_ANNOTATIONS)")
    skip_gpu = pytest.mark.skip(reason="CUDA not available")
    cuda = _has_cuda()
    for item in items:
        if "data" in item.keywords and not (ann and Path(ann).exists()):
            item.add_marker(skip_data)
        if "gpu" in item.keywords and not cuda:
            item.add_marker(skip_gpu)


@pytest.fixture
def repo() -> Path:
    return REPO


@pytest.fixture
def impact_annotations() -> Path:
    return Path(os.environ.get("SOPVERIFY_TEST_ANNOTATIONS") or os.environ["SOPVERIFY_ANNOTATIONS"])
