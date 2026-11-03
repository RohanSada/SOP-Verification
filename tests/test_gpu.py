import numpy as np
import pytest

pytestmark = pytest.mark.gpu


def test_dinov2_smoke():
    from sopverify.features.dinov2 import DINOv2Extractor

    ex = DINOv2Extractor("facebook/dinov2-base", batch_size=4)
    feats = ex.encode(np.random.randint(0, 255, (6, 224, 224, 3), dtype=np.uint8))
    assert feats.shape == (6, 768) and np.allclose(np.linalg.norm(feats, axis=1), 1, atol=1e-3)
