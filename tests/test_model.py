import numpy as np
from src.model import build_cnn_lstm


def test_model_output_shape():
    model = build_cnn_lstm((200, 40, 1), 4)
    out = model(np.zeros((2, 200, 40, 1), dtype=np.float32), training=False)
    assert tuple(out.shape) == (2, 4)
    assert np.allclose(np.sum(out.numpy(), axis=1), 1.0, atol=1e-5)
