import sys

def main():
    print("Quick code structure test\n")
    try:
        from mic_utils import compute_mic_weights
        from modelUtil import (
            InputNorm, MICNorm, FeatureNorm, FeatureNorm_MIC,
            mnist_fully_connected_IN, mnist_fully_connected_MIC
        )
    except Exception as e:
        print(f"Import failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    import torch
    model_in = mnist_fully_connected_IN(num_classes=10)
    model_mic = mnist_fully_connected_MIC(num_classes=10)
    x = torch.randn(2, 1, 28, 28)
    logits_in, _ = model_in(x)
    logits_mic, _ = model_mic(x)
    assert logits_in.shape == logits_mic.shape == (2, 10)

    import numpy as np
    from mic_utils import compute_mic_matrix, compute_mic_weights
    X = np.random.randn(20, 5)
    y = np.random.randint(0, 2, 20)
    compute_mic_matrix(X, y)
    compute_mic_weights(X, y)

    try:
        import FedAverage
        import inspect
        src = inspect.getsource(FedAverage.parse_arguments)
        assert 'mnist_fully_connected_MIC' in src
    except Exception as e:
        print(f"FedAverage check: {e}")

    print("OK")

if __name__ == "__main__":
    import traceback
    main()

