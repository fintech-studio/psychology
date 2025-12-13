import torch


def check_env():
    """Check torch environment; safe to import without auto-print."""
    print("PyTorch version:", torch.__version__)
    print("PyTorch file:", torch.__file__)
    has_gpu = torch.cuda.is_available()
    print("CUDA available:", has_gpu)
    device_name = (
        torch.cuda.get_device_name(0) if has_gpu else "No GPU"
    )
    print("CUDA device:", device_name)


if __name__ == "__main__":
    check_env()
