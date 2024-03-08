import gdown


def download_model(model_url: str, save_dir: str) -> None:
    gdown.download(model_url, save_dir, quiet=False)


def get_model_url(model_name: str) -> str:
    return f"https://github.com/onnx/models/raw/main/validated/vision/body_analysis/ultraface/models/{model_name}"
