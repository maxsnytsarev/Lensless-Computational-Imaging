from pathlib import Path
import gdown
import zipfile

def download_dataset(google_url, path_to_dataset):
    assert google_url != "https://drive.google.com/...."
    path_to_dataset = Path(path_to_dataset)
    zip_path = Path("demo_dataset.zip")
    print("Downloading dataset zip...")
    download = gdown.download(url=google_url, output=str(zip_path), quiet=False, fuzzy=True)
    if download is None or not zipfile.is_zipfile(zip_path):
        raise RuntimeError("Failed to download valid .zip file.")
    path_to_dataset.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as file:
        file.extractall(path_to_dataset)
    print(f"Dataset saved to: {str(path_to_dataset)}")
