from pathlib import Path
import gdown
import zipfile
import matplotlib.pyplot as plt
from PIL import Image

def download_dataset(google_url, path_to_dataset):
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

def show_result(path_to_dataset, max_images=3):
    assert max_images >= 1
    path_to_dataset = Path(path_to_dataset)
    dirs = sorted(list(path_to_dataset.iterdir()), key=lambda x: int(x.name.removeprefix("ID")))[:max_images]
    if not dirs:
        raise RuntimeError(f"No directories found in {path_to_dataset}")
    fig, axes = plt.subplots(
        1,
        len(dirs),
        figsize=(5 * len(dirs), 5),
        squeeze=False,
    )
    for i in range(len(dirs)):
        cur_dir = dirs[i]
        cur_name = dirs[i].name
        reconstructed_path = cur_dir / f"reconstructed_roi_{cur_name}.png"
        if not reconstructed_path.is_file():
            raise FileNotFoundError(f"Reconstruction not found in {reconstructed_path}")
        image = Image.open(reconstructed_path).convert("RGB")
        axes[0, i].imshow(image)
        axes[0, i].set_title(cur_name)
        axes[0, i].axis("off")
    plt.tight_layout()
    plt.show()