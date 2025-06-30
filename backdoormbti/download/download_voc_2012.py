import os
import requests
from tqdm import tqdm
import tarfile


def download_file(url, dest_path):
    """Download a file from the given URL with a progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 KB
    t = tqdm(
        total=total_size,
        unit="B",
        unit_scale=True,
        desc=f"Downloading {os.path.basename(dest_path)}",
    )

    with open(dest_path, "wb") as f:
        for data in response.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()


def extract_file(file_path, extract_to):
    """Extract a tar file to the specified directory."""
    if tarfile.is_tarfile(file_path):
        print(f"Extracting {file_path}...")
        with tarfile.open(file_path, "r") as tar:
            tar.extractall(path=extract_to)
        print(f"Extraction complete. Files extracted to: {extract_to}")
    else:
        print(f"{file_path} is not a valid tar file.")


def main():
    # VOC2012 dataset URL
    url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
    dest_dir = "../data/PascalVOC/"
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, os.path.basename(url))

    # Check if the file already exists
    if not os.path.exists(dest_path):
        print("Downloading VOC2012 dataset...")
        download_file(url, dest_path)
    else:
        print(f"{os.path.basename(dest_path)} already exists, skipping download.")

    # Extract the file
    extract_file(dest_path, dest_dir)


if __name__ == "__main__":
    main()
