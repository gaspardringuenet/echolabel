"""Helpers to download demo data"""

import logging
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

# URL adress of demo data file
URL = "https://github.com/gaspardringuenet/echolabel/releases/download/demo-data-v1/demo.zip"

# Init logger
logger = logging.getLogger(__name__)


def download_file(url: str, output_dir: Path) -> Path:
    """Download a file from a url using requests"""

    local_filename = url.split("/")[-1]
    output_file = output_dir / local_filename

    headers = {"User-Agent": "curl/8.0", "Accept-Encoding": "identity"}

    with requests.get(url, stream=True, headers=headers) as r:
        r.raise_for_status()
        total_size = int(r.headers.get("content-length", 0))

        with (
            open(output_file, "wb") as f,
            tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading demo data") as pbar,
        ):
            for chunk in r.raw.stream(1024 * 1024, decode_content=False):
                f.write(chunk)
                pbar.update(len(chunk))

    return output_file


def download_demo_data(cache_dir: Path) -> Path:
    """Download sample data from GitHub release"""

    zipped_filename = URL.split("/")[-1]
    output_file = (cache_dir / zipped_filename).with_suffix(".zarr")

    if output_file.exists():
        logger.warning(f"Demo data file already exists at {output_file}.")
        return output_file

    output_file.parent.mkdir(exist_ok=True, parents=True)

    logger.info("Downloading demo data.")

    zipped_file = download_file(URL, cache_dir)

    logger.info("Unzipping.")
    with zipfile.ZipFile(zipped_file, "r") as zip_ref:
        zip_ref.extractall(output_file.parent)

    zipped_file.unlink()
    logger.info(f"Sample data saved to {output_file}")

    return output_file
