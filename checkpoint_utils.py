"""Utilities for locating and downloading the pretrained SYLPH checkpoint."""

from pathlib import Path


HF_REPO_ID = "hechengyang/sylph"
HF_FILENAME = "net_checkpoint.pkl"

# For maximum reproducibility, replace "main" later with a full Hugging Face
# commit hash or a release tag such as "v1.0.0".
HF_REVISION = "main"

REPO_ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = REPO_ROOT / "models" / "sylph"
DEFAULT_CHECKPOINT_PATH = CHECKPOINT_DIR / HF_FILENAME

MIN_CHECKPOINT_SIZE_BYTES = 100 * 1024 * 1024


def get_sylph_checkpoint(
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
    *,
    force_download: bool = False,
) -> Path:
    """Return the checkpoint path, downloading it from Hugging Face if needed."""

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()

    if checkpoint_path.name != HF_FILENAME:
        raise ValueError(
            f"The checkpoint filename must be {HF_FILENAME!r}, "
            f"not {checkpoint_path.name!r}."
        )

    if checkpoint_path.is_file() and not force_download:
        return checkpoint_path

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "The 'huggingface_hub' package is required to download the SYLPH "
            "checkpoint. Install or update the project environment first."
        ) from exc

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        "\nSYLPH pretrained checkpoint was not found locally.\n"
        f"Downloading {HF_REPO_ID}/{HF_FILENAME} from Hugging Face...\n"
        f"Destination: {checkpoint_path}\n"
    )

    downloaded_path = Path(
        hf_hub_download(
            repo_id=HF_REPO_ID,
            repo_type="model",
            filename=HF_FILENAME,
            revision=HF_REVISION,
            local_dir=checkpoint_path.parent,
            force_download=force_download,
        )
    ).resolve()

    if not downloaded_path.is_file():
        raise FileNotFoundError(
            "Hugging Face reported a successful download, but the checkpoint "
            f"was not found at {downloaded_path}."
        )

    if downloaded_path.stat().st_size < MIN_CHECKPOINT_SIZE_BYTES:
        raise RuntimeError(
            "The downloaded checkpoint appears unexpectedly small: "
            f"{downloaded_path.stat().st_size / 1024**2:.2f} MiB."
        )

    print(
        f"Checkpoint ready: {downloaded_path} "
        f"({downloaded_path.stat().st_size / 1024**2:.2f} MiB)"
    )

    return downloaded_path


if __name__ == "__main__":
    get_sylph_checkpoint()
