"""Download and safely extract checksum-pinned EuroSAT RGB v2."""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath

ROOT = Path(__file__).resolve().parents[1]
URL = "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1"
ARCHIVE = ROOT / "data" / "raw" / "EuroSAT_RGB.zip"
PROCESSED = ROOT / "data" / "processed"
EXTRACTED = PROCESSED / "EuroSAT_RGB"
EXPECTED_IMAGES = 27_000
EXPECTED_MD5 = "f46e308c4d50d4bf32fedad2d3d62f3b"
EXPECTED_SHA256 = "b4f5b234ecb7d7ff9c6cddb046543b4717c53fd6e9815be6c0e80cc614f51b90"


def _digests(path: Path) -> tuple[str, str]:
    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            md5.update(block)
            sha256.update(block)
    return md5.hexdigest(), sha256.hexdigest()


def _download() -> None:
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    if ARCHIVE.exists():
        md5, sha256 = _digests(ARCHIVE)
        if (md5, sha256) == (EXPECTED_MD5, EXPECTED_SHA256):
            print(f"Verified existing EuroSAT archive: {ARCHIVE.relative_to(ROOT)}")
            return
        raise ValueError("Existing EuroSAT archive does not match the pinned checksums")
    with tempfile.NamedTemporaryFile(
        prefix="eurosat-", suffix=".part", dir=ARCHIVE.parent, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        request = urllib.request.Request(
            URL, headers={"User-Agent": "eurosat-landcover-vision/1.0"}
        )
        with (
            urllib.request.urlopen(request, timeout=60) as response,
            temporary_path.open("wb") as output,
        ):
            shutil.copyfileobj(response, output, length=1024 * 1024)
        md5, sha256 = _digests(temporary_path)
        if (md5, sha256) != (EXPECTED_MD5, EXPECTED_SHA256):
            raise ValueError(f"Downloaded checksum mismatch: md5={md5}, sha256={sha256}")
        os.replace(temporary_path, ARCHIVE)
    finally:
        temporary_path.unlink(missing_ok=True)
    print(f"Downloaded and verified EuroSAT RGB v2 ({EXPECTED_SHA256})")


def _safe_member_path(name: str, root: Path) -> Path:
    posix_name = PurePosixPath(name)
    windows_name = PureWindowsPath(name)
    if posix_name.is_absolute() or windows_name.is_absolute() or ".." in posix_name.parts:
        raise ValueError(f"Unsafe archive path: {name}")
    resolved_root = root.resolve()
    resolved_target = (root / Path(*posix_name.parts)).resolve()
    if not resolved_target.is_relative_to(resolved_root):
        raise ValueError(f"Archive member escaped its extraction root: {name}")
    return resolved_target


def _extract() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    if EXTRACTED.is_dir() and len(list(EXTRACTED.rglob("*.jpg"))) == EXPECTED_IMAGES:
        print(f"Verified extracted images: {EXTRACTED.relative_to(ROOT)}")
        return
    with zipfile.ZipFile(ARCHIVE) as archive:
        if archive.testzip() is not None:
            raise ValueError("EuroSAT archive failed CRC validation")
        image_members = [
            member for member in archive.infolist() if member.filename.lower().endswith(".jpg")
        ]
        if len(image_members) != EXPECTED_IMAGES:
            raise ValueError(
                f"Expected {EXPECTED_IMAGES:,} JPEGs in archive; got {len(image_members):,}"
            )
        for member in archive.infolist():
            _safe_member_path(member.filename, PROCESSED)
            member_mode = member.external_attr >> 16
            if stat.S_ISLNK(member_mode):
                raise ValueError(f"Symbolic link found in dataset archive: {member.filename}")
        if EXTRACTED.exists():
            shutil.rmtree(EXTRACTED)
        with tempfile.TemporaryDirectory(prefix="eurosat-extract-", dir=PROCESSED) as temp_dir:
            temp_root = Path(temp_dir)
            archive.extractall(temp_root)
            staged = temp_root / "EuroSAT_RGB"
            if not staged.is_dir() or len(list(staged.rglob("*.jpg"))) != EXPECTED_IMAGES:
                raise ValueError("Extracted EuroSAT image folder is incomplete")
            shutil.move(str(staged), str(EXTRACTED))
    print(f"Extracted {EXPECTED_IMAGES:,} EuroSAT RGB images")


def main() -> None:
    _download()
    _extract()


if __name__ == "__main__":
    main()
