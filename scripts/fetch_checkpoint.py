#!/usr/bin/env python
"""Download the trained detector into artifacts/.

    uv run python scripts/fetch_checkpoint.py

The file is 234 MB, too large for a git repository, so it is published as a release asset on this
repo. It holds only the weights we trained, the Dinomaly bottleneck and decoder. The frozen DINOv2
encoder is not in there: timm downloads that itself the first time the model is constructed.

Override the source if you host it elsewhere:

    CKPT_URL=https://example.com/detector.pt python scripts/fetch_checkpoint.py

Publishing a new one, for whoever maintains this:

    gh release create v1.0 artifacts/detector.pt --title "v1.0" --notes "Dinomaly detector, VisA pcb1-4"
"""
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "artifacts/detector.pt"

RELEASE_URL = ("https://github.com/nareshmlx/Dinomaly-Anomaly-Detection"
               "/releases/download/v1.0/detector.pt")
CKPT_URL = os.environ.get("CKPT_URL", RELEASE_URL)
EXPECTED_MB = 234


def from_url(url):
    import urllib.request
    print(f"downloading {url}")
    tmp = DEST.with_suffix(".part")
    try:
        with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        tmp.rename(DEST)
    except BaseException:
        tmp.unlink(missing_ok=True)   # never leave a half-file that looks like a finished download
        raise


def main():
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists():
        mb = DEST.stat().st_size / 1048576
        print(f"artifacts/detector.pt already here, {mb:.1f} MB. Delete it to re-download.")
        return

    try:
        from_url(CKPT_URL)
    except Exception as e:
        sys.exit(f"could not download {CKPT_URL}\n  {type(e).__name__}: {e}\n\n"
                 "  If the release is not published yet, point at any direct download:\n"
                 "      CKPT_URL=https://.../detector.pt python scripts/fetch_checkpoint.py\n")

    mb = DEST.stat().st_size / 1048576
    print(f"wrote artifacts/detector.pt, {mb:.1f} MB")
    if abs(mb - EXPECTED_MB) > 20:
        print(f"  warning: expected about {EXPECTED_MB} MB. Check that the download completed.")


if __name__ == "__main__":
    main()
