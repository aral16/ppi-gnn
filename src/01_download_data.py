"""Download the STRING protein-protein interaction files for the target species."""
import sys
import requests

from config import LINKS_FILE, INFO_FILE, LINKS_URL, INFO_URL


def download(url: str, dest) -> None:
    if dest.exists():
        print(f"[skip] {dest.name} already present ({dest.stat().st_size / 1e6:.1f} MB)")
        return
    print(f"[get ] {url}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        got = 0
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                got += len(chunk)
                if total:
                    print(f"\r       {got / 1e6:6.1f} / {total / 1e6:.1f} MB", end="")
        print()
    print(f"[done] {dest.name}")


if __name__ == "__main__":
    try:
        download(LINKS_URL, LINKS_FILE)
        download(INFO_URL, INFO_FILE)
    except requests.HTTPError as e:
        sys.exit(f"Download failed: {e}")
    print("\nSTRING files ready in data/.")
