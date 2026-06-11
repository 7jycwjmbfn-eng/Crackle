"""Fetch public multi-crack datasets via Mendeley / Zenodo APIs.

Usage:
  python -m scripts.fetch_datasets --list
  python -m scripts.fetch_datasets --get kth_dic_concrete --out ./external_datasets
  python -m scripts.fetch_datasets --all --out ./external_datasets

Notes
-----
- Mendeley and Zenodo expose stable public APIs; both are wired here.
- Some priority datasets (registry section "manual") live in journal
  supplements without an API; the script prints exact retrieval steps.
- Respect licenses: each entry records the license string reported by the
  host; the downloader writes it into <out>/<name>/LICENSE_NOTE.txt.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import urllib.request

REGISTRY: dict[str, dict] = {
    "kth_dic_concrete": {
        "backend": "mendeley",
        "id": "dns97tfdjn",
        "version": 1,
        "desc": "KTH: concrete cracking tracked with DIC, fixed + moving camera "
                "(Sjolander et al.). Check newer versions/companion records for "
                "full image series.",
    },
    "craquelure_paintings": {
        "backend": "zenodo",
        "id": "17862067",
        "desc": "Craquelure (crack network) development in paintings on canvas. "
                "Dense 2D crack networks; topologically the richest real-world "
                "pattern family here.",
    },
    "desiccation_slope": {
        "backend": "zenodo",
        "id": "10199729",
        "desc": "Soil desiccation crack data accompanying a slope-instability "
                "study; verify content before building a loader.",
    },
}

MANUAL: dict[str, str] = {
    "rimkus_rc_ties": "Rimkus & Gribniak, Data in Brief (2017), "
        "doi:10.1016/j.dib.2017.05.038 — 22 RC ties, sequential transverse "
        "cracking, DIC crack development schemes. Download the article "
        "supplement from the DOI landing page.",
    "rc_slab_multitemporal": "arXiv:2411.04620 — 2 m^2 RC slab, 8 load epochs "
        "to failure, multi-temporal crack segmentation. Check the paper/repo "
        "for a data link; availability not guaranteed.",
    "d_cracks_compilation": "D-CRACKS (Scientific Data, 2026), "
        "doi:10.1038/s41597-026-06632-6 — ~1000 desiccation crack images from "
        "41 studies with SQL metadata. Mostly final patterns (limited time "
        "series); follow the Data Availability section.",
}


def _get_json(url: str) -> object:
    req = urllib.request.Request(url, headers={"User-Agent": "crackle-topo-fetch/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "crackle-topo-fetch/1.0"})
    with urllib.request.urlopen(req, timeout=600) as resp, dest.open("wb") as out:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
    print(f"  -> {dest} ({dest.stat().st_size/1e6:.1f} MB)")


def fetch_mendeley(entry: dict, out_dir: Path) -> None:
    url = (f"https://data.mendeley.com/public-api/datasets/{entry['id']}/files"
           f"?folder_id=root&version={entry['version']}")
    files = _get_json(url)
    print(f"  {len(files)} files listed")
    for item in files:
        _download(item["content_details"]["download_url"], out_dir / item["filename"])


def fetch_zenodo(entry: dict, out_dir: Path) -> None:
    record = _get_json(f"https://zenodo.org/api/records/{entry['id']}")
    license_id = record.get("metadata", {}).get("license", {}).get("id", "unknown")
    (out_dir / "LICENSE_NOTE.txt").parent.mkdir(parents=True, exist_ok=True)
    (out_dir / "LICENSE_NOTE.txt").write_text(
        f"zenodo:{entry['id']} license={license_id}\n", encoding="utf-8")
    files = record.get("files", [])
    print(f"  {len(files)} files listed (license: {license_id})")
    for item in files:
        link = item.get("links", {}).get("self")
        if link:
            _download(link, out_dir / item["key"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--get", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("external_datasets"))
    args = parser.parse_args(argv)

    if args.list or (not args.get and not args.all):
        print("API-fetchable:")
        for name, entry in REGISTRY.items():
            print(f"  {name:24s} [{entry['backend']}] {entry['desc']}")
        print("\nManual retrieval:")
        for name, note in MANUAL.items():
            print(f"  {name:24s} {note}")
        return 0

    names = list(REGISTRY) if args.all else [args.get]
    for name in names:
        if name not in REGISTRY:
            print(f"unknown dataset: {name}; see --list"); return 1
        entry = REGISTRY[name]
        out_dir = args.out / name
        print(f"[{name}] backend={entry['backend']}")
        try:
            if entry["backend"] == "mendeley":
                fetch_mendeley(entry, out_dir)
            else:
                fetch_zenodo(entry, out_dir)
        except Exception as exc:  # keep going; report at end
            print(f"  FAILED: {exc}")
    if args.all:
        print("\nManual items still needed:")
        for name, note in MANUAL.items():
            print(f"  {name}: {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
