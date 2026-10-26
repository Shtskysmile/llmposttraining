#!/usr/bin/env python3
"""Write a SHA-256 manifest for a local Transformers checkpoint snapshot."""
import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="local Hugging Face Transformers checkpoint directory")
    parser.add_argument("--model-id", default=None, help="optional upstream model identifier")
    parser.add_argument("--revision", default=None, help="upstream revision used for the download")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_dir():
        parser.error(f"checkpoint directory does not exist: {checkpoint}")
    files = sorted(path for path in checkpoint.iterdir() if path.is_file())
    manifest = {
        "model_id": args.model_id,
        "upstream_revision": args.revision,
        "checkpoint_directory": str(checkpoint.resolve()),
        "files": [{"name": path.name, "size_bytes": path.stat().st_size,
                   "sha256": sha256(path)} for path in files],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "files": len(files),
                      "total_bytes": sum(item["size_bytes"] for item in manifest["files"])}, indent=2))


if __name__ == "__main__":
    main()
