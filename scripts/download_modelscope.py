#!/usr/bin/env python3
"""Download a full Transformers checkpoint from the ModelScope mirror."""
import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-Math-7B-Instruct")
    p.add_argument("--revision", default="master")
    p.add_argument("--output", default="/root/.cache/modelscope/Qwen2.5-Math-7B-Instruct")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    if args.workers < 1:
        p.error("--workers must be positive")
    from modelscope import snapshot_download
    out = Path(args.output).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    result = snapshot_download(model_id=args.model, revision=args.revision,
                              local_dir=str(out), max_workers=args.workers)
    print(f"ModelScope snapshot ready: {result}")


if __name__ == "__main__":
    main()
