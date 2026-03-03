#!/usr/bin/env python3
"""Download and inspect key names/shapes in an NVFP4 checkpoint from HF Hub.

Usage:
  python scripts/inspect_nvfp4_checkpoint.py \
      --repo_id jiosephlee/gpt-oss-20B-NVFP4-packed-clean \
      --filter experts   # only print keys containing this substring
"""
import argparse
import os
from huggingface_hub import snapshot_download
from safetensors import safe_open


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", required=True)
    parser.add_argument("--local_dir", default="/vast/projects/myatskar/design-documents/hf_home/inspect-tmp")
    parser.add_argument("--filter", default="", help="Only show keys containing this substring")
    args = parser.parse_args()

    print(f"Downloading {args.repo_id} ...")
    local = snapshot_download(repo_id=args.repo_id, local_dir=args.local_dir, ignore_patterns=["*.bin"])
    print(f"Downloaded to: {local}\n")

    shard_files = sorted(f for f in os.listdir(local) if f.endswith(".safetensors"))
    print(f"Found {len(shard_files)} shard(s)\n")

    all_keys = {}
    for fname in shard_files:
        path = os.path.join(local, fname)
        with safe_open(path, framework="pt") as sf:
            for key in sf.keys():
                if args.filter and args.filter not in key:
                    continue
                t = sf.get_tensor(key)
                all_keys[key] = (t.shape, t.dtype)

    # Print grouped: layer 0 first to show naming convention, then summarize
    print(f"{'Key':<80} {'Shape':<25} {'Dtype'}")
    print("-" * 120)
    layer0 = {k: v for k, v in all_keys.items() if "layers.0" in k}
    rest_sample = {k: v for k, v in all_keys.items() if "layers.0" not in k}

    for k, (shape, dtype) in sorted(layer0.items()):
        print(f"{k:<80} {str(list(shape)):<25} {dtype}")

    if rest_sample:
        print(f"\n... ({len(rest_sample)} more keys from other layers, same pattern)")

    print(f"\nTotal matching keys: {len(all_keys)}")

    import shutil
    shutil.rmtree(args.local_dir)
    print(f"Cleaned up {args.local_dir}")


if __name__ == "__main__":
    main()
