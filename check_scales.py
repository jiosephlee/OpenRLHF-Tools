import torch
from safetensors import safe_open
from pathlib import Path
from huggingface_hub import snapshot_download

def main():
    repo_id = "jiosephlee/gpt-oss-20B-NVFP4-packed"
    local_dir = "/tmp/gpt-oss-scale-check"
    
    print(f"Downloading safetensors from {repo_id}...")
    # Only download the safetensors file to save time/bandwidth
    snapshot_download(
        repo_id, 
        local_dir=local_dir, 
        allow_patterns=["*.safetensors"]
    )
    
    model_path = Path(local_dir)
    safetensors_files = list(model_path.glob("*.safetensors"))
    
    print(f"\nFound {len(safetensors_files)} safetensors files.")
    
    found = False
    for f in safetensors_files:
        with safe_open(f, framework="pt", device="cpu") as st:
            for k in st.keys():
                if "scales_2" in k or "scale_2" in k:
                    print(f"{k}: {st.get_tensor(k).shape}")
                    found = True
                    
            # If we found scales in this file, we can probably stop looking
            if found:
                break
                
    if not found:
        print("No scale_2 tensors found in the downloaded safetensors!")

if __name__ == "__main__":
    main()
