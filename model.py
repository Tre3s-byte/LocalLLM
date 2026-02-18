from huggingface_hub import hf_hub_download

fragments = [
    "qwen2.5-7b-instruct-q5_k_m-00001-of-00002.gguf",
    "qwen2.5-7b-instruct-q5_k_m-00002-of-00002.gguf"
]

for fragment in fragments:
    try:
        path = hf_hub_download(
            repo_id="Qwen/Qwen2.5-7B-Instruct-GGUF",
            filename=fragment,
            cache_dir="./models",
            force_download=True,
            local_dir_use_symlinks=False  # fuerza descarga física
        )
        print(f"Downloaded {fragment} to {path}")
    except Exception as e:
        print(f"Error downloading {fragment}: {e}")