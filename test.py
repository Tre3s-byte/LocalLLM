from llama_cpp import Llama

llm = Llama(
    model_path="./models/models--Qwen--Qwen2.5-7B-Instruct-GGUF/snapshots/bb5d59e06d9551d752d08b292a50eb208b07ab1f/qwen2.5-7b-instruct-q5_k_m-00001-of-00002.gguf",
    n_gpu_layers=-1,
    n_ctx=4096,
    n_threads=8,
    verbose=True
)

output = llm("Explain in simple terms how a transformer model works.", max_tokens=200)
print(output["choices"][0]["text"])