import gradio as gr
from llama_cpp import Llama

# Load the model once (outside the chat function for efficiency)
llm = Llama(
    model_path="./models/models--Qwen--Qwen2.5-7B-Instruct-GGUF/snapshots/bb5d59e06d9551d752d08b292a50eb208b07ab1f/qwen2.5-7b-instruct-q5_k_m-00001-of-00002.gguf",
    n_gpu_layers=-1,
    n_ctx=8192,
    n_batch=1024,
    n_threads=8,
    verbose=False,
)

SYSTEM_PROMPT = {"role": "system", "content": "You are a helpful assistant."}


def _normalize_history(history):
    """Return Gradio history in OpenAI-style message format.

    Supports both Chatbot tuple format: [[user, assistant], ...]
    and message format: [{"role": ..., "content": ...}, ...].
    """
    if not history:
        return []

    normalized = []
    for item in history:
        if isinstance(item, dict) and "role" in item and "content" in item:
            normalized.append({"role": item["role"], "content": item["content"]})
            continue

        if isinstance(item, (list, tuple)) and len(item) == 2:
            user_msg, assistant_msg = item
            if user_msg:
                normalized.append({"role": "user", "content": user_msg})
            if assistant_msg:
                normalized.append({"role": "assistant", "content": assistant_msg})

    return normalized


def chat(message, history):
    history = _normalize_history(history)

    messages = [SYSTEM_PROMPT, *history, {"role": "user", "content": message}]

    try:
        output = llm.create_chat_completion(
            messages=messages,
            max_tokens=1000,
            temperature=0.7,
            top_p=0.9,
        )
        response = output["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        response = f"Error: {exc}"

    updated_history = [*history, {"role": "user", "content": message}, {"role": "assistant", "content": response}]
    return "", updated_history


custom_css = """
body {
    background-color: #212121;
}

.gradio-container {
    background-color: #212121 !important;
}

textarea {
    background-color: #303030 !important;
    color: #ffffff !important;
}

.message.user {
    background-color: #303030 !important;
    color: #ffffff !important;
}

.message.bot {
    background-color: #303030 !important;
    color: #ffffff !important;
}
"""

with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("## Local LLM Chat")

    chatbot = gr.Chatbot(type="messages")
    msg = gr.Textbox(placeholder="Type your message here...", lines=3)
    submit_btn = gr.Button("Submit")
    clear_btn = gr.Button("Clear")

    msg.submit(chat, [msg, chatbot], [msg, chatbot])
    submit_btn.click(chat, [msg, chatbot], [msg, chatbot])
    clear_btn.click(lambda: [], None, chatbot, queue=False)

demo.launch()
