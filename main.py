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


def _coerce_content_to_text(content):
    """Convert Gradio message content variants to plain text for llama.cpp."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (int, float, bool)):
        return str(content)

    # Gradio message content can be a list in newer versions.
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text is not None:
                    parts.append(str(text))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)

    if isinstance(content, dict):
        text = content.get("text") or content.get("content")
        if text is not None:
            return str(text)

    return str(content)


def _normalize_history_for_model(history):
    """Return history in OpenAI-style message format for llama.cpp."""
    if not history:
        return []

    normalized = []
    for item in history:
        if isinstance(item, dict) and "role" in item and "content" in item:
            normalized.append(
                {
                    "role": item["role"],
                    "content": _coerce_content_to_text(item["content"]),
                }
            )
            continue

        if isinstance(item, (list, tuple)) and len(item) == 2:
            user_msg, assistant_msg = item
            if user_msg:
                normalized.append(
                    {"role": "user", "content": _coerce_content_to_text(user_msg)}
                )
            if assistant_msg:
                normalized.append(
                    {
                        "role": "assistant",
                        "content": _coerce_content_to_text(assistant_msg),
                    }
                )

    return normalized


def chat(message, history):
    user_text = _coerce_content_to_text(message)
    model_history = _normalize_history_for_model(history)
    messages = [SYSTEM_PROMPT, *model_history, {"role": "user", "content": user_text}]

    try:
        output = llm.create_chat_completion(
            messages=messages,
            max_tokens=1000,
            temperature=0.7,
            top_p=0.9,
        )
        response = _coerce_content_to_text(output["choices"][0]["message"]["content"]).strip()
    except Exception as exc:
        response = f"Error: {exc}"

    updated_history = list(history or [])
    updated_history.append((user_text, response))
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

with gr.Blocks() as demo:
    gr.Markdown("## Local LLM Chat")

    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Type your message here...", lines=3)
    submit_btn = gr.Button("Submit")
    clear_btn = gr.Button("Clear")

    msg.submit(chat, [msg, chatbot], [msg, chatbot])
    submit_btn.click(chat, [msg, chatbot], [msg, chatbot])
    clear_btn.click(lambda: [], None, chatbot, queue=False)

demo.launch(css=custom_css)
