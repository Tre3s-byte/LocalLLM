import inspect

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


def _detect_chatbot_format():
    """Pick a stable chatbot history format for the installed Gradio version."""
    try:
        params = inspect.signature(gr.Chatbot).parameters
    except (TypeError, ValueError):
        return "tuples"

    # Newer Gradio versions support/expect message dictionaries.
    if "type" in params:
        return "messages"

    # Older versions use tuple-pair format.
    return "tuples"


CHATBOT_FORMAT = _detect_chatbot_format()


def _coerce_content_to_text(content):
    """Convert Gradio message content variants to plain text for llama.cpp."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (int, float, bool)):
        return str(content)

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if text is None:
                    text = item.get("content")
                parts.append(_coerce_content_to_text(text))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)

    if isinstance(content, dict):
        text = content.get("text")
        if text is None:
            text = content.get("content")
        return _coerce_content_to_text(text)

    return str(content)


def _normalize_history_for_model(history):
    """Return history in OpenAI-style message format for llama.cpp."""
    if not history:
        return []

    normalized = []
    for item in history:
        if isinstance(item, dict) and "role" in item and "content" in item:
            role = item["role"]
            if role in {"user", "assistant", "system"}:
                normalized.append(
                    {"role": role, "content": _coerce_content_to_text(item["content"])}
                )
            continue

        if isinstance(item, (list, tuple)) and len(item) == 2:
            user_msg, assistant_msg = item
            if user_msg is not None:
                normalized.append(
                    {"role": "user", "content": _coerce_content_to_text(user_msg)}
                )
            if assistant_msg is not None:
                normalized.append(
                    {
                        "role": "assistant",
                        "content": _coerce_content_to_text(assistant_msg),
                    }
                )

    return normalized


def _normalize_history_for_ui(history):
    """Return history in format expected by the current Chatbot component."""
    model_messages = _normalize_history_for_model(history)

    if CHATBOT_FORMAT == "messages":
        return model_messages

    pairs = []
    pending_user = None
    for message in model_messages:
        role = message["role"]
        content = message["content"]

        if role == "user":
            if pending_user is not None:
                pairs.append([pending_user, ""])
            pending_user = content
        elif role == "assistant":
            if pending_user is None:
                pairs.append(["", content])
            else:
                pairs.append([pending_user, content])
                pending_user = None

    if pending_user is not None:
        pairs.append([pending_user, ""])

    return pairs


def _append_turn_to_ui_history(history, user_text, response):
    updated_history = _normalize_history_for_ui(history)
    if CHATBOT_FORMAT == "messages":
        updated_history.append({"role": "user", "content": user_text})
        updated_history.append({"role": "assistant", "content": response})
    else:
        updated_history.append([user_text, response])
    return updated_history


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

    return "", _append_turn_to_ui_history(history, user_text, response)


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

    if CHATBOT_FORMAT == "messages":
        chatbot = gr.Chatbot(type="messages")
    else:
        chatbot = gr.Chatbot()

    msg = gr.Textbox(placeholder="Type your message here...", lines=3)
    submit_btn = gr.Button("Submit")
    clear_btn = gr.Button("Clear")

    msg.submit(chat, [msg, chatbot], [msg, chatbot])
    submit_btn.click(chat, [msg, chatbot], [msg, chatbot])
    clear_btn.click(lambda: [], None, chatbot, queue=False)

demo.launch(css=custom_css)
