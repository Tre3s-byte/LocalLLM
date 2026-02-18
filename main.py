import gradio as gr
from llama_cpp import Llama

# Load the model once (outside the chat function for efficiency)
llm = Llama(
    model_path="./models/models--Qwen--Qwen2.5-7B-Instruct-GGUF/snapshots/bb5d59e06d9551d752d08b292a50eb208b07ab1f/qwen2.5-7b-instruct-q5_k_m-00001-of-00002.gguf",
    n_gpu_layers=-1,
    n_ctx=8192,  # Increased context window for larger responses
    n_batch=1024,  # Increased batch size for faster processing
    n_threads=8,
    verbose=False
)

def chat(message, history):
    try:
        # Build messages list from history (Gradio history is a list of message dicts)
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        
        # Add conversation history
        if history:
            for msg in history:
                messages.append(msg)
        
        # Add the current message
        messages.append({"role": "user", "content": message})
        
        # Generate response
        output = llm.create_chat_completion(
            messages=messages,
            max_tokens=1000,  # Increased for larger responses
            temperature=0.7,
            top_p=0.9  # Added for better sampling and potentially faster generation
        )
        
        response = output["choices"][0]["message"]["content"].strip()
        
        # Add user message and response to history
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})
        
        return "", history
    
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": error_msg})
        return "", history

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