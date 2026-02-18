import gradio as gr
import requests

MODEL = "qwen2.5:7b"

def chat(prompt, history):
    reponse = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model" : MODEL,
            "prompt": prompt,
            "stream": False
        }
    )
    result = response.json()["response"]
    history.append((prompt, result))
    return "", history

custom_css = """
body {
    background-color: #212121;}

.gradio-container {
    background-color: #212121 !important;}

textarea{
bakground-color: #303030 !important;
color: #ffffff !important;}

.message.bot{
background-color: #fcfcfc !important;
color: #000000 !important;}
"""

with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("## Local LLM")

    chatbot = gr.Chatbot()
    prompt = gr.Textbox(placerholder= "Dime", lines=3)

    prompt.submit(chat, [prompt, chatbot],[prompt, chatbot])

demo.launch()