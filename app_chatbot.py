# -*- coding: utf-8 -*-
import gradio as gr
from src.chatbot import ChatState, ask_chat

STATE = {"chat": ChatState()}

def on_send(user_message, topk, model, chat_history):
    if not user_message:
        return chat_history, [], ""
    text, passages = ask_chat(user_message, STATE["chat"], topk=int(topk), backend="ollama", model=model)
    STATE["chat"].add("user", user_message)
    STATE["chat"].add("assistant", text)
    # update chat UI
    chat_history = (chat_history or []) + [(user_message, text)]
    rows = [[i+1, p["entry_index"], p["chapter_number"], p["narrator"], p["date_iso"], f"{p['score']:.3f}", p["text"][:320]] 
            for i,p in enumerate(passages)]
    return chat_history, rows, ""

def on_clear():
    STATE["chat"] = ChatState()
    return [], []

with gr.Blocks() as demo:
    gr.Markdown("# Dracula Chatbot — Local RAG (Ollama)")
    with gr.Row():
        topk = gr.Slider(1, 8, value=4, step=1, label="Top-k passages")
        model = gr.Textbox(value="gpt-oss:20b", label="Ollama model tag")
    chat = gr.Chatbot(height=380)
    with gr.Row():
        msg = gr.Textbox(placeholder="Ask about Dracula…", scale=4)
        send = gr.Button("Send", variant="primary", scale=1)
        clear = gr.Button("Clear", scale=1)
    sources = gr.Dataframe(headers=["#","entry","chapter","narrator","date","score","preview"], wrap=True)

    send.click(on_send, [msg, topk, model, chat], [chat, sources, msg])
    msg.submit(on_send, [msg, topk, model, chat], [chat, sources, msg])
    clear.click(on_clear, [], [chat, sources])

if __name__ == "__main__":
    demo.launch()
