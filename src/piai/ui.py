
import os
import shutil

import gradio as gr
from dotenv import load_dotenv

from principal_investigator import chat_with_principal_investigator

load_dotenv(override=True)


def main():
    def put_message_in_chatbot(message, history):
        return "", history + [{"role": "user", "content": message}]

    theme = gr.themes.Soft(font=["Inter", "system-ui", "sans-serif"])

    with gr.Blocks(title="Quantum Chemistry Lab", theme=theme) as ui:
        gr.Markdown("# 🏢 Quantum Chemistry Lab")

        with gr.Row():
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(label="💬 Lab") 
                message = gr.Textbox(
                    label="Your Research Project",
                    placeholder="Ask anything about quantum chemistry...",
                    show_label=False,
                )

            with gr.Column(scale=1):
                event_html = gr.HTML(
                    label="Research Project Progress",
                    value="Events will appear here",
                    container=True,
                    height=600,
                )

        message.submit(
            put_message_in_chatbot,
            inputs=[message, chatbot],
            outputs=[message, chatbot],
        ).then(
            chat_with_principal_investigator, 
            inputs=[chatbot, event_html], 
            outputs=[chatbot, event_html]
        )

    ui.launch(inbrowser=True, allowed_paths=[".", "artifacts"])


def clean_artifacts():
    if not os.path.exists("artifacts"):
        os.mkdir("artifacts")
    else:
        # Move old artifacts
        items = [i for i in os.listdir("artifacts")]
        old_artifact_dirs = [i for i in os.listdir(".") if i.startswith("artifacts_")]
        if items:
            a = max([0] + [int(i.split("_")[-1]) for i in old_artifact_dirs])
            shutil.move("artifacts", f"artifacts_{a+1}")

if __name__ == "__main__":
    clean_artifacts()
    main()
