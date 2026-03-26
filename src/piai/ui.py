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
                # CRITICAL: Add type="messages" to support the dict format
                chatbot = gr.Chatbot(label="💬 Lab") 
                message = gr.Textbox(
                    label="Your Research Project",
                    placeholder="Ask anything about quantum chemistry...",
                    show_label=False,
                )

            with gr.Column(scale=1):
                # event_markdown = gr.Markdown(
                event_markdown = gr.HTML(
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
            inputs=[chatbot], 
            outputs=[chatbot, event_markdown]
        )

    ui.launch(inbrowser=True)

if __name__ == "__main__":
    main()
