import gradio as gr
from src.inference import predict
from src.utils import reset_user_input, reset_state

def launch_gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown(">THYGPT基于大语言模型的甲状腺诊疗系统")
        chatbot = gr.Chatbot()
        with gr.Row():
            with gr.Column(scale=4):
                with gr.Column(scale=12):
                    user_input = gr.Textbox(
                        show_label=False,
                        placeholder="Shift + Enter发送消息...",
                        lines=10).style(container=False)
                with gr.Column(min_width=32, scale=1):
                    submitBtn = gr.Button("Submit", variant="primary")
            with gr.Column(scale=1):
                emptyBtn = gr.Button("Clear History")
                max_new_token = gr.Slider(
                    0, 4096, value=512, step=1.0, label="Maximum New Token Length", interactive=True)
                top_p = gr.Slider(0, 1, value=0.9, step=0.01,
                                  label="Top P", interactive=True)
                temperature = gr.Slider(
                    0, 1, value=0.5, step=0.01, label="Temperature", interactive=True)
                top_k = gr.Slider(1, 40, value=40, step=1,
                                  label="Top K", interactive=True)
                do_sample = gr.Checkbox(
                    value=True, label="Do Sample", info="use random sample strategy", interactive=True)
                repetition_penalty = gr.Slider(
                    1.0, 3.0, value=1.1, step=0.1, label="Repetition Penalty", interactive=True)

        params = [user_input, chatbot]
        predict_params = [
            chatbot, max_new_token, top_p, temperature, top_k, do_sample, repetition_penalty]

        submitBtn.click(
            user, params, params, queue=False).then(
            predict, predict_params, chatbot).then(
            lambda: gr.update(interactive=True),
            None, [user_input], queue=False)

        user_input.submit(
            user, params, params, queue=False).then(
            predict, predict_params, chatbot).then(
            lambda: gr.update(interactive=True),
            None, [user_input], queue=False)

        submitBtn.click(reset_user_input, [], [user_input])

        emptyBtn.click(reset_state, outputs=[chatbot], show_progress=True)

    demo.queue().launch(
        share=args.share,
        inbrowser=True,
        server_name='0.0.0.0',
        server_port=args.port)
