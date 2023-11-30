import os
import argparse
import gradio as gr
from main import load_models, cache_path
from os import path

canvas_size = 512

if not path.exists(cache_path):
    os.makedirs(cache_path, exist_ok=True)

with gr.Blocks() as demo:
    infer = load_models()

    with gr.Column():
        with gr.Row():
            with gr.Column():
                s = gr.Slider(label="steps", minimum=4, maximum=8, step=1, value=2, interactive=True)
                c = gr.Slider(label="cfg", minimum=0.0, maximum=3, step=0.1, value=0.0, interactive=True)
            with gr.Column():
                t = gr.Text(label="Prompt", value="A cinematic shot of a baby racoon wearing an intricate italian priest robe.", interactive=True)
                se = gr.Number(label="seed", value=1337, interactive=True)
        with gr.Row(equal_height=True):
            o = gr.Image(width=canvas_size, height=canvas_size)

            def process_image(p, steps, cfg, seed):
                return infer(
                    prompt=p,
                    num_inference_steps=steps,
                    guidance_scale=cfg,
                    seed=int(seed)
                )

            reactive_controls = [t, s, c, se]

            for control in reactive_controls:
                control.change(fn=process_image, inputs=reactive_controls, outputs=o)

            def update_model(model_name):
                global infer
                infer = load_models(model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # If the option python ui.py --share is attached, it will be deployed to Gradio
    parser.add_argument("--share", action="store_true", help="Deploy on Gradio for sharing", default=False)
    args = parser.parse_args()
    demo.launch(share=args.share, server_name='0.0.0.0', server_port=7855)
