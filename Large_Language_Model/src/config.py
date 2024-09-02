import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--base_model',
        default=None,
        type=str,
        required=True,
        help='Base model path')
    parser.add_argument('--lora_model', default=None, type=str,
                        help="If None, perform inference on the base model")
    parser.add_argument(
        '--tokenizer_path',
        default=None,
        type=str,
        help='If None, lora model path or base model path will be used')
    parser.add_argument(
        '--gpus',
        default="0",
        type=str,
        help='If None, cuda:0 will be used. Inference using multi-cards: --gpus=0,1,... ')
    parser.add_argument('--share', default=True, help='Share gradio domain name')
    parser.add_argument('--port', default=19324, type=int, help='Port of gradio demo')
    parser.add_argument(
        '--max_memory',
        default=256,
        type=int,
        help='Maximum input prompt length, if exceeded model will receive prompt[-max_memory:]')
    parser.add_argument(
        '--load_in_8bit',
        action='store_true',
        help='Use 8 bit quantified model')
    parser.add_argument(
        '--only_cpu',
        action='store_true',
        help='Only use CPU for inference')
    parser.add_argument(
        '--alpha',
        type=str,
        default="1.0",
        help="The scaling factor of NTK method, can be a float or 'auto'. ")
    args = parser.parse_args()
    if args.only_cpu is True:
        args.gpus = ""
    return args
