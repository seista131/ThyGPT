import os
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from peft import PeftModel

def setup(args):
    global tokenizer, model, device, vector_store

    embeddings = HuggingFaceEmbeddings(model_name="/root/autodl-tmp/text2vec-large-chinese", model_kwargs={'device': 'cuda'})

    if not os.path.exists("/root/autodl-tmp/my_faiss_store"):
        filepath = "/root/autodl-tmp/TI-RAS_ENG.txt"
        loader = UnstructuredFileLoader(filepath)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=20, chunk_overlap=10)
        docs = text_splitter.split_documents(docs)
        vector_store = FAISS.from_documents(docs, embeddings)
        vector_store.save_local("/root/autodl-tmp/my_faiss_store")
    else:
        vector_store = FAISS.load_local("/root/autodl-tmp/my_faiss_store", embeddings=embeddings)

    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        device = torch.device('cpu')

    if args.tokenizer_path is None:
        args.tokenizer_path = args.lora_model if args.lora_model else args.base_model

    tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_path)

    base_model = LlamaForCausalLM.from_pretrained(
        args.base_model,
        load_in_8bit=args.load_in_8bit,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map='auto',
    )

    model_vocab_size = base_model.get_input_embeddings().weight.size(0)
    tokenzier_vocab_size = len(tokenizer)
    if model_vocab_size != tokenzier_vocab_size:
        base_model.resize_token_embeddings(tokenzier_vocab_size)

    if args.lora_model is not None:
        model = PeftModel.from_pretrained(
            base_model,
            args.lora_model,
            torch_dtype=torch.float16,
            device_map='auto',
        )
    else:
        model = base_model

    if device == torch.device('cpu'):
        model.float()

    model.eval()
