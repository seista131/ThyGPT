import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os



filepath="/root/autodl-tmp/TI-RAS_EXP.txt"
loader=UnstructuredFileLoader(filepath)
docs=loader.load()
#打印一下看看，返回的是一个列表，列表中的元素是Document类型
print(docs)
text_splitter=RecursiveCharacterTextSplitter(chunk_size=20,chunk_overlap=10)
docs=text_splitter.split_documents(docs)


embeddings=HuggingFaceEmbeddings(model_name="/root/autodl-tmp/text2vec-large-chinese", model_kwargs={'device': 'cuda'})
#如果之前没有本地的faiss仓库，就把doc读取到向量库后，再把向量库保存到本地
if os.path.exists("/root/autodl-tmp/my_faiss_store.faiss")==False:
    vector_store=FAISS.from_documents(docs,embeddings)
    vector_store.save_local("/root/autodl-tmp/my_faiss_store.faiss")
#如果本地已经有faiss仓库了，说明之前已经保存过了，就直接读取
else:
    vector_store=FAISS.load_local("/root/autodl-tmp/my_faiss_store.faiss",embeddings=embeddings)
#注意！！！！
#如果修改了知识库（knowledge.txt）里的内容
#则需要把原来的 my_faiss_store.faiss 删除后，重新生成向量库


#先做tokenizer
tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/Llama2-chat-13B-Chinese-50W',trust_remote_code=True)
#加载本地基础模型
#low_cpu_mem_usage=True,
#load_in_8bit="load_in_8bit",
base_model = AutoModelForCausalLM.from_pretrained(
        "/root/autodl-tmp/Llama2-chat-13B-Chinese-50W",
        torch_dtype=torch.float16,
        device_map='auto',
        trust_remote_code=True
    )
model=base_model.eval()


query="你是谁？"
docs=vector_store.similarity_search(query)#计算相似度，并把相似度高的chunk放在前面
context=[doc.page_content for doc in docs]#提取chunk的文本内容
print(context)


inputs = tokenizer([f"Human:{prompt}\nAssistant:"], return_tensors="pt")
input_ids = inputs["input_ids"].to('cuda')

generate_input = {
    "input_ids":input_ids,
    "max_new_tokens":1024,
    "do_sample":True,
    "top_k":50,
    "top_p":0.95,
    "temperature":0.3,
    "repetition_penalty":1.3
}
generate_ids  = model.generate(**generate_input)

new_tokens = tokenizer.decode(generate_ids[0], skip_special_tokens=True)
print("new_tokens",new_tokens)