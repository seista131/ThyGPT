import torch
from src.setup import setup, tokenizer, model, device, vector_store
from src.utils import clear_torch_cache, generate_prompt
from transformers import LlamaTokenizer

@torch.no_grad()
def predict(
    history,
    max_new_tokens=128,
    top_p=0.75,
    temperature=0.1,
    top_k=40,
    do_sample=True,
    repetition_penalty=1.0
):
    history[-1][1] = ""

    docs = vector_store.similarity_search(history[-1][0])
    context = [doc.page_content for doc in docs]
    input = f"### Instruction:{history[-1][0]} ### Response:{history[-1][1]}"
    prompt = generate_prompt(input, "".join(context))

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    generate_params = {
        'input_ids': input_ids,
        'max_new_tokens': max_new_tokens,
        'top_p': top_p,
        'temperature': temperature,
        'top_k': top_k,
        'do_sample': do_sample,
        'repetition_penalty': repetition_penalty,
    }

    def generate_with_callback(callback=None, **kwargs):
        if 'stopping_criteria' in kwargs:
            kwargs['stopping_criteria'].append(Stream(callback_func=callback))
        else:
            kwargs['stopping_criteria'] = [Stream(callback_func=callback)]
        clear_torch_cache()
        with torch.no_grad():
            model.generate(**kwargs)

    def generate_with_streaming(**kwargs):
        return Iteratorize(generate_with_callback, kwargs, callback=None)

    with generate_with_streaming(**generate_params) as generator:
        for output in generator:
            next_token_ids = output[len(input_ids[0]):]
            if next_token_ids[0] == tokenizer.eos_token_id:
                break
            new_tokens = tokenizer.decode(
                next_token_ids, skip_special_tokens=True)
            if isinstance(tokenizer, LlamaTokenizer) and len(next_token_ids) > 0:
                if tokenizer.convert_ids_to_tokens(int(next_token_ids[0])).startswith('▁'):
                    new_tokens = ' ' + new_tokens

            history[-1][1] = new_tokens
            yield history
            if len(next_token_ids) >= max_new_tokens:
                break
