from pastalib.pasta import PASTA

from transformers import LlamaTokenizer, LlamaForCausalLM,LogitsProcessorList,LogitsProcessor,BloomForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Union, List
import torch
def main():
    # Create instances of the classes
    instance_a = PASTA()
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf",cache_dir = '/work/pi_mccallum_umass_edu/aparashar_umass_edu/models/.cache')

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
    head_config = {
    "3": [17, 7, 6, 12, 18], "8": [28, 21, 24], "5": [24, 4], 
    "0": [17], "4": [3], "6": [14], "7": [13], "11": [16], 
}   
    pasta = PASTA(
    model=model,
    tokenizer=tokenizer,
    head_config=head_config, 
    alpha=0.01, # scaling coefficient
    scale_position="exclude", # downweighting unselected tokens
)   
    texts = ["Give the answer to this difficult interrogation","Give the answer to this led zeppelin"]

    inputs, offset_mapping = pasta.inputs_from_batch(texts)
            # User highlights specific input spans
    emphasized_texts = ["difficult interrogation","led zeppelin"]
# PASTA registers the pre_forward_hook to edit attention
    with pasta.apply_steering(
        model=model, 
        strings=texts, 
        substrings=emphasized_texts, 
        model_input=inputs, 
        offsets_mapping=offset_mapping
    ) as steered_model: 
        outputs = steered_model(**inputs)
    # input_ids = tokenizer(input_texts, padding=True, return_tensors="pt").input_ids.to(device) 
    # outputs = model(input_ids)
    probs = torch.log_softmax(outputs.logits, dim=-1).detach()
    probs = probs[:, :-1, :]
    #print(probs.shape)
    print(input_ids)
    input_ids = input_ids[:, 1:]
    # max_score_ids  = torch.topk(probs,10,dim=2)
    # #print(max_score_ids.indices.shape[1])
    # last_tokenid = max_score_ids.indices.shape[1]-1
    # print(tokenizer.decode(max_score_ids.indices[0][last_tokenid]))
    gen_probs = torch.gather(probs, 2, input_ids[:, :, None]).squeeze(-1)
    batch = []  
    for input_sentence, input_probs in zip(input_ids, gen_probs):
        text_sequence = []
        for token, p in zip(input_sentence, input_probs):
            if token not in tokenizer.all_special_ids:
                text_sequence.append((tokenizer.decode(token), p.item()))
        batch.append(text_sequence)
    breakpoint()
    return batch

    # Call functions from ClassA and ClassB
    instance_a.function_a()

if __name__ == "__main__":
    main()