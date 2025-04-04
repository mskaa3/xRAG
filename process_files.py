## third-party
from transformers import AutoTokenizer
import torch
from datasets import load_dataset
import pandas as pd
import json
from uuid import uuid4

import numpy as np
from tqdm import tqdm
import os
import shutil
import subprocess
import time

from copy import deepcopy
from datasets import Dataset, DatasetDict, load_dataset


## own
from src.model import SFR,XMistralForCausalLM
from src.language_modeling.utils import get_retrieval_embeds,XRAG_TOKEN

device = torch.device("cuda:1")
llm_name_or_path = "Hannibal046/xrag-7b"
llm = XMistralForCausalLM.from_pretrained(llm_name_or_path,torch_dtype = torch.bfloat16,low_cpu_mem_usage = True,).to(device).eval()
llm_tokenizer = AutoTokenizer.from_pretrained(llm_name_or_path,add_eos_token=False,use_fast=False,padding_side='left')
llm.set_xrag_token_id(llm_tokenizer.convert_tokens_to_ids(XRAG_TOKEN))

retriever_name_or_path = "Salesforce/SFR-Embedding-Mistral"
retriever = SFR.from_pretrained(retriever_name_or_path,torch_dtype = torch.bfloat16).eval().to(device)
retriever_tokenizer = AutoTokenizer.from_pretrained(retriever_name_or_path)

question = """What kind of task IMBD dataset is used for?"""
documents = [
    "Cooperative multi-agent reinforcement learning (MARL) represents a paradigm shift in the field of artificial intelligence (AI), where multiple autonomous agents coevolve within a complex system, resulting in the emergence of new skills [Foerster, 2018, Yang and Wang, 2020, Oroojlooy and Hajinezhad, 2023, Zang et al., 2023]. Language is an outcome of such multi-agent coevolution. In a society, numerous individuals utilize language for communication. Languages develop through agent interactions and are shaped by societal and cultural influences. As languages progress, they influence and are influenced by these interactions [Cavalli-Sforza and Feldman, 1981, Duéñez-Guzmán et al., 2023]. Inspired by this, fine-tuning an LLM within a cooperative MARL framework might lead to the emergence of superior policies during coevolution.",
    "To verify this hypothesis, we fine-tune the Llama-2-7b-chat model on the grade school math 8K (GSM8K) dataset [Cobbe et al., 2021b] using both PPO and CORY. We measure the KL divergence and the task reward obtained by each policy after convergence. By adjusting the preference, i.e., η in Equation 2, we are able to generate sub-optimal frontiers for both the methods, as illustrated in Figure 2(c). It is important to note that the Y-axis represents the negative KL divergence (larger values indicate better performance). As expected, the sub-optimal frontier achieved by CORY consistently outperforms that of PPO, empirically validating the hypothesis.",
    "Task Setup. To evaluate our method under the subjective reward setting, we select the IMDB Review dataset [Tripathi et al., 2020]. This dataset contains 50K <text,label> pairs, with the training set and the test set each contains 25K pieces of data. The texts in the IMDB dataset are movie reviews, and the labels are the binary sentiment classification labels.",
    "Moreover, it can be observed that the curves of CORY-LLM1 and CORY-LLM2 are very close, indicating that the two LLM agents initially playing different roles finally achieve very similar performance levels at the end of the training. Consistent with the motivation of CORY, both the fine-tuned LLM agents can be used to finish tasks individually, which verifies the effectiveness of the bootstrapped learning and coevolution principles in CORY.",
    "Ablation on Model Size. Our method employs two models during training, with the total parameters trained being doubled in comparison to single-PPO. In order to ablate whether the enhancement of CORY is derived from the expansion of the model parameters, an additional fine-tuning of GPT2-XL (1.5B) with single-PPO is conducted on the IMDB dataset, which has twice the number of parameters as GPT2-Large. The results are presented in Figure 12. While the task reward of the model rapidly reaches its maximum value, the KL penalty part does not exhibit a notable improvement compared to GPT2-Large. The KL divergence continues to increase, leading to the collapse of the distribution."
]



rag_template = """[INST] Refer to the background document and answer the questions:

Background: {document}

Question: {question} [/INST] The answer is:"""





def is_rclone_installed():
    return shutil.which('rclone') is not None


def upload(source: str, destination: str, verbose: bool = False) -> None:
    if not is_rclone_installed():
        raise RuntimeError('rclone is not installed')

    rclone_remote = os.getenv('RCLONE_REMOTE_CONFIG', None)

    if rclone_remote is None:
        raise RuntimeError('RCLONE_REMOTE_CONFIG is not set')

    flags = "--progress" if verbose else ""

    if os.path.isfile(source):
        destination = os.path.join(destination, os.path.basename(source))

        # When uploading a single file, I get 'Access Denied' errors.
        # Adding this flag solves that problem. I guess I don't have some permissions connected to bucket checks.
        # ~Bartosz Żuk 26.06.2024
        flags = f'{flags} --s3-no-check-bucket'

    start = time.time()
    command = f'rclone copyto {source} {rclone_remote}:{destination} {flags}'

    subprocess.run(command, shell=True, check=True)

    print(f'Uploaded {source} to {destination} in {time.time() - start:.2f} seconds')


def download(source: str, destination: str = None, verbose: bool = False, exclude: str = None) -> str:
    if not is_rclone_installed():
        raise RuntimeError('rclone is not installed')

    rclone_remote = os.getenv('RCLONE_REMOTE_CONFIG', None)

    if rclone_remote is None:
        raise RuntimeError('RCLONE_REMOTE_CONFIG is not set')

    if destination is None:
        destination = os.getenv('TMPDIR', os.path.expanduser('~'))
        destination = os.path.join(destination, os.path.basename(source))

    flags = '--progress' if verbose else ''
    flags = f'{flags} --exclude={exclude}' if exclude else flags

    start = time.time()
    command = f'rclone copyto {rclone_remote}:{source} {destination} {flags}'

    subprocess.run(command, shell=True, check=True)
    print(f'Downloaded {source} to {destination} in {time.time() - start:.2f} seconds')

    return destination


def process(
    tokenizer,
    ds,
    max_tokens,
    retriever_max_length,
    sampling_params,
    name,
):
    df = ds.to_pandas()
    documents = df["text"]
    questions = df["question"]

    retriever_input = retriever_tokenizer(documents,max_length=retriever_max_length,padding=True,truncation=True,return_tensors='pt').to(device)
    with torch.no_grad():
        doc_embeds = retriever.get_doc_embedding(input_ids=retriever_input.input_ids,attention_mask=retriever_input.attention_mask)

    
    prompts = [rag_template.format_map({"question": q, "document": XRAG_TOKEN}) for q in questions]


    # for idx, prompt in enumerate(prompts):
    #     input_ids = llm_tokenizer(prompt,return_tensors='pt').input_ids.to(device)
    input_ids = llm_tokenizer(prompts, padding=True, truncation=True, return_tensors='pt').input_ids.to(device)

    generated_outputs = llm.generate(
        input_ids=input_ids,  
        do_sample=False,
        max_new_tokens=max_tokens,
        pad_token_id=llm_tokenizer.pad_token_id,
        retrieval_embeds=doc_embeds,  
    )
    results = llm_tokenizer.batch_decode(generated_outputs, skip_special_tokens=True)
    df["generated_text"] = results
    ds = Dataset.from_pandas(df)


    print(f"saving to results/{name}.jsonl")
    with open(f"results/{name}.jsonl", "w", encoding="utf-8") as outfile:
        for idx in tqdm(range(len(ds))):
            json.dump(ds[idx], outfile, ensure_ascii=False,default=json_serial)
            outfile.write("\n")



def run(self,
        data: str = "data.json",
        output: str = "output",
        max_tokens: int = 1024,
        retriever_max_length: int = 8000,
        temperature: float = 0.5,
        ):

        if "s3" in data:
        data = download(data, verbose=True)
        ds = load_dataset('json', data_files={
            'train': f'{data}/data.jsonl',
        })
        else:
            ds = load_dataset(data)
        ds = DatasetDict({
            "train": ds['train'],
        })
        os.makedirs("results", exist_ok=True)

        process(model, tokenizer, ds['train'], max_tokens, retriever_max_length, sampling_params, "train")
        print("uploading")
        upload("results", output)




class Main:
    def __init__(self) -> None:
            tqdm.pandas()

    def run(
        self,
        data: str = "data.json",
        output: str = "output",
        max_tokens: int = 1024,
        retriever_max_length: int = 8000,
        temperature: float = 0.5,
    ):
        run(
            data=data,
            output=output,
            max_tokens=max_tokens,
            temperature=temperature
        )





if __name__=="__main__":
    CLI(Main)