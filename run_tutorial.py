## third-party
from transformers import AutoTokenizer
import torch

## own
from src.model import SFR,XMistralForCausalLM
from src.language_modeling.utils import get_retrieval_embeds,XRAG_TOKEN

device = torch.device("cuda:1")
llm_name_or_path = "Hannibal046/xrag-7b"
llm = XMistralForCausalLM.from_pretrained(llm_name_or_path,torch_dtype = torch.bfloat16,low_cpu_mem_usage = True,).to(device).eval()
llm_tokenizer = AutoTokenizer.from_pretrained(llm_name_or_path,add_eos_token=False,use_fast=False,padding_side='left')

## here, XRAG_TOKEN is just a place holder
llm.set_xrag_token_id(llm_tokenizer.convert_tokens_to_ids(XRAG_TOKEN))
print(XRAG_TOKEN)

question = """What is the advantage of the CORY method over single-agent RL method?"""
# documents = [
#     'Alvin and the Chipmunks | " Alvin and the Chipmunks, originally David Seville and the Chipmunks or simply The Chipmunks, are an American animated virtual band created by Ross Bagdasarian for a novelty record in 1958. The group consists of three singing animated anthropomorphic chipmunks named Alvin, Simon, and Theodore. They are managed by their human adoptive father, David ""Dave"" Seville. Bagdasarian provided the group\'s voices sped up to create high-pitched squeaky voices (which wasn\'t entirely new to him, having worked on ""Witch Doctor"" earned the record two Grammy Awards for engineering). ""The Chipmunk Song"" became a number-one single in the United States. After Bagdasarian died in 1972, the characters’ voices were provided by his son Ross Bagdasarian Jr. and the latter\'s wife Janice Karman in the subsequent incarnations of "',
#     "Jamie Lee Curtis |  Jamie Lee Curtis (born November 22, 1958) is an American actress and writer. She is the recipient of several accolades, including a British Academy Film Award, two Golden Globe Awards and a star on the Hollywood Walk of Fame in 1998. Curtis made her film acting debut as Laurie Strode in John Carpenter's horror film Halloween (1978), which established her as a scream queen, and she thereafter appeared in a string of horror films, including The Fog, Prom Night, Terror Train (all 1980) and Roadgames (1981). She reprised the role of Laurie in the sequels Halloween II (1981), Halloween H20: 20 Years Later (1998), Halloween: Resurrection (2002), Halloween (2018), and Halloween Kills (2021). Her filmography is largely characterized by independent film that have been box-office successes, with 8 of her lead-actress credits ",
#     'Sunset Boulevard (musical) | " The American premiere was at the Shubert Theatre in Century City, Los Angeles, California, on 9 December 1993, with Close as Norma and Alan Campbell as Joe. Featured were George Hearn as Max and Judy Kuhn as Betty. Lloyd Webber had reworked both the book and score, tightening the production, better organising the orchestrations, and adding the song ""Every Movie\'s a Circus"". This new production was better received by the critics and was an instant success, running for 369 performances. The Los Angeles production also recorded a new cast album that is well regarded. It is also the only unabridged cast recording of the show, since the original London recording was trimmed by over thirty minutes. A controversy arose with this production after Faye Dunaway was hired to replace Glenn Close. Dunaway went into rehearsals with Rex Smith as Joe and Jon Cypher as Max. Tickets "',
#     'Arthur Balfour |  Balfour was appointed prime minister on 12 July 1902 while the King was recovering from his recent appendicitis operation. Changes to the Cabinet were thus not announced until 9 August, when the King was back in London. The new ministers were received in audience and took their oaths on 11 August.',
#     'Motel 6 | " Beginning in 1986, Motel 6 has advertised through radio commercials featuring the voice of writer and National Public Radio commentator Tom Bodett, with the tagline "We\'ll leave the light on for you." The ads were created by Dallas advertising agency The Richards Group. They feature a tune composed by Tom Faulkner, performed by him on guitar and Milo Deering on fiddle. The first spots were conceived and written by David Fowler. In 1996, the ads won a Clio Award. The campaign itself has won numerous national and international awards and was selected by Advertising Age magazine as one of the Top 100 Advertising Campaigns of the Twentieth Century."',
# ]
documents = [
    "Cooperative multi-agent reinforcement learning (MARL) represents a paradigm shift in the field of artificial intelligence (AI), where multiple autonomous agents coevolve within a complex system, resulting in the emergence of new skills [Foerster, 2018, Yang and Wang, 2020, Oroojlooy and Hajinezhad, 2023, Zang et al., 2023]. Language is an outcome of such multi-agent coevolution. In a society, numerous individuals utilize language for communication. Languages develop through agent interactions and are shaped by societal and cultural influences. As languages progress, they influence and are influenced by these interactions [Cavalli-Sforza and Feldman, 1981, Duéñez-Guzmán et al., 2023]. Inspired by this, fine-tuning an LLM within a cooperative MARL framework might lead to the emergence of superior policies during coevolution.",
    "To verify this hypothesis, we fine-tune the Llama-2-7b-chat model on the grade school math 8K (GSM8K) dataset [Cobbe et al., 2021b] using both PPO and CORY. We measure the KL divergence and the task reward obtained by each policy after convergence. By adjusting the preference, i.e., η in Equation 2, we are able to generate sub-optimal frontiers for both the methods, as illustrated in Figure 2(c). It is important to note that the Y-axis represents the negative KL divergence (larger values indicate better performance). As expected, the sub-optimal frontier achieved by CORY consistently outperforms that of PPO, empirically validating the hypothesis.",
    "Task Setup. To evaluate our method under the subjective reward setting, we select the IMDB Review dataset [Tripathi et al., 2020]. This dataset contains 50K <text,label> pairs, with the training set and the test set each contains 25K pieces of data. The texts in the IMDB dataset are movie reviews, and the labels are the binary sentiment classification labels.",
    "Moreover, it can be observed that the curves of CORY-LLM1 and CORY-LLM2 are very close, indicating that the two LLM agents initially playing different roles finally achieve very similar performance levels at the end of the training. Consistent with the motivation of CORY, both the fine-tuned LLM agents can be used to finish tasks individually, which verifies the effectiveness of the bootstrapped learning and coevolution principles in CORY.",
    "Ablation on Model Size. Our method employs two models during training, with the total parameters trained being doubled in comparison to single-PPO. In order to ablate whether the enhancement of CORY is derived from the expansion of the model parameters, an additional fine-tuning of GPT2-XL (1.5B) with single-PPO is conducted on the IMDB dataset, which has twice the number of parameters as GPT2-Large. The results are presented in Figure 12. While the task reward of the model rapidly reaches its maximum value, the KL penalty part does not exhibit a notable improvement compared to GPT2-Large. The KL divergence continues to increase, leading to the collapse of the distribution."
]

retriever_name_or_path = "Salesforce/SFR-Embedding-Mistral"
retriever = SFR.from_pretrained(retriever_name_or_path,torch_dtype = torch.bfloat16).eval().to(device)
retriever_tokenizer = AutoTokenizer.from_pretrained(retriever_name_or_path)

retriever_input = retriever_tokenizer(documents,max_length=180,padding=True,truncation=True,return_tensors='pt').to(device)
with torch.no_grad():
    doc_embeds = retriever.get_doc_embedding(input_ids=retriever_input.input_ids,attention_mask=retriever_input.attention_mask)
print(doc_embeds.shape)

datastore = (documents,doc_embeds)

retriever_input = retriever_tokenizer(question,max_length=180,padding=True,truncation=True,return_tensors='pt').to(device)
with torch.no_grad():
    query_embed = retriever.get_query_embedding(input_ids=retriever_input.input_ids,attention_mask=retriever_input.attention_mask)
print(query_embed.shape)



_,index = torch.topk(torch.matmul(query_embed,doc_embeds.T),k=1)
top1_doc_index = index[0][0].item()
print(top1_doc_index)

rag_template = """[INST] Refer to the background document and answer the questions:

Background: {document}

Question: {question} [/INST] The answer is:"""

relevant_embedding = datastore[1][top1_doc_index]

## build prompt where XRAG_TOKEN is only a player holder taking up only one token
prompt = rag_template.format_map(dict(question=question,document=XRAG_TOKEN))
print(prompt)


input_ids = llm_tokenizer(prompt,return_tensors='pt').input_ids.to(device)
generated_output = llm.generate(
        input_ids = input_ids,
        do_sample=False,
        max_new_tokens=20,
        pad_token_id=llm_tokenizer.pad_token_id,
        retrieval_embeds = relevant_embedding.unsqueeze(0),
    )
result = llm_tokenizer.batch_decode(generated_output,skip_special_tokens=True)[0]
print(result)