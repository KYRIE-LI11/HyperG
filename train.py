import os
import logging
import torch
import torch.nn as nn
import transformers
from transformers import AutoTokenizer,AutoTokenizer, AutoModelForCausalLM, HfArgumentParser, AutoConfig, TrainingArguments, Trainer
from transformers.optimization import AdamW
from torch.optim import Adam
from peft import get_peft_model, LoraConfig, TaskType, get_peft_model_state_dict
import pyarrow as pa
from model import Encoder
from collections import OrderedDict
import re
from dataclasses import dataclass, field
from typing import Optional
from datasets import Dataset
from torch.utils.data import DataLoader
from data import BipartiteData
from tqdm import tqdm
from data_utils import clean_cell_value
from safetensors import safe_open
import random
from torch_geometric.data.batch import Batch
from torch.optim.lr_scheduler import StepLR
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer


CAP_TAG = "<caption>"
HEADER_TAG = "<header>"
ROW_TAG = "<row>"
ROW_DESCRIPTION_TAG = "<row_description>"
COL_DESCRIPTION_TAG = "<col_description>"
NEG_CLAIM_TAG = "<neg_claim>"
POS_CLAIM_TAG = "<pos_claim>"

MISSING_CAP_TAG = '[TAB]'
MISSING_CELL_TAG = "[CELL]"
MISSING_HEADER_TAG = "[HEAD]"
huggingface_token = ""#set private token

def set_random_seed(seed):
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed_all(seed)  # GPU
    random.seed(seed)

set_random_seed(520)


@dataclass
class DataArguments:
    """
    Arguments pertaining to which config/tokenizer we are going use.
    """
    task_type: str = field(
        default= 'tfv',
        metadata={
            "help": "tfv, tqa"
        },
    )

    pretrain_hyperg: bool = field(
        default= False,
        metadata={
            "help": "是否需要加载已经与训练好的hyperg"
        },
    )

    tokenizer_config_type: str = field(
        default='bert-base-uncased',
        metadata={
            "help": "bert-base-cased, bert-base-uncased etc"
        },
    )
    max_token_length: int = field(
        default=64,
        metadata={
            "help": "The maximum total input token length for cell/caption/header after tokenization. Sequences longer "
                    "than this will be truncated."
        },
    )
    max_row_length: int = field(
        default=30,
        metadata={
            "help": "The maximum total input rows for a table"
        },
    )
    max_column_length: int = field(
        default=20,
        metadata={
            "help": "The maximum total input columns for a table"
        },
    )

    dataset_path: str = field(
        default='./data/pretrain/fine_tune_claim_arrow/dataset.arrow',
        metadata={
            "help": "The path of training data"
        },
    )

    save_path: str = field(
        default='/hpc2hdd/JH_DATA/share/zrao538/PrivateShareGroup/zrao538_NLPGroup/hanqianli/hanqianli/WWW25/tfv/test_unpretrained_ours_lora',
        metadata={
            "help": "The path to save fine tune checkpoint, encoder,projector,lora"
        },
    )

    pretrain_hyperg_ckpt: str = field(
        default=None,#'/hpc2hdd/JH_DATA/share/zrao538/PrivateShareGroup/zrao538_NLPGroup/hanqianli/hanqianli/WWW25/tfv/test_pretrained_ours_lora/pretrain_ckpt/epoch=46-step=5687.ckpt',#'/hpc2hdd/JH_DATA/share/zrao538/PrivateShareGroup/zrao538_NLPGroup/huangsirui/WWW25/nohup_logs/test_pretrained_ours_lora/pretrain_ckpt/last.ckpt',
        metadata={
            "help": "The checkpoint path of pretrain hyperg model"
        },
    )

    llm_name: str = field(
        default='meta-llama/Llama-2-7b-chat-hf',#'/hpc2hdd/JH_DATA/share/zrao538/PrivateShareGroup/zrao538_NLPGroup/yggu/huggingface/google/gemma-2-2b-it',#meta-llama/Meta-Llama-3-8B-Instruct
        metadata={
            "help": "The name of llm(llama,etc)"
        },

    
    )
    
    def __post_init__(self):
        if self.tokenizer_config_type not in ["bert-base-cased", "bert-base-uncased"]:
            raise ValueError(
                f"The model type should be bert-base-(un)cased. The current value is {self.tokenizer_config_type}."
            )

@dataclass
class OptimizerConfig:
    step_size : int = field(
        default= 200,
        metadata={
            "help": "设定每多少步降低学习率"
        },
    )

    gamma : float = field(
        default= 0.8,
        metadata={
            "help": "学习率衰减倍数"
        },
    )


    batch_size: int = field(
        default= 1,
        metadata={
            "help": "The batch size of training"
        },
    )

    learning_rate: float = field(
        default= 1e-4,
        metadata={
            "help": "The learing rate of training"
        },
    )

    scale: float = field(
        default= 20,
        metadata={
            "help": "for scaling the learning rate for hyperg"
        },
    )
    
    epochs: int = field(
        default= 4,
        metadata={
            "help": "The epoch of training"
        },
    )

    accumulation_steps: int = field(
        default= 16,
        metadata={
            "help": "The accumulation steps of training"
        },
    )

class PolynomialLR(_LRScheduler):
    r"""
    Set the learning rate for each parameter group using a polynomial defined as: `lr = base_lr * (1 - t / T) ^ (power)`,
    where `t` is the current epoch and `T` is the maximum number of epochs.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int = 0,
        steps: int = 100000,
        power: float = 1.,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.steps = steps
        self.power = power
        super(PolynomialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = max(self.last_epoch, 1)
        if epoch <= self.warmup_steps:
            return [epoch / self.warmup_steps * lr for lr in self.base_lrs]
        t, T = (epoch - self.warmup_steps), (self.steps - self.warmup_steps)
        return [lr * (1 - t / T) ** self.power for lr in self.base_lrs]

def LinearLR(optimizer: Optimizer, warmup_steps: int = 0, steps: int = 100000, last_epoch: int = -1):
    return PolynomialLR(optimizer, warmup_steps, steps, 1, last_epoch)



class HyperG(nn.Module):
    def __init__(
        self,
        data_args = None,
        llm_model_path = None,
        hyperg_encoder = None,
        bert_tokenizer = None,
        use_lora = None,
        pretrain_hyperg = None,
        lora_config = None,

    ):

        super(HyperG, self).__init__()
        self.data_args = data_args
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_path, token=huggingface_token)
        self.llm_tokenizer.add_special_tokens(
            {'pad_token': '[PAD]',
             'unk_token': '[UNK]'
             }    
        )
        self.unk_ = self.llm_tokenizer.unk_token
        prompt_prefix = "A markdown table with descriptions of its rows and columns is given below:"
        self.len_prefix = (self.llm_tokenizer.encode(prompt_prefix,add_special_tokens = True, return_tensors='pt')).shape[1]
        config = transformers.AutoConfig.from_pretrained(llm_model_path)
        config._attn_implementation_internal = "eager"
        self.llm_model = AutoModelForCausalLM.from_pretrained(llm_model_path, token=huggingface_token,torch_dtype=torch.bfloat16, attn_implementation = "eager", device_map = 'auto',  config =config)
        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))
        self.device = self.llm_model.device
        self.hyperg_encoder = hyperg_encoder
        self.hyperg_encoder.to(self.device)
        self.bert_tokenizer = bert_tokenizer
        ###freeze the llama model
        for name, param in self.llm_model.named_parameters():
                param.requires_grad = False

        if use_lora:
            self.llm_model = get_peft_model(self.llm_model, lora_config)
            self.llm_model.enable_input_require_grads()


        #self.llama_model.print_trainable_parameters()#
        #trainable params: 4,718,592 || all params: 8,035,037,184 || trainable%: 0.0587
        #load hypergraph encoder
        if pretrain_hyperg:# load pretrained hypergraph encoder
            state_dict = torch.load(open(data_args.pretrain_hyperg_ckpt, 'rb'))['state_dict']
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith('model.'):
                    new_state_dict[k[6:]] = v  # 去掉 "module." 前缀
                else:
                    new_state_dict[k] = v
            self.hyperg_encoder.load_state_dict(new_state_dict, strict = False)

        llm_embedding_layer = self.llm_model.get_input_embeddings()
        llm_embedding_dim = llm_embedding_layer.embedding_dim #4096
        hyperg_encoder_dim = 768 #hyperg输出的维度
        hidden_layer_dim = 2048
        self.proj = nn.Sequential(
                nn.Linear(hyperg_encoder_dim , hidden_layer_dim),
                nn.ReLU(),
                nn.Linear(hidden_layer_dim, llm_embedding_dim) #hidden_layer_dim随便设置的
            )
        self.proj.to(self.device)#projector将hypergraph输出embedding与llama的embedding进行对齐
        #self.llama_proj.load_state_dict(torch.load("/hpc2hdd/JH_DATA/share/zrao538/PrivateShareGroup/zrao538_NLPGroup/huangsirui/WWW25/hyperg/baseline_ckpt/proj/projector_weights_step_10.pt"))
    
    def llama_process_prompt(self, input_list, response_list:list):
        input_ids = []
        labels = []
        for i in range(len(response_list)):
            prompt = [
                {"role": "user", "content": input_list[i]}
            ]
            prompt=self.llm_tokenizer.apply_chat_template(prompt,add_generation_prompt=True,tokenize=False)
            prompt_tokenized = self.llm_tokenizer(prompt)#, return_tensors="pt").to(self.device)
            input = [
                {"role": "user", "content": input_list[i]},
                {"role": "assistant", "content": response_list[i]}
            ]
            input=self.llm_tokenizer.apply_chat_template(input,add_generation_prompt=False,tokenize=False)
            input_tokenized = self.llm_tokenizer(input)#,return_tensors="pt").to(self.device)

            label = input_tokenized['input_ids'].copy()
            label[:len(prompt_tokenized['input_ids'])] = [-100] * len(prompt_tokenized['input_ids'])

            input_ids.append(input_tokenized['input_ids'])
            labels.append(label)
            
        input_ids, attention_mask = self.padding(input_ids)
        labels, _ = self.padding(labels)
        return input_ids, attention_mask, labels

    def gemma_process_prompt(self, input_list, response_list:list):
        input_ids = []
        labels = []
        for i in range(len(response_list)):
            prompt = [
                {"role": "user", "content": input_list[i]}
            ]
            prompt=self.llm_tokenizer.apply_chat_template(prompt,add_generation_prompt=True,tokenize=False)
            prompt_tokenized = self.llm_tokenizer(prompt)#, return_tensors="pt").to(self.device)
            input = [
                {"role": "user", "content": input_list[i]},
                {"role": "assistant", "content": response_list[i]}
            ]
            input=self.llm_tokenizer.apply_chat_template(input,add_generation_prompt=False,tokenize=False)
            input_tokenized = self.llm_tokenizer(input)#,return_tensors="pt").to(self.device)

            label = input_tokenized['input_ids'].copy()
            label[:len(prompt_tokenized['input_ids'])] = [-100] * len(prompt_tokenized['input_ids'])

            input_ids.append(input_tokenized['input_ids'])
            labels.append(label)
            
        input_ids, attention_mask = self.padding(input_ids)
        labels, _ = self.padding(labels)
        return input_ids, attention_mask, labels

    def llama_process_prompt(self, input_list, response_list:list):
        input_ids = []
        labels = []
        for i in range(len(response_list)):
            prompt = [
                {"role": "system", "content": ""},
                {"role": "user", "content": input_list[i]}
            ]
            prompt=self.llm_tokenizer.apply_chat_template(prompt,add_generation_prompt=True,tokenize=False)
            prompt_tokenized = self.llm_tokenizer(prompt)#, return_tensors="pt").to(self.device)
            input = [
                {"role": "system", "content": ""},
                {"role": "user", "content": input_list[i]},
                {"role": "assistant", "content": response_list[i]}
            ]
            input = self.llm_tokenizer.apply_chat_template(input,add_generation_prompt=False,tokenize=False)
            input_tokenized = self.llm_tokenizer(input)#,return_tensors="pt").to(self.device)

            label = input_tokenized['input_ids'].copy()
            label[:len(prompt_tokenized['input_ids'])] = [-100] * len(prompt_tokenized['input_ids'])

            input_ids.append(input_tokenized['input_ids'])
            labels.append(label)
            
        input_ids, attention_mask = self.padding(input_ids)
        labels, _ = self.padding(labels)
        return input_ids, attention_mask, labels
            
    def padding(self, input_list:list):
        tokenized_input = self.llm_tokenizer.pad(
            {"input_ids": input_list},
            padding=True,
            return_tensors="pt"
        )
        return tokenized_input['input_ids'].to(self.device), tokenized_input['attention_mask'].to(self.device)

    
    def _tokenize_word(self, word):
        # refer to numBERT: https://github.com/google-research/google-research/tree/master/numbert
        number_pattern = re.compile(
            r"(\d+)\.?(\d*)")  # Matches numbers in decimal form.
        def number_repl(matchobj):
            """Given a matchobj from number_pattern, it returns a string writing the corresponding number in scientific notation."""
            pre = matchobj.group(1).lstrip("0")
            post = matchobj.group(2)
            if pre and int(pre):
                # number is >= 1
                exponent = len(pre) - 1
            else:
                # find number of leading zeros to offset.
                exponent = -re.search("(?!0)", post).start() - 1
                post = post.lstrip("0")
            return (pre + post).rstrip("0") + " scinotexp " + str(exponent)
        
        def apply_scientific_notation(line):
            """Convert all numbers in a line to scientific notation."""
            res = re.sub(number_pattern, number_repl, line)
            return res

        word = clean_cell_value(word)
        word = apply_scientific_notation(word)        
        wordpieces = self.bert_tokenizer.tokenize(word)[:self.data_args.max_token_length]

        mask = [1 for _ in range(len(wordpieces))]
        while len(wordpieces)<self.data_args.max_token_length:
            wordpieces.append('[PAD]')
            mask.append(0)
        return wordpieces, mask

    def _text2table(self, sample):
        smpl = sample.split(HEADER_TAG)
        cap = smpl[0].replace(CAP_TAG, '').strip()
        smpl = smpl[1].split(ROW_TAG)
        headers = [h.strip() for h in smpl[0].strip().split(' | ')]
        descriptions = smpl[-1].split(ROW_DESCRIPTION_TAG)[1:]
        cells_tmp = smpl[1:-1]+[smpl[-1].split(ROW_DESCRIPTION_TAG)[0]]
        cells = [list(map(lambda x: x.strip(), row.strip().split(' | '))) for row in cells_tmp]
        col_descriptions = descriptions[-1].split(COL_DESCRIPTION_TAG)[1:]
        row_descriptions = descriptions[:-1]+[descriptions[-1].split(COL_DESCRIPTION_TAG)[0]]
        col_descriptions = [descriptions[-1].split(COL_DESCRIPTION_TAG)[1]]+col_descriptions[1:]
        smpl = col_descriptions[-1].split(POS_CLAIM_TAG)[-1]
        col_descriptions = col_descriptions[:-1]+[col_descriptions[-1].split(POS_CLAIM_TAG)[0]]
        pos_claim = smpl.split(NEG_CLAIM_TAG)[0].strip()
        neg_claim = smpl.split(NEG_CLAIM_TAG)[-1].strip()
        return cap, headers, cells, row_descriptions, col_descriptions, pos_claim, neg_claim

    def _table2graph(self, examples:list):
        graph_list=[]

        for exm in examples:#exm:str
            #"none (no entries)"
            wordpieces_xs_all, mask_xs_all = [], []
            wordpieces_xt_all, mask_xt_all = [], []
            nodes, edge_index = [], []
            wordpieces_pos_claim, mask_pos_claim = [], []
            wordpieces_neg_claim, mask_neg_claim = [], []

            cap, header, data, row_description, col_description, pos_claim, neg_claim = self._text2table(exm)
            wordpieces, mask = self._tokenize_word(pos_claim)
            wordpieces_pos_claim.append(wordpieces)
            mask_pos_claim.append(mask)
            wordpieces, mask = self._tokenize_word(neg_claim)
            wordpieces_neg_claim.append(wordpieces)
            mask_neg_claim.append(mask)
            
            assert len(data[0]) == len(header)
            # caption to hyper-edge (t node)
            wordpieces, mask = self._tokenize_word(cap)#到这里出现问题
            wordpieces_xt_all.append(wordpieces)
            mask_xt_all.append(mask)

         # hyperedge with row description
            for i, row in enumerate(row_description):
                wordpieces, mask = self._tokenize_word(row)
                wordpieces_xt_all.append(wordpieces)
                mask_xt_all.append(mask)
                
            # hyperedge with col description
            for i, col in enumerate(col_description):
                wordpieces, mask = self._tokenize_word(col)
                wordpieces_xt_all.append(wordpieces)
                mask_xt_all.append(mask)


            
            for row_i, row in enumerate(data):#遍历每一个cell
                for col_i, word in enumerate(row):
                    if word == "none": #modified by hanqian
                        word = MISSING_CELL_TAG #tabfact
                        wordpieces = [word] + ['[PAD]' for _ in range(self.data_args.max_token_length - 1)]
                        mask = [1] + [0 for _ in range(self.data_args.max_token_length - 1)]
                    else:
                        word = ' '.join(word.split()[:self.data_args.max_token_length])
                        wordpieces, mask = self._tokenize_word(word)

                    wordpieces_xs_all.append(wordpieces)
                    mask_xs_all.append(mask)
                    node_id = len(nodes)
                    nodes.append(node_id)
                    edge_index.append([node_id, 0])
                    edge_index.append([node_id, col_i+1]) # cell-col
                    edge_index.append([node_id, row_i + 1 + len(header)])
                    
            tab_mask = (torch.zeros(len(wordpieces_xt_all), dtype=torch.long)).to(self.device)
            tab_mask[0] = 1
                
            xs_ids = torch.tensor([self.bert_tokenizer.convert_tokens_to_ids(x) for x in wordpieces_xs_all], dtype=torch.long).to(self.device)
            xt_ids = torch.tensor([self.bert_tokenizer.convert_tokens_to_ids(x) for x in wordpieces_xt_all], dtype=torch.long).to(self.device)
            pos_ids = torch.tensor([self.bert_tokenizer.convert_tokens_to_ids(x) for x in wordpieces_pos_claim], dtype=torch.long).to(self.device) 
            neg_ids = torch.tensor([self.bert_tokenizer.convert_tokens_to_ids(x) for x in wordpieces_neg_claim], dtype=torch.long).to(self.device)           
            
            # check all 0 input
            xs_tem = torch.count_nonzero(xs_ids, dim =1)
            xt_tem = torch.count_nonzero(xt_ids, dim=1)
            assert torch.count_nonzero(xs_tem) == len(xs_tem)
            assert torch.count_nonzero(xt_tem) == len(xt_tem)
            edge_index = (torch.tensor(edge_index, dtype=torch.long).T).to(self.device)
            
            bigraph = BipartiteData(edge_index=edge_index, x_s=xs_ids, x_t=xt_ids, tab_mask = tab_mask, pos_claim= pos_ids, neg_claim=neg_ids, num_nodes=len(xs_ids), this_num_nodes=len(xs_ids), this_num_edges=len(xt_ids))
            graph_list.append(bigraph)
        batch_bigraph = Batch.from_data_list(graph_list)
        return batch_bigraph
    
    def llm_output(self, hyperg_embedding, input_ids, attention_mask, labels):#来进行concat的时候把llama的embedding抠出来，替换成hyperg的embedding
        llm_embedding = self.llm_model.model.base_model.embed_tokens(input_ids)
        unk_ids = self.llm_tokenizer.unk_token_id
        unk_mask = (input_ids == unk_ids).unsqueeze(-1)
        embedding = torch.where(unk_mask, hyperg_embedding, llm_embedding).bfloat16()
      
        output = self.llm_model(
                    inputs_embeds=embedding,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=labels,
                )
        return output
    
    def hypergraph_output(self, data):
        outputs= self.hyperg_encoder(data)#(source node embedding, target node embedding)
        hyperedge_outputs = outputs[1]
        tab_embeds = torch.index_select(hyperedge_outputs, 0, torch.nonzero(data.tab_mask).squeeze())#torch.Size([1, 768]) #batch.tab_mask,除了第0维其他都mask住，在构建超图的时候，第0维代表的是整个table的embedding
        return tab_embeds

    def forward(self, sample):
        #process_saple = fun(sample)#对输入进行处理之后再给hyperg
        tag_table = sample['tag_table']
        prompts = sample['prompts']
        labels = sample['labels']
        prompts = [prompt.replace('<HYPER_EMB_TAG>', self.unk_) for prompt in prompts]
        graph = self._table2graph(tag_table)
        hyperg_embedding = self.hypergraph_output(graph)#获取的是整个table的embedding
        hyperg_embedding = self.proj(hyperg_embedding).unsqueeze(1)
        input_ids, attention_mask, labels = self.llama_process_prompt(prompts, labels)
        output = self.llm_output(hyperg_embedding, input_ids, attention_mask, labels)
        return output

# ********************************* prepare the dataset (arrow) *********************************
def extract_table(text):#获取输入中的table内容
    pattern_content = r"(<caption>.*?)(?=<row_description>)"#not including description,tag
    pattern_content_des =  r"(<caption>.*?)(?=<pos_claim>)"#including description,tag
    pos_claim = r"<pos_claim>(.*?)<neg_claim>"
    neg_claim = r"<neg_claim>(.*?)<MARK_DOWN>"
    tag_all = r"(<caption>.*?)(?=<MARK_DOWN>)"#including description,tag and claim
    markdown_table = r"<MARK_DOWN>(.*)"

    content_match = re.search(pattern_content, text, re.DOTALL)
    content_des_match = re.search(pattern_content_des,text, re.DOTALL) 
    pos_claim_match = re.search(pos_claim, text, re.DOTALL)
    neg_claim_match = re.search(neg_claim,text, re.DOTALL)
    tag_all_match = re.search(tag_all,text, re.DOTALL)
    markdown_table_macth = re.search(markdown_table,text, re.DOTALL)

    return content_match.group(1).strip(), content_des_match.group(1).strip(), pos_claim_match.group(1).strip(), neg_claim_match.group(1).strip(), tag_all_match.group(1).strip(), markdown_table_macth.group(1).strip()

def process_func_tfv(examples):
    tag_table, prompts, labels= [],[],[]
    for example in examples['text']:
        table_no_des, table_des, pos_claim, neg_claim, tag_all_table, markdown_table= extract_table(example)
        
        if len(markdown_table)>5000:
            continue
        prompt_prefix = "A markdown table is provided below:"
        hyperg_emb_tag = '<HYPER_EMB_TAG>'
        prompt_pos = f"{prompt_prefix}\n {markdown_table}.\n The key information extracted from the data is encoded in the feature {hyperg_emb_tag}, which relates to this claim: {pos_claim}\n Based on all the available information, predict whether this claim is correct. Respond with 'Yes' or 'No' only."

        prompt_neg = f"{prompt_prefix}\n {markdown_table}.\n The key information extracted from the data is encoded in the feature {hyperg_emb_tag}, which relates to this claim: {neg_claim}\n Based on all the available information, predict whether this claim is correct. Respond with 'Yes' or 'No' only."
        tag_table.append(tag_all_table)#ours是加了description的
        
        prompt = [(prompt_pos, "Yes"), (prompt_neg, "No")]
        selected_prompt, label = random.choice(prompt)
        prompts.append(selected_prompt)
        labels.append(label)

    return {
        "tag_table": tag_table,
        "prompts": prompts,
        "labels" : labels
        }

def process_func_tqa(examples):#需要加一个prompt
    tag_table, prompts, labels= [],[],[]
    for example in examples['text']:
        try:
            table_no_des, table_des, question, answer, tag_all_table, markdown_table= extract_table(example)
        except:
            continue
        prompt_prefix = "A markdown table is provided below:"
        hyperg_emb_tag = '<HYPER_EMB_TAG>'
        prompt= f"{prompt_prefix}\n {markdown_table}.\n The key information extracted from the data is encoded in the feature {hyperg_emb_tag}, which relates to this question: {question}\n Based on all the available information, provide a concise answer only. Do not include any reasoning or additional details."
        """tokenized_prompt =  tokenizer.encode(prompt)
        if len(tokenized_prompt)>2500:
            breakpoint()
            continue"""
        prompts.append(prompt)
        tag_table.append(tag_all_table)#ours是加了description的
        labels.append(answer)
    return {
        "tag_table": tag_table,
        "prompts": prompts,
        "labels" : labels
        }

def load_arrow_as_dataset(datapath):
    mmap = pa.memory_map(datapath)
    dataset = (pa.ipc.open_file(mmap).read_all()).combine_chunks()
    df = dataset.to_pandas()
    hf_dataset = Dataset.from_pandas(df)    
    return hf_dataset#['text']

parser = HfArgumentParser((DataArguments, OptimizerConfig))
(data_args, optimizer_cfg) =  parser.parse_args_into_dataclasses()

# ********************************* prepare args *********************************
datapath = data_args.dataset_path
model_name = data_args.llm_name
saving_ckpt_path = data_args.save_path
task_type = data_args.task_type
pretrain_hyperg = data_args.pretrain_hyperg

step_size = optimizer_cfg.step_size
gamma = optimizer_cfg.gamma
batch_size = optimizer_cfg.batch_size
lr = optimizer_cfg.learning_rate
scale = optimizer_cfg.scale
num_train_epochs = optimizer_cfg.epochs
accumulation_steps = optimizer_cfg.accumulation_steps
# ********************************************************************************
dataset = load_arrow_as_dataset(datapath)
if task_type == 'tfv':
    dataset = dataset.map(process_func_tfv,batched = True,remove_columns=dataset.column_names)
elif task_type == 'tqa':
    dataset = dataset.map(process_func_tqa,batched = True,remove_columns=dataset.column_names)

dataloader = DataLoader(dataset, batch_size = batch_size, shuffle=True)



bert_tokenizer = AutoTokenizer.from_pretrained(
        data_args.tokenizer_config_type)
new_tokens = ['[TAB]', '[HEAD]', '[CELL]', '[ROW]', "scinotexp"]
bert_tokenizer.add_tokens(new_tokens)
model_config = AutoConfig.from_pretrained(data_args.tokenizer_config_type)
#model_config.update({'vocab_size': len(bert_tokenizer), "pre_norm": False, "activation_dropout":0.1, "gated_proj": False})#原始
model_config.update({'vocab_size': len(bert_tokenizer), "pre_norm": False, "activation_dropout":0.15, "gated_proj": False, "hidden_dropout_prob":0.15, "num_hidden_layers" :12})#调参
hypergraph_encoder = Encoder(model_config)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,  # LoRA秩
    lora_alpha=32,  # LoRA缩放系数
    lora_dropout=0.1,  # dropout
    target_modules=["q_proj", "k_proj", "v_proj" ],
    )

HyperG_model = HyperG(
                        data_args,
                        model_name,
                        hypergraph_encoder,
                        bert_tokenizer,
                        True,#是否用lora
                        pretrain_hyperg,
                        lora_config,
                            )

  
# optimizer = Adam(hyper_llama_model.parameters(), lr=lr)
# scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
# total_steps = len(dataloader) * num_train_epochs

param_groups = [
    {'params': [p for n, p in HyperG_model.hyperg_encoder.named_parameters()], 'lr': lr*scale},
    {'params': [p for n, p in HyperG_model.proj.named_parameters()], 'lr': lr*scale},
    {'params': [p for n, p in HyperG_model.llm_model.named_parameters()], 'lr': lr},
]
optimizer = torch.optim.AdamW(param_groups)
total_steps = len(dataloader) * num_train_epochs
scheduler = LinearLR(optimizer=optimizer,
                     warmup_steps=total_steps*0.1,
                     steps=total_steps)

progress_bar = tqdm(total=total_steps, desc="Training Progress")
global_step = 0

saving_ckpt_path = os.path.join(saving_ckpt_path, model_name.split('/')[-1])
os.makedirs(saving_ckpt_path, exist_ok=True)
os.makedirs(os.path.join(saving_ckpt_path,"encoder"), exist_ok=True)
os.makedirs(os.path.join(saving_ckpt_path,"lora"), exist_ok=True)
os.makedirs(os.path.join(saving_ckpt_path,"proj"), exist_ok=True)

for epoch in range(num_train_epochs):
    for step, batch in enumerate(dataloader):#输入给模型的，整个enchanced_table(加上相应的claim),两个cliam，表格本身(用来过hypergraph)
        optimizer.zero_grad()
        output = HyperG_model(batch)
        loss = output.loss
        loss = loss/accumulation_steps#梯度累加
        loss.backward()
        if ((step+1)%accumulation_steps)==0:
            optimizer.step()
        progress_bar.update(1)
        if ((global_step+1)%10)==0:
            print(f"Epoch: {epoch + 1}, Step: {global_step+1}, Loss: {loss.item()}",flush=True)
        
        
        """if  ((global_step+1)%total_steps)==0: #or ((global_step+1)%1000)==0 : 
            os.makedirs(os.path.join(saving_ckpt_path,"encoder"), exist_ok=True)
            os.makedirs(os.path.join(saving_ckpt_path,"lora"), exist_ok=True)
            os.makedirs(os.path.join(saving_ckpt_path,"proj"), exist_ok=True)"""
        
        if ((global_step+1)%3000)==0 or ((global_step+1)%total_steps)==0:
            encoder_path = os.path.join(saving_ckpt_path, "encoder", f"encoder_weights_step_{global_step+1}.pt")
            lora_path = os.path.join(saving_ckpt_path, "lora", f"lora_weights_step_{global_step+1}")
            projector_path = os.path.join(saving_ckpt_path, "proj", f"projector_weights_step_{global_step+1}.pt")

            HyperG_model.llm_model.save_pretrained(lora_path)
            torch.save(HyperG_model.hyperg_encoder.state_dict(), encoder_path)
            torch.save(HyperG_model.proj.state_dict(), projector_path)
        global_step += 1
        scheduler.step()

