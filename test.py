import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import argparse
import pickle
import json
import pandas as pd
import pyarrow as pa
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import trange
from peft import PeftModel
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, HfArgumentParser
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import get_peft_model, LoraConfig, TaskType, get_peft_model_state_dict
from data_utils import clean_cell_value
from data import BipartiteData
from collections import OrderedDict
import re
from model import Encoder
from torch_geometric.data.batch import Batch
import random
from tqdm import tqdm
from dataclasses import dataclass, field

def set_random_seed(seed):
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed_all(seed)  # GPU
    random.seed(seed)
set_random_seed(520)


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
huggingface_token = "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"


def prepare_llama_lora(model_name, lora_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name,token=huggingface_token)
    tokenizer.add_special_tokens(
            {'pad_token': '[PAD]',
             'unk_token': '[UNK]'
             }    
        )
    model = AutoModelForCausalLM.from_pretrained(model_name, token=huggingface_token,torch_dtype=torch.bfloat16, device_map = "auto")
    embedding_dim = model.get_input_embeddings().embedding_dim
    model.resize_token_embeddings(len(tokenizer))
    lora_model = PeftModel.from_pretrained(model, lora_path, device_map="auto")
    return lora_model, tokenizer,embedding_dim


def prepare_hyperg(data_args, model_path):
    bert_tokenizer = AutoTokenizer.from_pretrained(data_args.tokenizer_config_type)
    new_tokens = ['[TAB]', '[HEAD]', '[CELL]', '[ROW]', "scinotexp"]
    bert_tokenizer.add_tokens(new_tokens)

    model_config = AutoConfig.from_pretrained(data_args.tokenizer_config_type)
    #model_config.update({'vocab_size': len(bert_tokenizer), "pre_norm": False, "activation_dropout":0.1, "gated_proj": False})
    model_config.update({'vocab_size': len(bert_tokenizer), "pre_norm": False, "activation_dropout":0.15, "gated_proj": False, "hidden_dropout_prob":0.15, "num_hidden_layers":12})#调参
    
    hypergraph_encoer = Encoder(model_config)

    hypergraph_encoer.load_state_dict(torch.load(model_path))

    return bert_tokenizer, hypergraph_encoer

def prepare_llama_projecter(model_path, hyperg_encoder_dim= 768, hidden_layer_dim = 2048, llama_embedding_dim=4096):#8b是4096,3b是3072
    #breakpoint()
    llama_proj = nn.Sequential(
        nn.Linear(hyperg_encoder_dim , hidden_layer_dim),
        nn.Tanh(),
        nn.Linear(hidden_layer_dim, llama_embedding_dim) #hidden_layer_dim随便设置的
        )
    llama_proj.load_state_dict(torch.load(model_path))

    return llama_proj

class hyper_llama(nn.Module):
    def __init__(
        self,
        data_args = None,
        llama_lora_model = None,
        llama_tokenizer = None,
        hyperg_encoder = None,
        bert_tokenizer = None,
        projector = None,
        shuffle_ratio = 0
    ):
        super(hyper_llama, self).__init__()

        self.data_args = data_args
        self.llama_lora_model = llama_lora_model
        self.device = self.llama_lora_model.device
        self.llama_tokenizer = llama_tokenizer
        self.hyperg_encoder = hyperg_encoder.to(self.device)
        self.bert_tokenizer = bert_tokenizer
        self.projector = projector.to(self.device)
        self.unk_ = self.llama_tokenizer.unk_token
        self.shuffle_ratio = shuffle_ratio
        self.shuffle = None
    def llama_process_prompt(self, input_list):#"Yes" or "No"
        #tokenized_input = self.llama_tokenizer(input, padding="longest", add_special_tokens = True, truncation=False,return_tensors="pt").to(self.device)       
        #input_ids = tokenized_input["input_ids"]
        #attention_mask = tokenized_input["attention_mask"]
        prompts = []
        for i in range(len(input_list)):
            prompt = [
                #{"role": "system", "content": ""},#gemma去掉这行
                {"role": "user", "content": input_list[i]}
            ]
            prompts.append(prompt)
        tokenized_input = self.llama_tokenizer.apply_chat_template(prompts,add_generation_prompt=True,tokenize=True, return_tensors="pt",padding = True, return_dict = True).to(self.device)
        input_ids = tokenized_input["input_ids"]
        attention_mask = tokenized_input["attention_mask"]
        return input_ids, attention_mask
    
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

    """def _text2table(self, sample):
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
        breakpoint()
        return cap, headers, cells, row_descriptions, col_descriptions, pos_claim, neg_claim"""

    def shuffle_row(self,cell_list, row_description_list):
        combined = list(zip(cell_list, row_description_list))
        random.shuffle(combined)
        
        shuffled_cell_list, shuffled_row_descriptions = zip(*combined)
        
        shuffled_cell_list = list(shuffled_cell_list)
        shuffled_row_descriptions = list(shuffled_row_descriptions)
        #breakpoint()
        return shuffled_cell_list, shuffled_row_descriptions

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
        if self.shuffle:
            cells, row_descriptions = self.shuffle_row(cells,row_descriptions)
        return cap, headers, cells, row_descriptions, col_descriptions, pos_claim, neg_claim

    def _table2graph(self, examples:list):
        #data_list = []
        #batch_edge_index, batch_xs_ids, batch_xt_ids, batch_tab_mask, bacth_pos_cla_ids, bacth_neg_cla_ids= [torch.tensor([], dtype=torch.long).to(self.device) for _ in range(6)]
        graph_list=[]
        MISSING_CELL_TAG = "[CELL]"
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


            # cell to nodes (s node)
            """for row_i, row in enumerate(data):
                for col_i, word in enumerate(row):
                    if not word:
                        wordpieces = ['[CELL]'] + ['[PAD]' for _ in range(self.data_args.max_token_length - 1)]
                        mask = [1] + [0 for _ in range(self.data_args.max_token_length - 1)]
                    else:
                        word = ' '.join(str(word).split()[:self.data_args.max_token_length])
                        wordpieces, mask = self._tokenize_word(word)
                    wordpieces_xs_all.append(wordpieces)
                    mask_xs_all.append(mask)
                    node_id = len(nodes)
                    nodes.append(node_id)
                    edge_index.append([node_id, 0]) # connect to table-level hyper-edge
                    edge_index.append([node_id, col_i+1]) # # connect to col-level hyper-edge
                    edge_index.append([node_id, row_i + 1 + len(header)])  # connect to row-level hyper-edge"""

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
                    node_id = len(nodes)# 给每个cell一个id
                    nodes.append(node_id)
                    edge_index.append([node_id, 0]) # edge_index的格式是[[node_id, edge_id],...], 列表中每个元素代表一条node-hyperedge, edge_id=0为cell-caption
                    edge_index.append([node_id, col_i+1]) # cell-col
                    edge_index.append([node_id, row_i + 1 + len(header)])
                    
            # add label,不需要
            #label_ids =  torch.tensor(node_id, dtype=torch.long)#label 换成了node_id
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
    
    def llama_output(self, hyperg_embedding, input_ids, attention_mask):#来进行concat的时候把llama的embedding抠出来，替换成hyperg的embedding
        llama_embedding = self.llama_lora_model.model.base_model.embed_tokens(input_ids)
        unk_ids = self.llama_tokenizer.unk_token_id
        unk_mask = (input_ids == unk_ids).unsqueeze(-1)
        sampling_params = {
            "max_new_tokens": input_ids.shape[-1]+50,#50,#input_ids.shape[-1]+50,#llama3.2-3b
            "temperature": 0.6,
            "top_p": 0.95,
            #"min_length": 1,
            "do_sample": True,
        }
        embedding = torch.where(unk_mask, hyperg_embedding, llama_embedding).bfloat16()
        output = self.llama_lora_model.generate(
                    inputs_embeds=embedding,
                    attention_mask=attention_mask,
                    return_dict=True,
                    **sampling_params,
                )
        return output
    
    def hypergraph_output(self, data):
        outputs= self.hyperg_encoder(data)#(source node embedding, target node embedding)
        hyperedge_outputs = outputs[1]
        tab_embeds = torch.index_select(hyperedge_outputs, 0, torch.nonzero(data.tab_mask).squeeze())#torch.Size([1, 768]) #batch.tab_mask,除了第0维其他都mask住，在构建超图的时候，第0维代表的是整个table的embedding
        return tab_embeds

    def forward(self, tag_table, prompts):
        #tag_table = sample['tag_table']
        #prompts = sample['prompts']
        random_value = torch.rand(1).item()
        if random_value <= self.shuffle_ratio:
            self.shuffle = True
        else:
            self.shuffle = False
        prompts = [prompt.replace('<HYPER_EMB_TAG>', self.unk_) for prompt in prompts]
        graph = self._table2graph(tag_table)
        hyperg_embedding = self.hypergraph_output(graph)
        hyperg_embedding = self.projector(hyperg_embedding).unsqueeze(1)
        input_ids, attention_mask = self.llama_process_prompt(prompts)
        output = self.llama_output(hyperg_embedding, input_ids, attention_mask)
        return output

def load_arrow_as_dataset(datapath):
    mmap = pa.memory_map(datapath)
    dataset = (pa.ipc.open_file(mmap).read_all()).combine_chunks()
    df = dataset.to_pandas()
    hf_dataset = Dataset.from_pandas(df)    
    return hf_dataset


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


def process_func(examples):#需要加一个prompt
    tag_table, prompts, labels= [],[],[]
    for example in examples['text']:
        table_no_des, table_des, pos_claim, neg_claim, tag_all_table, markdown_table= extract_table(example)
        prompt_prefix = "A markdown table is provided below:"
        hyperg_emb_tag = '<HYPER_EMB_TAG>'
        prompt_pos = f"{prompt_prefix}\n {markdown_table}.\n The key information extracted from the data is encoded in the feature {hyperg_emb_tag}, which relates to this claim: {pos_claim}\n Based on all the available information, predict whether this claim is correct. Respond with 'Yes' or 'No' only."

        prompt_neg = f"{prompt_prefix}\n {markdown_table}.\n The key information extracted from the data is encoded in the feature {hyperg_emb_tag}, which relates to this claim: {neg_claim}\n Based on all the available information, predict whether this claim is correct. Respond with 'Yes' or 'No' only."
        tag_table.append(tag_all_table)

        prompt = [(prompt_pos, "Yes"), (prompt_neg, "No")]
        selected_prompt, label = random.choice(prompt)
        prompts.append(selected_prompt)
        labels.append(label)

    return {
        "tag_table": tag_table,
        "prompts": prompts,
        "labels" : labels
        }

@dataclass
class DataArguments:
    """
    Arguments pertaining to which config/tokenizer we are going use.
    """
    tokenizer_config_type: str = field(
        default='bert-base-uncased',
        metadata={
            "help": "bert-base-cased, bert-base-uncased etc"
        },
    )
    data_path: str = field(default='/hpc2hdd/JH_DATA/share/zrao538/PrivateShareGroup/zrao538_NLPGroup/hanqianli/hanqianli/WWW25/data/scalability_data/large_tables.arrow', metadata={"help": "data path"})#'/hpc2hdd/JH_DATA/share/zrao538/PrivateShareGroup/zrao538_NLPGroup/huangsirui/WWW25/hyperg/ckpt_data/data/ttd/ttd/'
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
    """label_type_num: int = field(
        default=255,
        metadata={
            "help": "The total label types"

        },
    )

    valid_ratio: float = field(
        default=0.3,
        metadata={"help": "Number of workers for dataloader"},
    )"""


    def __post_init__(self):
        if self.tokenizer_config_type not in ["bert-base-cased", "bert-base-uncased"]:
            raise ValueError(
                f"The model type should be bert-base-(un)cased. The current value is {self.tokenizer_config_type}."
            )
#parser = HfArgumentParser(DataArguments)
#data_args = parser.parse_args_into_dataclasses()[0]

def parse_args():
    parser = argparse.ArgumentParser(description="Get executor data for train and evaluation task")

    parser.add_argument(
        "--save_path",
        type=str,
        default= "/hpc2hdd/JH_DATA/share/zrao538/PrivateShareGroup/zrao538_NLPGroup/hanqianli/hanqianli/WWW25/tfv/test_unpretrained_ours_lora/gemma-2-9b-it",
        help="path to directory containing model weights and config file"
    )

    parser.add_argument(
        "--llama_path",
        type=str,
        default= 'google/gemma-2-9b-it',#'/hpc2hdd/JH_DATA/share/zrao538/PrivateShareGroup/zrao538_NLPGroup/yggu/huggingface/meta-llama/Llama-3.2-3B-Instruct',#'/hpc2hdd/JH_DATA/share/zrao538/PrivateShareGroup/zrao538_NLPGroup/yggu/huggingface/meta-llama/Llama-3.2-3B-Instruct',#'meta-llama/Llama-2-7b-chat-hf',meta-llama/Meta-Llama-3-8B-Instruct
        help="path to directory containing model weights and config file"
    )

    parser.add_argument(
        "--lora_path",
        type=str,
        default= '/hpc2hdd/JH_DATA/share/zrao538/PrivateShareGroup/zrao538_NLPGroup/hanqianli/hanqianli/WWW25/tfv/test_unpretrained_ours_lora/gemma-2-9b-it/lora/lora_weights_step_7396'
    )

    parser.add_argument(
        "--hyper_encoder_path",
        type=str,
        default= '/hpc2hdd/JH_DATA/share/zrao538/PrivateShareGroup/zrao538_NLPGroup/hanqianli/hanqianli/WWW25/tfv/test_unpretrained_ours_lora/gemma-2-9b-it/encoder/encoder_weights_step_7396.pt'
    )

    parser.add_argument(
        "--pro_path",
        type=str,
        default= '/hpc2hdd/JH_DATA/share/zrao538/PrivateShareGroup/zrao538_NLPGroup/hanqianli/hanqianli/WWW25/tfv/test_unpretrained_ours_lora/gemma-2-9b-it/proj/projector_weights_step_7396.pt'
    )
    parser.add_argument(
        "--shuffle_ratio",
        type=float,
        default= 0.2
    )
    args, unknown = parser.parse_known_args()
    return args, unknown


args, remaining_args = parse_args()
hf_parser = HfArgumentParser(DataArguments)
data_args, = hf_parser.parse_args_into_dataclasses(remaining_args)
save_path = args.save_path

llama_lora_model, llama_tokenizer, embedding_dim = prepare_llama_lora(args.llama_path, args.lora_path)
bert_tokenizer, hyper_encoder = prepare_hyperg(data_args, args.hyper_encoder_path)
projector = prepare_llama_projecter(args.pro_path, llama_embedding_dim=embedding_dim)
#breakpoint()
shuffle_ratio = args.shuffle_ratio

ours_model = hyper_llama(
    data_args,
    llama_lora_model,
    llama_tokenizer,
    hyper_encoder,
    bert_tokenizer,
    projector,
    shuffle_ratio
    )
ours_model.eval()
test_datapath = "/hpc2hdd/JH_DATA/share/zrao538/PrivateShareGroup/zrao538_NLPGroup/huangsirui/WWW25/hyperg/data/pretrain/test_claim_arrow/dataset.arrow"#"/hpc2hdd/JH_DATA/share/zrao538/PrivateShareGroup/zrao538_NLPGroup/hanqianli/hanqianli/WWW25/data/scalability_data/large_tables.arrow"#"/hpc2hdd/JH_DATA/share/zrao538/PrivateShareGroup/zrao538_NLPGroup/huangsirui/WWW25/hyperg/data/pretrain/test_claim_arrow/dataset.arrow"
dataset = load_arrow_as_dataset(test_datapath)
dataset = dataset.map(process_func,batched = True,remove_columns=dataset.column_names)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

progress_bar = tqdm(total=len(dataloader), desc="Testing Progress")

def find_label_from_response(response: str):
    labels = ['YES', 'NO']
    for label in labels:
        if label in response:
            return label
    return 'NONE'

def eval(df, avg_metric='weighted'):
    accuracy = accuracy_score(df['label'], df['pred']) * 100
    f1 = f1_score(df['label'], df['pred'], average=avg_metric, zero_division=0) * 100
    precision = precision_score(df['label'], df['pred'], average=avg_metric, zero_division=0) * 100
    recall = recall_score(df['label'], df['pred'], average=avg_metric, zero_division=0) * 100

    print('Accuracy (%)', round(accuracy, 2))
    print('F1 Score (%)', round(f1, 2))
    print('Precision (%)', round(precision, 2))
    print('Recall (%)', round(recall, 2))

resp_list = []
ground_truth_list = []
for step, batch in enumerate(dataloader):
    tag_table = batch['tag_table']
    prompts =batch['prompts']
    labels = batch['labels']
    inputs = prompts.copy()
    with torch.no_grad():
        outputs = ours_model(
            tag_table,
            inputs,
            )
    progress_bar.update(1)
    decode_token = ours_model.llama_tokenizer.batch_decode(outputs, skip_special_tokens=True)
    resp_list.extend(decode_token)
    ground_truth_list.extend(labels) 

df = pd.DataFrame({'resp': resp_list, 'label': ground_truth_list})
df = df.map(lambda x: x.upper() if isinstance(x, str) else x)
df['pred'] = df['resp'].apply(find_label_from_response)
#save_path = "/hpc2hdd/JH_DATA/share/zrao538/PrivateShareGroup/zrao538_NLPGroup/hanqianli/hanqianli/WWW25/tfv/test_pretrained_ours_lora/Meta-Llama-3-8B-Instruct"
csv_path = os.path.join(save_path, f"1_23_shuffle_{shuffle_ratio}.csv")
df.to_csv(csv_path, index=False, header=True, sep='\t', encoding='utf-8', mode='w')
eval(df)




