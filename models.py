import torch
#from layer_baseline import *
from layers import *

#********************************* HGNN Model Start*********************************
from torch import nn
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter
from transformers import AutoTokenizer, AutoModelForCausalLM


class Embedding(nn.Module):
  def __init__(self, config):
    super(Embedding, self).__init__()
    self.tok_embed = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)  # token embedding
    self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.dropout = nn.Dropout(config.hidden_dropout_prob)

  def forward(self, data): 
    x_s, x_t = data.x_s, data.x_t
    #pos_claim, neg_claim = data.pos_claim, data.neg_claim
    embedding_s, embedding_t = self.tok_embed(x_s), self.tok_embed(x_t)
    embedding_s, embedding_t = torch.div(torch.sum(embedding_s, dim=1), torch.count_nonzero(x_s, dim=1).unsqueeze(-1)), torch.div(torch.sum(embedding_t, dim=1), torch.count_nonzero(x_t, dim=1).unsqueeze(-1))#1.计算每个样本的特征总和（embedding_s 的第 1 维求和). 2.计算每个样本中非零特征的数量（x_s 的非零计数）。3.对特征总和进行归一化：通过非零特征的数量除以特征总和，得到归一化的平均特征值。

    if data.pos_claim is not None:
        pos_claim = data.pos_claim
        embedding_claim = self.tok_embed(pos_claim)
        embedding_claim = torch.div(torch.sum(embedding_claim, dim=1), torch.count_nonzero(pos_claim, dim=1).unsqueeze(-1))
        
        # embedding_pos_repeated_t = torch.cat([embedding_pos[i].repeat(data.this_num_edges[i], 1, 1) for i in range(len(data.this_num_edges))], dim=0) # repeated to match size of embedding_t
        # pos_claim_repeated_t = torch.cat([pos_claim[i].repeat(data.this_num_edges[i], 1) for i in range(len(data.this_num_edges))], dim=0)
        # embedding_pos_t = torch.div(torch.sum(embedding_pos_repeated_t, dim=1), torch.count_nonzero(pos_claim_repeated_t, dim=1).unsqueeze(-1))
        
        # embedding_pos_repeated_s = torch.cat([embedding_pos[i].repeat(data.num_nodes[i], 1, 1) for i in range(len(data.num_nodes))], dim=0)
        # pos_claim_repeated_s = torch.cat([pos_claim[i].repeat(data.num_nodes[i], 1) for i in range(len(data.num_nodes))], dim=0)
        # embedding_pos_s = torch.div(torch.sum(embedding_pos_repeated_s, dim=1), torch.count_nonzero(pos_claim_repeated_s, dim=1).unsqueeze(-1))
        ########### directly add task instruction to edges
        # lst_idx = 0
        # for i in range(len(data)):
        #     d = data[i]
        #     embedding_s[lst_idx:int(d.this_num_edges),:]+=embedding_pos[i,:]
        #     lst_idx += int(d.this_num_edges)
        
    #if data.neg_claim is not None:
        #neg_claim = data.neg_claim
        #embedding_claim = self.tok_embed(neg_claim)
        #embedding_claim = torch.div(torch.sum(embedding_claim, dim=1), torch.count_nonzero(neg_claim, dim=1).unsqueeze(-1))
    
    embedding_claim = self.dropout(self.norm(embedding_claim))
    embedding_instruct_s = torch.cat([embedding_claim[i].repeat(data.this_num_nodes[i], 1) for i in range(len(data.this_num_nodes))], dim=0)
    embedding_instruct_t = torch.cat([embedding_claim[i].repeat(data.this_num_edges[i], 1) for i in range(len(data.this_num_edges))], dim=0)
    embedding_instruct = torch.cat([embedding_instruct_t, embedding_instruct_s])

    return self.dropout(self.norm(embedding_s)), self.dropout(self.norm(embedding_t)), embedding_instruct
        

class EncoderLayer(nn.Module):
  #SetTransformer Encoder Layer
  def __init__(self, config):
    super().__init__()
    self.dropout = config.hidden_dropout_prob
    self.V2E = AllSetTrans(config = config)
    self.fuse = nn.Linear(config.hidden_size*2, config.hidden_size)
    self.E2V = AllSetTrans(config = config)


  def forward(self, embedding_s, embedding_t, edge_index, embedding_instruct, i, data):
    data.i = i
    data.reversed = False
    ##### V2E first, then E2V ######## # change
    # reverse the index
    reversed_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
    # from nodes to hyper-edges
    embedding_t_tem = F.relu(self.V2E(embedding_s, edge_index, None, data))
    # from hyper-edges to nodes
    embedding_t = torch.cat([embedding_t, embedding_t_tem], dim=-1) #concat node aggr features and the original edge features at the last dim
    
    # fuse the output t_embeds with original t_embeds, or the t_embeds will not have the original info
    embedding_t = F.dropout(self.fuse(embedding_t), p=self.dropout, training=self.training)
    data.reversed = True
    embedding_s = F.relu(self.E2V(embedding_t, reversed_edge_index, embedding_instruct, data))
    embedding_s = F.dropout(embedding_s, p=self.dropout, training=self.training)

    return embedding_s, embedding_t



class Encoder(nn.Module):#model
  def __init__(self, config):
    super(Encoder, self).__init__()
    self.config = config
    self.embed_layer = Embedding(config)
    self.layer = nn.ModuleList([EncoderLayer(config) for _ in range(config.num_hidden_layers)])
  def forward(self, data):
    embedding_s, embedding_t, embedding_instruct = self.embed_layer(data)
    embedding_t = torch.cat([embedding_t, embedding_s], dim=0)

    # # Add self-loop
    num_nodes, num_hyper_edges = data.x_s.size(0), data.x_t.size(0)
    self_edge_index = torch.tensor([[i, num_hyper_edges+i] for i in range(num_nodes)]).T
    if ('edge_neg_view' in self.config.to_dict() and self.config.edge_neg_view == 1):
        edge_index = torch.cat([data.edge_index_corr1, self_edge_index.to(data.edge_index_corr1.device)], dim=-1)
    elif ('edge_neg_view' in self.config.to_dict() and self.config.edge_neg_view == 2):
        edge_index = torch.cat([data.edge_index_corr2, self_edge_index.to(data.edge_index_corr2.device)], dim=-1)
    else:
        edge_index = torch.cat([data.edge_index, self_edge_index.to(data.edge_index.device)], dim=-1)

    for i, layer_module in enumerate(self.layer):
      embedding_s, embedding_t  = layer_module(embedding_s, embedding_t, edge_index, embedding_instruct, i, data)
    # breakpoint()
    outputs = (embedding_s, embedding_t[:num_hyper_edges])
    return outputs#return embedding



class ContrastiveLoss(nn.Module):
   """
   Vanilla Contrastive loss, also called InfoNceLoss as in SimCLR paper
   """
   def __init__(self, temperature=0.5):
       super().__init__()
  
       self.temperature = temperature
       self.loss_fct = nn.CrossEntropyLoss()


   def calc_similarity_batch(self, a, b):
       representations = torch.cat([a, b], dim=0)
       return F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

   def forward(self, proj_1, proj_2):
       """
       proj_1 and proj_2 are batched embeddings [batch, embedding_dim]
       where corresponding indices are pairs
       z_i, z_j in the SimCLR paper
       """
       
       z_i = F.normalize(proj_1, p=2, dim=1)
       z_j = F.normalize(proj_2, p=2, dim=1)
       
       cos_sim = torch.einsum('id,jd->ij', z_i, z_j)/self.temperature
       labels = torch.arange(cos_sim.size(0)).long().to(proj_1.device)       
       loss = self.loss_fct(cos_sim, labels)

       return loss
     
     
     
