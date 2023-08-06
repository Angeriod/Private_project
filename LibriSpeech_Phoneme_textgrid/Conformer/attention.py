import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .embedding import PositionalEncoding
from .wrapper import Linear

class RelativeMultiAttention(nn.Module):
    def __init__(self,d_model=512, num_heads=16, dropout_p=0.1):
        super(RelativeMultiAttention,self).__init__()
        self.d_model = d_model
        self.d_head = int(d_model / num_heads) #32
        self.num_heads = num_heads
        self.sqrt_dim = math.sqrt(d_model)

        self.query_proj = Linear(d_model,d_model)# num_heads(16) divide query, key, value into 16 another subspace
        self.key_proj = Linear(d_model,d_model) # (batch_size, seq_length, num_heads, d_head)
        self.value_proj = Linear(d_model,d_model)
        self.pos_proj = Linear(d_model,d_model, bias=False) # used as fixed constant

        self.dropout = nn.Dropout(p=dropout_p)
        self.u_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head)) # shape: (num_heads, d_head) 
        self.v_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        torch.nn.init.xavier_uniform_(self.u_bias)# 각각 초기화
        torch.nn.init.xavier_uniform_(self.v_bias)

        self.out_proj = Linear(d_model, d_model)

    def forward(self,query,key,value,pos_embedding,mask):
        batch_size = value.size(0)

        query = self.query_proj(query).view(batch_size,-1,self.num_heads,self.d_head) 
        #(batch_size, seq_length, num_heads, d_head)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head).permute(0,2,1,3)
        #(batch_size, num_heads, seq_length, d_head)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head).permute(0,2,1,3)
        #(batch_size, num_heads, seq_length, d_head)
        pos_embedding = self.pos_proj(pos_embedding).view(batch_size,-1,self.num_heads, self.d_head)
        #(batch_size, seq_length, num_heads, d_head)

        content_score = torch.matmul((query + self.u_bias).transpose(1, 2), key.transpose(2,3))
        #  (batch_size, seq_length, num_heads, d_head):broadcasting-> (batch_size, num_heads,seq_length, d_head)* 
        # (batch_size, num_heads, d_head, seq_length) ==> (batch_size, num_heads, seq_length, seq_length)
        pos_score = torch.matmul((query + self.v_bias).transpose(1,2), pos_embedding.permute(0,2,3,1))
        # (batch_size, num_heads , seq_length, d_head) * (batch_size, num_heads, d_head, seq_length,)
        # ==> (batch_size, num_heads, seq_length, seq_length)
        pos_score = self._relative_shift(pos_score)

        score = (content_score + pos_score) / self.sqrt_dim

        if mask is not None:
            mask = mask.unsqueeze(1)
            score.masked_fill(mask,-1e9)
        
        attn = F.softmax(score, -1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, value).transpose(1 ,2)
        context = context.contiguous().view(batch_size, -1 , self.d_model)

        return self.out_proj(context)
    
    def _relative_shift(self, pos_score):
        batch_size, num_heads, seq_length1, seq_length2 = pos_score.size()
        zeros = pos_score.new_zeros(batch_size, num_heads, seq_length1, 1)
        padded_pos_score = torch.cat([zeros, pos_score], dim=-1) #  (batch_size, num_heads, seq_length1, seq_length2 + 1)

        padded_pos_score = padded_pos_score.view(batch_size, num_heads, seq_length2 + 1, seq_length1)
        #  (batch_size, num_heads, seq_length2 + 1, seq_length1)
        pos_score = padded_pos_score[:,:,1:].view_as(pos_score)# eliminate zeors

        return pos_score


class MultiHeadedSelfAttentionModule(nn.Module):
    def __init__(self,d_model, num_heads, dropout_p):
        super(MultiHeadedSelfAttentionModule, self).__init__()
        self.positional_encoding = PositionalEncoding(d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.attention = RelativeMultiAttention(d_model, num_heads, dropout_p)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, inputs, mask=None):
        batch_size, seq_length, _ = inputs.size()
        pos_embedding = self.positional_encoding(seq_length)
        pos_embedding = pos_embedding.repeat(batch_size, 1, 1)

        inputs = self.layer_norm(inputs)
        outputs = self.attention(inputs, inputs, inputs, pos_embedding=pos_embedding, mask=mask)

        return self.dropout(outputs)
        # input (batch, time, dim)
        # mask (batch, 1, time2) or (batch, time1, time2)
        # output (batch, time ,dim)
