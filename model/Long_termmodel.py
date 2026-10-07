import torch
from torch import nn
from torch.nn.init import xavier_uniform_, constant_
import torch.nn.functional as F
from enum import IntEnum
import numpy as np
import copy
import math



class TransformerModel(nn.Module):
    def __init__(self,  d_model, n_it, max_len, nhid=2048, nlayers=1, dropout=0.2):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.max_len = max_len
        self.d_model = d_model
        self.final_fc_dim = 512
        self.dropout = nn.Dropout(dropout)


        # 位置编码和时间间隔编码
        self.pos_encoder = nn.Embedding(self.max_len, d_model)
        self.inteveltime_encoder = nn.Embedding(n_it + 10, d_model)
        self.time_pos_lgt_linear = nn.Linear(d_model * 2, d_model)
        self.time_pos_lgt_linear = nn.Linear(self.d_model + self.d_model, self.d_model)


        self.qa_behavior_linear1 = nn.Linear(d_model * 2, 1)
        self.qa_behavior_linear2 = nn.Linear(d_model * 2, d_model)
        self.relu = nn.ReLU()
        # self.qa_behavior_linear3 = nn.Linear(d_model * 4, d_model)

        # self.fusion_layer = nn.Linear(d_model * 2, 1)
        # self.fusion_layer2 = nn.Linear(d_model * 3, d_model)
        # self.layer_norm1 = nn.LayerNorm(d_model)
        #
        #
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size= d_model ,
            num_layers=1,
            batch_first=True,
            # bidirectional=True,
            # dropout=dropout if nlayers > 1 else 0
        )
        self.gru_linear = nn.Linear(200,d_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=4,
            dropout=dropout
        )

        self.encoder = TransformerEncoder(
            embed_dim=d_model,
            num_heads=8,
            ff_dim=512,
            num_layers=3
        )



        self.dropout1 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, src, inter_time):
        batch_size,length, dim = src.size()

        positions = torch.arange(0, src.size(1), device=src.device).unsqueeze(0)
        positions = positions.repeat(batch_size, 1).long()
        position_embeddings = self.pos_encoder(positions)

        interval_time_embeddings = self.inteveltime_encoder(inter_time)
        time_pos_lgt = self.time_pos_lgt_linear(torch.cat((position_embeddings,interval_time_embeddings),dim=-1))
        time_pos_lgt1 = torch.sigmoid(time_pos_lgt)

        # src1 = time_pos_lgt + src
        # src2 = self.dropout1(src)
        src2 = src

        mask = torch.triu(
            torch.ones(src2.size(1), src2.size(1), device=src2.device),
            diagonal=1).bool()
        encoder_output = self.encoder(src2, mask)
        encoder_output = torch.sigmoid(encoder_output)


        gru_input =  self.dropout3(self.layer_norm1(time_pos_lgt1 * encoder_output  + src) )

        gpu_output, _ = self.gru(gru_input)
        # gpu_output = self.gru_linear(gpu_output)
        gpu_output = torch.sigmoid(gpu_output)


        # # gpu_output=gru_input
        #
        # gpu_output = self.layer_norm2(
        #     self.dropout2(gpu_output)
        # )


        # gpu_output = self.dropout2(gpu_output)
        attention_int = gpu_output.transpose(0, 1)
        # attention_int = self.dropout3(attention_int)
        attn_output, _ = self.self_attn(
            attention_int,
            attention_int,
            attention_int,
            attn_mask=mask
        )
        # attn_output =  attn_output.transpose(0, 1)
        attn_output =  torch.sigmoid(attn_output.transpose(0, 1))


        concient = self.qa_behavior_linear1(torch.cat((gpu_output,attn_output), dim=-1))
        concient = torch.sigmoid(concient)
        short_input = self.layer_norm2(concient * src)
        # short_input = concient * src

        return gpu_output,short_input






class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        # 前馈网络
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embed_dim)
        # 层归一化
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # 自注意力层

        src2 = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )[0]

        # 残差连接 + 层归一化
        # src = src + self.dropout1(src2)
        # src = self.norm1(src)

        # 前馈网络
        src2 = self.linear2(self.dropout2(F.relu(self.linear1(src2))))
        # 残差连接 + 层归一化
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        output = src
        for layer in self.layers:
            output = layer(
                output,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask
            )
        return output






