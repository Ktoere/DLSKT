# @File    : DLSKT.py
# @Software: PyCharm


import math
from os import times

import torch
from torch import nn
from torch.nn.init import xavier_uniform_, constant_
import torch.nn.functional as F
from enum import IntEnum
import numpy as np
import copy
import math
from model.Long_termmodel import TransformerModel
import torch.nn.init as init


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class DLSKTnet(nn.Module):
    def __init__(self, Exercise_size, Concept_size, embedding_dim,time,interval,config,dataset_cof):
        super(DLSKTnet, self).__init__()

        self.Exercise_size = Exercise_size
        self.dropout = config["dropout"]
        self.final_fc_dim = config["final_fc_dim"]
        # self.sequence_last_m = config["sequence_last_m"]

        self.emb_dropout = nn.Dropout(self.dropout)
        self.user_short_nh = config["user_short_nh"]
        self.user_short_nv = config["user_short_nv"]
        self.transformer_encoder_layers = config["transformer_encoder_layers"]
        self.transformer_encoder_heads = config["transformer_encoder_heads"]
        self.hidden_dim = config["input_dim"]
        self.transformer_encoder_dim_feedforward = config["transformer_encoder_dim_feedforward"]
        self.transformer_encoder_layer_norm_eps = config["transformer_encoder_layer_norm_eps"]
        self.input_dim = config["input_dim"]
        self.final_fc_dim = config["final_fc_dim"]
        self.d_model = self.input_dim
        self.seq_max_length = config["max_seq_length"]
        self.window_size = config["window_size"]



        self.interval = interval
        self.time = time
        self.embedding_dim = embedding_dim
        # 嵌入层
        self.exercie_embed = nn.Embedding(Exercise_size + 2, embedding_dim)
        self.concept_embed = nn.Embedding(Concept_size + 1, embedding_dim)
        self.difficult_param = nn.Embedding(Exercise_size + 1, 1)
        self.a_embed = nn.Embedding(2, embedding_dim)

        self.time_embed = nn.Embedding(self.time + 2, embedding_dim, padding_idx=self.time + 1)
        self.attemptCount_embed = nn.Embedding(dataset_cof["attemptCount"] + 2, embedding_dim)
        self.hintCount_embed = nn.Embedding(dataset_cof["hintCount"] + 2, embedding_dim)
        self.inteveltime_encoder = nn.Embedding(self.interval + 10, embedding_dim)



        self.decoder_map = nn.Linear(self.d_model * 3, self.d_model)
        self.qa_liea1 = nn.Linear(embedding_dim  + embedding_dim, embedding_dim)
        self.qa_behav_map = nn.Linear(self.d_model * 2, self.d_model)
        self.relu = nn.ReLU()


        self.mulatte = TransformerModel(self.input_dim,self.interval,self.seq_max_length)



        # self.W3 = nn.Linear(1, self.window_size)
        self.W4 = nn.Linear(self.input_dim , self.input_dim)
        self.W5 = nn.Linear(self.input_dim *2 , 1)
        self.norm = nn.LayerNorm(self.input_dim)
        self.norm1 = nn.LayerNorm(self.input_dim)

        self.user_long_map = nn.Linear(self.input_dim, self.input_dim, bias=False)
        self.user_short_map = nn.Linear(self.input_dim, self.input_dim)
        # self.user_interest_fusion = nn.Linear(self.input_dim * 2, self.input_dim)

        self.dropout1 = nn.Dropout(self.dropout)
        self.dropout2 = nn.Dropout(self.dropout)
        self.dropout_miss = nn.Dropout(self.dropout)



        # 2.2 gating mechanism
        self.alpha_mlp = nn.Sequential(
            nn.Linear(self.d_model * 2 , self.d_model),
            nn.ReLU(),nn.Dropout(self.dropout),
            nn.Linear(self.d_model, 1)
            # nn.Sigmoid()
        )
        # 2.3 linear transformation after concatenation
        self.fusion_layer = nn.Linear(self.d_model * 2 , self.d_model)

        # 2.4 attention mechanism
        self.attention_fuse = AttentionFusion(self.d_model)



        self.mlp2 = nn.Sequential(
            nn.Linear(self.d_model , self.final_fc_dim),
            nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(self.final_fc_dim, 256),
            nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(256, self.d_model))

        # self.contact = nn.Linear(self.d_model * 2 , self.d_model )
        self.gru = nn.GRU(self.d_model, self.d_model, batch_first=True)


        self.mlp = nn.Sequential(
            nn.Linear(self.d_model + self.d_model, self.final_fc_dim),
            nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(self.final_fc_dim, 256),
            nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(256, 1))



    def _init_weights(self):
        for embed in [
            self.exercie_embed,
            self.concept_embed,
            self.difficult_param,
            self.attemptCount_embed,
            self.hintCount_embed,
            self.a_embed,
            self.time_embed
        ]:

            init.normal_(embed.weight.data, mean=0.0, std=0.1)  # 可调整 std


            if embed.padding_idx is not None:
                embed.weight.data[embed.padding_idx].zero_()







    def forward(self,  exercise_seq, concept_seq, response_seq,attemptCount_seq,hintCount_seq,taken_time_seq, interval_time_seq):

        exercise_embed = self.exercie_embed(exercise_seq)
        concept_embed = self.concept_embed(concept_seq)
        pid_embed_data = self.difficult_param(exercise_seq)
        anser_embed_data = self.a_embed(response_seq)
        in_dotime = self.time_embed(taken_time_seq)
        attemptCount_embed = self.attemptCount_embed(attemptCount_seq)
        hintCount_embed = self.hintCount_embed(hintCount_seq)
        # iterv_embed_data = self.inteveltime_encoder(interval_time_seq)


        # qa_embed_data = concept_embed + anser_embed_data
        q_embed_data = concept_embed + pid_embed_data * exercise_embed
        qa_embed_data = q_embed_data + anser_embed_data
        # qa_embed_data = self.qa_liea1(torch.cat([qa_embed_data, in_dotime], dim=-1))
        behavior_embed_data = self.decoder_map(torch.cat((in_dotime,attemptCount_embed, hintCount_embed), dim=-1))
        qa_embed_data = self.qa_behav_map(torch.cat((qa_embed_data, behavior_embed_data), dim=-1))
        qa_embed_data = self.relu(qa_embed_data)



        # 1. long-term knowledge state
        user_long_output,fix_seq = self.sequence_info_extract(
            item_seq = qa_embed_data,
            interval = interval_time_seq,
            mode='user_long'
        )

        # 2. short-term knowledge state
        user_short_output, sim  = self.user_short_interest_extract(
            item_seq = fix_seq,
            long_state=user_long_output
        )






        # decoupling
        mi_loss = self.decoupling_loss(user_long_output , user_short_output,concept_embed[:,:-1,:],sim)
        # knowledge distillation
        distillation_loss = self.distillation_loss(user_long_output, user_short_output,q_embed_data,sim )
        mean_value = torch.mean(sim)
        distillation_loss = mean_value * distillation_loss

        knowledge_state = self.norm(user_long_output + user_short_output)

        concat_q = torch.cat([knowledge_state , q_embed_data[:, 1:, :]], dim=-1)
        output = self.mlp(concat_q)
        x = torch.sigmoid(output)

        return x.squeeze(-1),distillation_loss,mi_loss


    def sequence_info_extract(self, item_seq,interval,  mode='user_long'):

        output,seq = self.mulatte(item_seq[:,:-1,:],interval[:,:-1])
        return output,seq


    def user_short_interest_extract(self, item_seq, long_state):

        pre_k_longstate = self.sliding_window_average(long_state, self.window_size)

        # batch_size, length, proj_dim = long_state.shape
        # shifted = torch.roll(long_state, shifts=1, dims=1)
        # shifted[:, 0, :] = long_state[:, 0, :]

        # sim = torch.cosine_similarity(long_state, pre_k_longstate, dim=-1)
        sim = F.pairwise_distance(long_state, pre_k_longstate, p=2)


        # chi_t = (1-torch.sigmoid(sim.unsqueeze(-1)))
        # max, idx = torch.max(chi_t, dim=0)
        # mapped_value = self.window_size * (torch.exp(chi_t) ) #/ (torch.exp(max) - 1)

        k = 1
        exp_neg_alpha = (torch.exp(-sim.unsqueeze(-1)))
        sigmoid_term = 1 / (k + exp_neg_alpha)
        mapped_value = (1 - sigmoid_term) * self.window_size
        l_t = torch.round(mapped_value).long()
        l_t = torch.clamp(l_t, min=1)
        l_t = l_t.squeeze(-1)


        output,avg = self.process_sequence(item_seq,l_t)

        output = self.W4(output)
        output1 = torch.sigmoid(output)
        # long_short_sim = F.pairwise_distance(long_state, torch.sigmoid(output), p=2)
        # long_short_sim = torch.sigmoid(long_short_sim)

        long_short_sim = torch.cosine_similarity(long_state, output1, dim=-1)
        # long_short_sim = torch.sigmoid(long_short_sim)
        return output,long_short_sim.unsqueeze(-1)



