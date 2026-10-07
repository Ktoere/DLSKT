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
        # self.user_short_nh = config["user_short_nh"]
        # self.user_short_nv = config["user_short_nv"]
        # self.transformer_encoder_layers = config["transformer_encoder_layers"]
        # self.transformer_encoder_heads = config["transformer_encoder_heads"]
        self.hidden_dim = config["input_dim"]
        # self.transformer_encoder_dim_feedforward = config["transformer_encoder_dim_feedforward"]
        # self.transformer_encoder_layer_norm_eps = config["transformer_encoder_layer_norm_eps"]
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
        # self.attention_fuse = AttentionFusion(self.d_model)



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
        mi_loss = self.contrastive_loss(user_long_output , user_short_output,concept_embed[:,:-1,:],sim)
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


    # 短期信息
    def user_short_interest_extract(self, item_seq, long_state):



        pre_k_longstate = self.sliding_window_average(long_state, self.window_size)
        sim = F.pairwise_distance(long_state, pre_k_longstate, p=2)

        # sim = torch.sigmoid(sim)
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



    def sliding_window_average(self,x, k):

        B, L, D = x.size()
        cum_sum = torch.cumsum(x, dim=1)  # 计算时间维度的累积和

        pad = torch.zeros(B, k, D, dtype=x.dtype, device=x.device)

        fix_x  = torch.cat([ pad,x], dim=1)

        later_sum = torch.cumsum(fix_x, dim=1)
        sum = cum_sum[:,:-1,:] - later_sum[:,:L-1,:]
        pad = torch.zeros(B, 1, D, dtype=x.dtype, device=x.device)

        sum = torch.cat([pad,sum], dim=1)

        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        positions = positions.repeat(x.size(0), 1)
        output_tensor = torch.clamp(positions, max=k)
        output_tensor[:,0] = 1
        avg = sum/output_tensor.unsqueeze(-1)
        # avg = torch.sigmoid(avg)

        return avg

    def avg_embedding(self, a,b):

        batch_size, length, dim = a.shape
        device = a.device

        # 计算累积和，并在前面补零
        cum_sum = torch.cat([
            torch.zeros(batch_size, 1, dim, device=device),
            a.cumsum(dim=1)
        ], dim=1)  # 形状 [batch_size, length+1, dim]

        # 生成位置索引矩阵 [batch_size, length]
        positions = torch.arange(length, device=device).unsqueeze(0).expand(batch_size, -1)

        # 计算每个位置的起始索引s，并进行截断处理
        s = positions - b + 1
        s = torch.clamp(s, min=0)

        # 结束索引为i+1
        end_indices = positions + 1

        # 收集起始和结束位置对应的累积和
        # 扩展索引以适应gather函数的维度
        start_indices_expanded = s.unsqueeze(-1).expand(-1, -1, dim)
        start_cum_sum = torch.gather(cum_sum, dim=1, index=start_indices_expanded)

        end_indices_expanded = end_indices.unsqueeze(-1).expand(-1, -1, dim)
        end_cum_sum = torch.gather(cum_sum, dim=1, index=end_indices_expanded)

        # 计算窗口总和和窗口长度
        window_sum = end_cum_sum - start_cum_sum
        window_length = end_indices - s  # 形状 [batch_size, length]

        # 转换为浮点数并扩展维度以进行除法
        window_length_float = window_length.unsqueeze(-1).float()

        # 计算平均值
        result = window_sum / window_length_float

        return result


    def process_sequence(self,a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:


        """
        使用注意力机制对窗口数据进行融合，避免数据泄露
        Args:
            a: [batch_size, length, dim] 输入序列
            b: [batch_size, length] 每个位置的窗口大小，指示当前位置可以关注前面多少个元素（包括自身）
        Returns:
            [batch_size, length, dim] 融合后的序列
        """

        avg = self.avg_embedding(a,b)

        batch_size, length, dim = a.shape
        device = a.device

        # 生成位置索引矩阵 [batch_size, length]
        i_indices = torch.arange(length, device=device).unsqueeze(0).expand(batch_size, -1)

        # 计算每个位置的窗口起始索引
        # start_i = max(0, i - (k_i - 1))，其中k_i = b[:, i]
        start_i = i_indices - (b - 1)
        start_i = torch.clamp(start_i, min=0)

        # 创建j坐标矩阵 [batch_size, length, length]
        j_indices = torch.arange(length, device=device).view(1, 1, -1).expand(batch_size, length, -1)

        # 创建i坐标矩阵 [batch_size, length, length]
        i_expanded = i_indices.unsqueeze(-1).expand(-1, -1, length)

        # 扩展start_i到三维 [batch_size, length, length]
        start_expanded = start_i.unsqueeze(-1).expand(-1, -1, length)

        # 计算有效掩码 (j >= start_i) & (j <= i)
        mask = (j_indices >= start_expanded) & (j_indices <= i_expanded)

        # 计算缩放点积注意力
        Q, K, V = a, a, a  # 使用原始输入作为Q,K,V

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-1, -2)) / (dim ** 0.5)

        # 使用掩码屏蔽无效位置
        scores = scores.masked_fill(~mask, float('-inf'))

        # 计算注意力权重
        attn_weights = F.softmax(scores, dim=-1)

        # 应用注意力权重到值向量
        output = torch.matmul(attn_weights, V)

        return output,avg




    def contrastive_loss(self,long_term_info, short_term_info,concept_seq,similary1):
        """
        计算长期和短期信息的对比损失(优化版本-避免梯度爆炸和负值)
        Args:
            long_term_info: 长期信息 [batch_size, T, dim]
            short_term_info: 短期信息 [batch_size, T, dim]
        Returns:
            对比损失值
        """

        batch_size, seq_len, dim = long_term_info.shape

        long_term_info =   similary1 * long_term_info
        short_term_info = similary1 * short_term_info
        # 1. 归一化输入向量 (在计算相似度前)
        # long_term_info = F.normalize(long_term_info, p=2, dim=-1)
        # short_term_info = F.normalize(short_term_info, p=2, dim=-1)

        # long_term_info = similary1 * long_term_info

        # long_term_info = self.dropout1(self.vector_projection(long_term_info,concept_seq))
        # short_term_info = self.dropout1(self.vector_projection(short_term_info,concept_seq))

        # long_term_info = self.vector_projection(long_term_info, concept_seq)
        # short_term_info = self.vector_projection(short_term_info, concept_seq)

        cos = F.cosine_similarity(long_term_info,short_term_info,dim=-1)



        # 正样本

        #1.余弦相似度
        # pos_term = F.cosine_similarity(long_term_info, short_term_info, dim=2)


        #2. 杰卡德相关系数
        x_bool = long_term_info  > 0.3
        y_bool = short_term_info > 0.1

        # x_bool = long_term_info  > cos.unsqueeze(-1) * short_term_info
        # y_bool = short_term_info > cos.unsqueeze(-1) * long_term_info



        # 计算交集和并集
        intersection = torch.sum(x_bool & y_bool, dim=2)
        union = torch.sum(x_bool | y_bool, dim=2)
        # 计算杰卡德相似系数
        pos_term = intersection / union
        pos_term = torch.log(pos_term + 1e-8)


        # 负样本
        long_tan = long_term_info.transpose(0, 1)
        short_tan = short_term_info.transpose(0, 1)

        cos1 = F.cosine_similarity(long_tan,short_tan, dim=-1)

        # 2. 杰卡德相关系数
        long_tan_binary = (long_tan > 0.3).float()  # 或者使用其他阈值
        short_tan_binary = (short_tan > 0.1).float()

        # long_tan_binary = (long_tan  > cos1.unsqueeze(-1) * short_tan).float()  # 或者使用其他阈值
        # short_tan_binary = (short_tan > cos1.unsqueeze(-1) * long_tan).float()



        # 计算交集
        intersection = torch.bmm(
            long_tan_binary,  # (L,B1,D)
            short_tan_binary.transpose(1, 2)  # (L,B1,D) -> (L,D,B1)
        )  # 结果形状: (L,B1,B1)
        # 计算并集
        # 首先计算长度和
        long_sum = long_tan_binary.sum(dim=2, keepdim=True)  # (L,B1,1)
        short_sum = short_tan_binary.sum(dim=2, keepdim=True)  # (L,B1,1)
        # 并集大小 = 长度和 - 交集
        union = long_sum + short_sum.transpose(1, 2) - intersection  # (L,B1,B1)
        # 计算杰卡德相似系数
        # 添加小的epsilon值避免除零错误
        epsilon = 1e-10
        similarity = intersection / (union + epsilon)  # 结果形状: (L,B1,B1)

        mask = torch.ones_like(similarity)
        mask = mask - torch.eye(batch_size, device=similarity.device).unsqueeze(0)  # 创建对角线为0的掩码并扩展到L维度

        # 应用掩码
        similarity = similarity * mask

        neg_term = torch.sum(similarity, dim=2)/(batch_size-1)
        neg_term = neg_term.transpose(0, 1)
        neg_term = torch.log(neg_term + 1e-8)

        # dist = torch.sum(similary1.squeeze(-1) * (pos_term - neg_term), dim=-1)
        dist = similary1.squeeze(-1) * (pos_term - neg_term)

        # 添加正交约束损失以增强特征解耦
        # orthogonal_loss = torch.mean(similary1)

        # 组合损失



        loss = torch.relu(torch.mean(dist))

        total_loss = loss

        return total_loss




        # # (可选) 如果需要在特征层面加入dropout
        # # long_term_info = self.dropout_miss(long_term_info)
        # # short_term_info = self.dropout_miss(short_term_info)
        #
        # batch_size, T, dim = long_term_info.shape
        #
        # # 归一化输入向量,避免数值过大
        # # long_term_info = F.normalize(long_term_info, p=2, dim=-1)
        # # short_term_info = F.normalize(short_term_info, p=2, dim=-1)
        #
        # # 计算L2距离矩阵 [B,T,T,D] -> [B,T,T]
        # # diff = torch.cdist(long_term_info, short_term_info, p=2)
        #
        #
        #
        #
        #
        # # 使用ReLU确保距离非负
        # # diff = F.relu(diff) + 1e-10
        #
        # # 计算相似度矩阵 [B,T,T]
        # sigma = 0.5
        # sigma = torch.tensor(sigma,device=diff.device)
        # sim_matrix = torch.exp(-torch.pow(diff, 2) / (2 * torch.pow(sigma, 2)))
        #
        # # sim_matrix = self.dropout1(sim_matrix)
        #
        # # 对相似度矩阵进行归一化,避免数值过大
        # # sim_matrix = sim_matrix / sim_matrix.sum(dim=-1, keepdim=True)
        #
        # # 对角线上的值为正样本对的相似度
        # pos_sim = torch.diagonal(sim_matrix, dim1=1, dim2=2)  # [B,T]
        # pos_sim = self.dropout1(pos_sim)
        # pos_term = torch.log(pos_sim + 1e-8)
        #
        # # 计算负样本项
        # long_tan = long_term_info.transpose(0, 1)
        # short_tan = long_term_info.transpose(0, 1)
        # # random_indices = torch.randperm(T)
        # # shuffled_tensor = short_term_info[random_indices]
        #
        #
        #
        # diff111 = torch.cdist(long_tan, short_tan, p=2)
        # # diff111 = F.relu(diff111) + 1e-10
        #
        # # 使用ReLU确保距离非负
        # # diff = F.relu(diff) + 1e-8
        #
        # # 计算相似度矩阵 [B,T,T]
        # sigma = 0.5
        # sigma = torch.tensor(sigma, device=diff.device)
        # sim_matrix111 = torch.exp(-torch.pow(diff111, 2) / (2 * torch.pow(sigma, 2)))
        # # sim_matrix111 = F.cosine_similarity(long_tan, short_tan, dim=-1)
        #
        #
        #
        # # mask = torch.eye(T).unsqueeze(0).to(sim_matrix.device)
        # neg_sim = sim_matrix111.transpose(0, 1)
        # neg_sim = self.dropout1(neg_sim)
        # neg_term = torch.log(neg_sim + 1e-8)
        # neg_term = torch.mean(neg_term, dim=2)
        #
        # # neg_term = torch.log(neg_sim.sum(dim=2) / (T - 1))
        #
        # # 使用ReLU确保loss非负
        # loss = torch.relu(torch.mean(pos_term - neg_term))

        # return loss

    def vector_projection(self,a, b):
        """
          计算一批向量 a 在一批向量 b 上的投影向量。
          假设 a 和 b 中的向量是一一对应的。

          参数:
            a (torch.Tensor): 要投影的向量批次，形状为 [batch_size, length, dim]。
            b (torch.Tensor): 投影到的目标向量批次，形状为 [batch_size, length, dim]。

          返回:
            torch.Tensor: 向量 a 在向量 b 上的投影向量批次，形状为 [batch_size, length, dim]。
                          如果 b 中的某个向量为零向量，则其对应的投影结果为零向量。
          """
        # 检查输入维度是否匹配
        if a.shape != b.shape:
            raise ValueError(f"输入张量 a 和 b 的形状必须相同，但得到 a: {a.shape}, b: {b.shape}")
        if a.ndim != 3:
            raise ValueError(f"输入张量应为3维 [batch_size, length, dim]，但得到: {a.ndim}")

        # 计算点积 a · b 沿着最后一个维度 (dim)
        # (B, L, D) * (B, L, D) -> (B, L, D) element-wise
        # sum over D -> (B, L)
        # keepdim=True -> (B, L, 1) for broadcasting
        dot_product = torch.sum(a * b, dim=-1, keepdim=True)

        # 计算向量 b 的模的平方 |b|^2 沿着最后一个维度 (dim)
        # (B, L, D) * (B, L, D) -> (B, L, D) element-wise
        # sum over D -> (B, L)
        # keepdim=True -> (B, L, 1) for broadcasting
        b_norm_sq = torch.sum(b * b, dim=-1, keepdim=True)

        # 计算标量部分 (a · b) / |b|^2
        # 为了处理 b_norm_sq 为 0 的情况（即 b 是零向量），我们使用 torch.nan_to_num
        # 如果 b_norm_sq 是 0：
        #   - 若 dot_product 也是 0 (例如 a=[1,1], b=[0,0]), 0/0 -> nan -> 0.0
        #   - 若 dot_product 非 0 (不可能，因为 b 是0), X/0 -> inf -> 0.0
        # 这样处理后，如果 b 是零向量，scalar_factor 会是 0。
        scalar_factor = torch.nan_to_num(dot_product / b_norm_sq, nan=0.0, posinf=0.0, neginf=0.0)

        # 计算投影向量: scalar_factor * b
        # (B, L, 1) * (B, L, D) -> (B, L, D) due to broadcasting
        projection_vector = scalar_factor * b


        return projection_vector







    def   distillation_loss(self, user_long_output, user_short_output,q_embed_data,sim,temperature=2.0):
        # user_long_output = self.vector_projection(user_long_output,concept_embed)
        # user_short_output = self.vector_projection(user_long_output,concept_embed)



        # p_long_term1 = F.log_softmax(sim * user_long_output.detach()/ temperature, dim=-1)
        # p_short_term1 = F.softmax(user_short_output/ temperature, dim=-1)
        #
        # loss1 = F.kl_div(p_long_term1, p_short_term1, reduction='batchmean') * (temperature ** 2)

        # p_long_term2 = F.log_softmax(long_output.detach() / temperature, dim=-1)
        # p_short_term2 = F.softmax(user_short_output / temperature, dim=-1)
        #
        # loss2 = F.kl_div(p_long_term2, p_short_term2, reduction='batchmean') * (temperature ** 2)
        # loss = (loss1 + loss2)/2

        # **非对称蒸馏损失**

        # f1 = F.normalize(sim * user_long_output, p=2, dim=-1)
        # f2 = F.normalize(sim * user_long_output + user_short_output, p=2, dim=-1)


        loss_st_to_lt1 = F.mse_loss( sim * user_long_output +  sim * user_short_output,   user_long_output )  # ST → LT 蒸馏损失
        # loss_st_to_lt = F.mse_loss(user_long_output, sim *  (user_short_output + user_long_output)) # ST → LT 蒸馏损失
        # cos_loss = 1 - torch.mean( sim.squeeze(-1) * F.cosine_similarity(sim * user_long_output, user_short_output, dim=-1))
        long_concat_q = torch.cat([user_long_output , q_embed_data[:, 1:, :]], dim=-1)
        long_output = self.mlp(long_concat_q)
        long_output = torch.sigmoid(long_output)

        short_concat_q = torch.cat([user_short_output, q_embed_data[:, 1:, :]], dim=-1)
        short_out = self.mlp(short_concat_q)
        short_out = torch.sigmoid(short_out)

        out_loss = F.mse_loss(long_output, short_out)



        #
        loss =   out_loss + loss_st_to_lt1



        # loss_st_to_lt = F.mse_loss((1-sim) *  long_output , short_output)  # ST → LT 蒸馏损失
        # loss_lt_to_st1 = F.mse_loss(long_output, sim * user_long_output)  # LT → ST 蒸馏损失


        return loss


        # p_long_term = F.softmax(long_term_logits, dim=-1)
        # p_short_term = F.softmax(short_term_logits, dim=-1)
        # return F.kl_div(p_short_term.log(), p_long_term, reduction='batchmean')










