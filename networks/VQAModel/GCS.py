import torch
import torch.nn as nn
from torch.nn.modules import BatchNorm1d
import sys
sys.path.insert(0, 'networks')
from q_v_transformer import CoAttention_init as CoAttention
from gcn import AdjLearner, GCN, AdjLearner_intra, AdjLearner_inter
from block import fusions #pytorch >= 1.1.0
from mlp import MLP,AttFlat, AttFlat_nofc, LayerNorm
import numpy as np
import random

class GINNet(nn.Module):
    def __init__(self, vid_encoder, layer_num=3, e=1.0):
        super(GINNet, self).__init__()
        hidden_size = vid_encoder.dim_hidden
        input_dropout_p = vid_encoder.input_dropout_p
        self.layer_num = layer_num

        self.mlp_v_list = nn.ModuleList()
        for i in range(self.layer_num ):
            self.mlp_v_list.append(MLP(2, 256, 256, 256))

        self.mlp_t_list = nn.ModuleList()
        for i in range(self.layer_num ):
            self.mlp_t_list.append(MLP(2, 256, 256, 256))

        self.mlp_qv_list = nn.ModuleList()
        for i in range(self.layer_num ):
            self.mlp_qv_list.append(MLP(2, 256, 256, 256))

        self.e = e

        self.loss_l2 = torch.nn.MSELoss()
        self.adj_learner_v = AdjLearner_intra(
            hidden_size, hidden_size, dropout=input_dropout_p)
        self.adj_learner_t = AdjLearner_intra(
            hidden_size, hidden_size, dropout=input_dropout_p)
        self.adj_learner_qv = AdjLearner_intra(
            hidden_size, hidden_size, dropout=input_dropout_p)

    def adj_onehot(self, adj):
        avg_adj = torch.mean(adj) * torch.ones_like(adj)
        adj_onehot = (adj > avg_adj).long()  # the final 0-1 adj
        return adj_onehot


    def GIN_multiple_maxpool(self, X, type):
        X_list = []
        if type == 1:
            for i in range(self.layer_num):
                adj = self.adj_onehot(self.adj_learner_qv(X, X))
                I = torch.eye(adj.shape[0], adj.shape[1]).cuda()
                X = self.mlp_qv_list[i](torch.mm(adj + I * self.e, X))
                X_list.append(X)
        if type == 2:
            for i in range(self.layer_num):
                adj = self.adj_onehot(self.adj_learner_t(X, X))
                I = torch.eye(adj.shape[0], adj.shape[1]).cuda()
                X = self.mlp_t_list[i](torch.mm(adj + I * self.e, X))
                X_list.append(X)
        if type == 3:
            for i in range(self.layer_num):
                adj = self.adj_onehot(self.adj_learner_v(X, X))
                I = torch.eye(adj.shape[0], adj.shape[1]).cuda()
                X = self.mlp_v_list[i](torch.mm(adj + I * self.e, X))
                X_list.append(X)

        G_QV_mean = X_list[0]
        for X in X_list[1:]:
            G_QV_mean = G_QV_mean + X
        G_QV_mean = G_QV_mean / self.layer_num
        return G_QV_mean


    def GIN_NET_maxpool(self, QV, Q, V):

        GIN_QV = self.GIN_multiple_maxpool(QV, type=1)
        GIN_Q = self.GIN_multiple_maxpool(Q, type=2)
        GIN_V = self.GIN_multiple_maxpool(V, type=3)

        return GIN_QV, GIN_Q, GIN_V

    def GIN_NET_maxpool_A(self, A):
        GIN_A = self.GIN_multiple_maxpool(A, type=2)
        return GIN_A

    def forward(self, QV, Q, V, A):
        QV_, Q_, V_ = self.GIN_NET_maxpool(QV, Q, V)
        A_ = self.GIN_NET_maxpool_A(A)
        loss_graph = self.loss_l2(QV_, A_)
        loss_graph_Q = self.loss_l2(Q_, A_)
        loss_graph_V = self.loss_l2(V_, A_)

        return loss_graph, loss_graph_Q, loss_graph_V

class GraphCrossSampleDebias(nn.Module):
    def __init__(self, vid_encoder, qns_encoder, device, layer_num=3, e=1.0):
        """
        Graph-based Cross-Sample Debiasing Method (PAMI2023)
        :param vid_encoder:
        :param qns_encoder:
        :param device:
        """
        super(GraphCrossSampleDebias, self).__init__()
        self.vid_encoder = vid_encoder
        self.qns_encoder = qns_encoder
        self.device = device
        self.layer_num = layer_num
        self.e = e
        hidden_size = vid_encoder.dim_hidden
        input_dropout_p = vid_encoder.input_dropout_p

        self.q_input_ln = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.a_input_ln = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.v_input_ln = nn.LayerNorm(hidden_size, elementwise_affine=False)

        self.co_attn = CoAttention(
            hidden_size, n_layers=vid_encoder.n_layers, dropout_p=input_dropout_p)

        self.adj_learner = AdjLearner(
            hidden_size, hidden_size, dropout=input_dropout_p)

        self.gcn = GCN(
            hidden_size,
            hidden_size,
            hidden_size,
            num_layers=2,
            dropout=input_dropout_p)

        self.gcn_atten_pool = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
            nn.Softmax(dim=-1))

        self.global_fusion = fusions.Block(
            [hidden_size, hidden_size], hidden_size, dropout_input=input_dropout_p)

        self.fusion = fusions.Block([hidden_size, hidden_size], 1)
        self.attFlat_query = AttFlat(hidden_size=256, flat_mlp_size=256, flat_out_size=256)
        self.attFlat_image = AttFlat(hidden_size=256, flat_mlp_size=256, flat_out_size=256)
        self.attFlat_option = AttFlat_nofc(hidden_size=256, flat_mlp_size=256, flat_out_size=256)
        self.final_mlp = torch.nn.Sequential(
            torch.nn.Linear(256 * 2, 256),
            torch.nn.ReLU(inplace=True),
        )
        self.final_BN = torch.nn.Sequential(
            BatchNorm1d(256)
        )
        self.fusion_BN = torch.nn.Sequential(
            BatchNorm1d(256)
        )
        self.final_mlp_linear = torch.nn.Sequential(
            torch.nn.Linear(256, 1)
        )
        self.BN_a = torch.nn.Sequential(
            BatchNorm1d(256)
        )
        self.BN_q = torch.nn.Sequential(
            BatchNorm1d(256)
        )
        self.BN_image = BatchNorm1d(256)
        self.proj_norm = LayerNorm(size=256)

        self.GIN = GINNet(self.vid_encoder, layer_num=self.layer_num, e=self.e)

    def find_cate(self, idx, cate_tensor):
        cate_tensor = cate_tensor.numpy()
        this_cate = set(cate_tensor[idx].tolist())
        length_list = []
        for i in range(cate_tensor.shape[0]):
            if i != idx:
                now_cate = set(cate_tensor[i].tolist())
                length_list.append(len(this_cate & now_cate))
        idx_find = length_list.index(max(length_list))
        return idx_find

    def generate_mask(self, max_len, bs, length):
        # length: [bs]
        q_arange = torch.arange(0, max_len).unsqueeze(0).expand(bs,max_len)
        q_lengths_ = length.unsqueeze(1).expand_as(q_arange)
        q_mask = (q_arange < q_lengths_).long().cuda()
        return q_mask

    def find_cate_new(self, idx, cate_tensor):
        cate_tensor = cate_tensor.numpy()
        this_cate = set(cate_tensor[idx].tolist())
        length_list = []
        length_list_idx = []
        for i in range(cate_tensor.shape[0]):
            if i != idx:
                now_cate = set(cate_tensor[i].tolist())
                length_list.append(len(this_cate & now_cate))
                length_list_idx.append(i)
        max_find = max(length_list)
        idx_find = []
        for idx, a in enumerate(length_list):
            if a == max_find:
                idx_find.append(length_list_idx[idx])

        idx_find_final = idx_find[random.randint(0, len(idx_find) - 1)]
        return idx_find_final

    def forward(self, vid_feats, qas, qas_lengths, candidate_as, a_lengths , candidate_qs, q_lengths, ans_targets, cate_tensor, epoch, cate_flag, isTrain=True):
        """
        :param vid_feats:
        :param qns:
        :param qns_lengths:
        :param mode:
        : ans_targets [bs]
        :return:
        """
        if self.qns_encoder.use_bert:
            cand_qas = qas.permute(1, 0, 2, 3)  # for BERT
            candidate_as = candidate_as.permute(1, 0, 2, 3)  # [5, bs, max_len, dim]
            candidate_qs = candidate_qs.permute(1, 0, 2, 3)
        else:
            cand_qas = qas.permute(1, 0, 2)
            candidate_as = candidate_as.permute(1,0,2)  #[5, bs, max_len]
            candidate_qs = candidate_qs.permute(1,0,2) #[5, bs, max_len]

        cand_len = qas_lengths.permute(1, 0)
        a_lengths = a_lengths.permute(1,0)
        q_lengths = q_lengths.permute(1,0)

        app_feat = vid_feats[:,:,:2048]
        mot_feat = vid_feats[:,:,2048:]

        new_vid_feats = torch.cat([app_feat, mot_feat], dim=2)

        v_output, v_hidden = self.vid_encoder(new_vid_feats)
        v_last_hidden = torch.squeeze(v_hidden)

        q_mask = self.generate_mask(candidate_qs[0].shape[1], candidate_qs[0].shape[0], q_lengths[0])

        out = []
        a_rep = []
        for idx, a in enumerate(candidate_as):
            q_output, s_hidden = self.qns_encoder(candidate_qs[idx], q_lengths[idx])
            q_output = self.q_input_ln(q_output)
            a_output, s_hidden = self.qns_encoder(a, a_lengths[idx])
            a_output = self.a_input_ln(a_output)

            encoder_out, fusion_qv, a_rep_bn, q_rep_bn, v_rep_bn = self.multimodal_fusion(v_output, q_output, a_output, q_lengths[idx], a_lengths[idx])
            out.append(encoder_out)
            a_rep.append(a_rep_bn)
        out = torch.stack(out, 0).transpose(1, 0) # [bs, 5, 1]
        _, predict_idx = torch.max(out, 1)

        a_rep = torch.stack(a_rep, 0).transpose(1, 0)  #[bs, 5, 256]
        a_rep_gt = []
        for idx, a_ in enumerate(a_rep):
            a_rep_gt.append(a_[ans_targets[idx]])
        a_rep_gt = torch.stack(a_rep_gt, 0) #[bs, 256]

        loss_graph, loss_graph_q, loss_graph_v = self.GIN(fusion_qv, q_rep_bn, v_rep_bn, a_rep_gt)

        margin = min(0.2 + 0.1*(epoch-1), 2)
        final_loss = loss_graph + max(0, loss_graph - loss_graph_q + margin) + max(0, loss_graph -loss_graph_v + margin)

        # new add
        if isTrain:
            candidate_as_new = candidate_as.clone()
            a_lengths_new = a_lengths.clone()
            cand_len_new = cand_len.clone()
            #
            # for each sample in the batchsize
            for idx in range(candidate_qs.shape[1]):
                if cate_flag[idx] == 1:  # not the head class
                    rand_idx = self.find_cate_new(idx, cate_tensor)
                    ans_new = candidate_as[:, rand_idx, :]
                    j = 0
                    flag = 0
                    for i in range(5):
                        if i == ans_targets[idx]:  # the ground-truth
                            pass
                        else:
                            if j == ans_targets[rand_idx]:  # the ground-truth ans of rand_ans
                                j += 1
                            candidate_as_new[i, idx, :] = ans_new[j]
                            a_lengths_new[i, idx] = a_lengths[j, idx]
                            flag += 1
                            j += 1
                    assert flag == 4

            out_new = []
            a_rep_new = []
            for idx, a in enumerate(candidate_as_new):
                a_output_new, s_hidden = self.qns_encoder(a, a_lengths_new[idx])
                a_output_new = self.a_input_ln(a_output_new)

                encoder_out_new, fusion_qv_new, a_rep_bn_new, q_rep_bn_new, v_rep_bnv = self.multimodal_fusion(v_output, q_output, a_output_new, q_lengths[idx],
                                                         a_lengths_new[idx])
                out_new.append(encoder_out_new)
                a_rep_new.append(a_rep_bn_new)
            out_new = torch.stack(out_new, 0).transpose(1, 0)  # [bs, 5, 1]
            _, predict_idx_new = torch.max(out_new, 1)

            a_rep_new = torch.stack(a_rep_new, 0).transpose(1, 0)  # [bs, 5, 256]
            a_rep_gt_new = []
            for idx, a_ in enumerate(a_rep_new):
                a_rep_gt_new.append(a_[ans_targets[idx]])
            a_rep_gt_new = torch.stack(a_rep_gt_new, 0)  # [bs, 256]

            loss_graph_new, loss_graph_q_new, loss_graph_v_new = self.GIN(fusion_qv, q_rep_bn, v_rep_bn, a_rep_gt_new)
            margin_new = min(0.2 + 0.1 * (epoch - 1), 2)
            final_loss_new = loss_graph_new + max(0, loss_graph_new - loss_graph_q_new + margin_new) + max(0,
                                                                                       loss_graph_new - loss_graph_v_new + margin_new)

            return out, predict_idx, final_loss, (loss_graph, loss_graph_q, loss_graph_v), out_new, predict_idx_new, final_loss_new
        else:
            return out, predict_idx, final_loss, (loss_graph, loss_graph_q, loss_graph_v)


    def multimodal_fusion(self, v_output, q_output, a_output, q_lengths, a_lengths):
        """
        :param vid_feats:
        :param qas:
        :param qas_lengths:
        :return:
        """

        # co-attention
        # q_output, v_output_q, _, _ = self.co_attn(q_output, v_output)
        # a_output, v_output_a, _, _ = self.co_attn(a_output, v_output)

        # self-attention
        v_mask = torch.ones(v_output.shape[0], v_output.shape[1]).long().cuda()
        v_rep = self.attFlat_image(v_output, v_mask)
        v_rep_bn = self.BN_image(v_rep)

        q_mask = self.generate_mask(q_output.shape[1], q_output.shape[0], q_lengths)
        q_rep = self.attFlat_query(q_output, q_mask)
        q_rep_bn = self.BN_q(q_rep)

        a_mask = self.generate_mask(a_output.shape[1], a_output.shape[0], a_lengths)
        a_rep = self.attFlat_option(a_output, a_mask)
        a_rep_bn = self.BN_a(a_rep)

        fusion_qv = self.fusion_QV(v_rep_bn, q_rep_bn)

        query_option_image_cat = torch.cat((a_rep_bn, fusion_qv), -1)
        query_option_image_cat = self.final_mlp(query_option_image_cat)
        query_option_image_cat = self.final_BN(query_option_image_cat)
        logits = self.final_mlp_linear(query_option_image_cat).squeeze(1)

        return logits, fusion_qv, a_rep_bn, q_rep_bn, v_rep_bn

    def vq_encoder(self, v_output, v_last_hidden, qas, qas_lengths):
        """
        :param vid_feats:
        :param qas:
        :param qas_lengths:
        :return:
        """
        q_output, s_hidden = self.qns_encoder(qas, qas_lengths)
        qns_last_hidden = torch.squeeze(s_hidden)

        q_output = self.q_input_ln(q_output)
        v_output = self.v_input_ln(v_output)

        q_output, v_output, _, _ = self.co_attn(q_output, v_output)

        ### GCN
        adj = self.adj_learner(q_output, v_output)
        q_v_inputs = torch.cat((q_output, v_output), dim=1)
        q_v_output = self.gcn(q_v_inputs, adj)

        ## attention pool
        local_attn = self.gcn_atten_pool(q_v_output)
        local_out = torch.sum(q_v_output * local_attn, dim=1)

        global_out = self.global_fusion((qns_last_hidden, v_last_hidden)) #[bs, 256], [bs, 256]
        out = self.fusion((global_out, local_out)).squeeze() #[bs, 256], [bs, 256]

        return out

    def fusion_QV(self, q, v):
        # q: [bs,4,512]
        # v: [bs, 512]
        fusion_qv = v + q  # element-wise add

        batch_size = q.shape[0]
        num_options = q.shape[1]
        fusion_qv = self.proj_norm(fusion_qv)
        fusion_qv = self.fusion_BN(fusion_qv)
        return fusion_qv
