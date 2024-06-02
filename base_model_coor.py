import torch
import torch.nn as nn
from attention import Attention, NewAttention
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier, VisualEncoder, SimpleClassifierlxmert
from fc import FCNet, MLP
import numpy as np
from torch.nn import functional as F
from transformers import LxmertModel
from vqa_debias_loss_functions import Distillation_Loss


def mask_softmax(x, mask):
    mask = mask.unsqueeze(2).float()
    x2 = torch.exp(x - torch.max(x))
    x3 = x2 * mask
    epsilon = 1e-5
    x3_sum = torch.sum(x3, dim=1, keepdim=True) + epsilon
    x4 = x3 / x3_sum.expand_as(x3)
    return x4


class BaseModel(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, v_net_2, classifier, c_q, c_v, visual_encoder):
        super(BaseModel, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.v_net_2 = v_net_2
        self.classifier = classifier
        self.debias_loss_fn = None
        self.c_q = c_q
        self.c_v = c_v
        # self.bias_scale = torch.nn.Parameter(torch.from_numpy(np.ones((1, ), dtype=np.float32)*1.2))
        self.bias_lin = torch.nn.Linear(1024, 1)
        self.diloss = Distillation_Loss(1.5, 0.2)
        self.visual_encoder = visual_encoder
        self.coor_loss = nn.CrossEntropyLoss()

    def get_negative_mask(self, batch_size):
        negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
        for i in range(batch_size):
            negative_mask[i, i] = 0
            negative_mask[i, i + batch_size] = 0

        negative_mask = torch.cat((negative_mask, negative_mask), 0)
        return negative_mask

    def criterion(self, out_1, out_2, tau_plus=0.1, batch_size=64, beta=1.0, estimator='easy', temperature=0.5):
        # neg score
        out = torch.cat([out_1, out_2], dim=0)
        neg = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        old_neg = neg.clone()
        mask = self.get_negative_mask(batch_size).cuda()
        neg = neg.masked_select(mask).view(2 * batch_size, -1)

        # pos score
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        pos = torch.cat([pos, pos], dim=0)

        # negative samples similarity scoring
        if estimator == 'hard':
            N = batch_size * 2 - 2
            imp = (beta * neg.log()).exp()
            reweight_neg = (imp * neg).sum(dim=-1) / imp.mean(dim=-1)
            Ng = (-tau_plus * N * pos + reweight_neg) / (1 - tau_plus)
            # constrain (optional)
            Ng = torch.clamp(Ng, min=N * np.e ** (-1 / temperature))
        elif estimator == 'easy':
            Ng = neg.sum(dim=-1)
        else:
            raise Exception('Invalid estimator selected. Please use any of [hard, easy]')
        # contrastive loss
        loss = (- torch.log(pos / (pos + Ng))).mean()
        return loss

    def forward(self, v, s, q, labels, bias, v_mask, coor_data, mode):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        w_emb = self.w_emb(q)
        q_emb, q_hidden = self.q_emb(w_emb)  # [batch, q_dim]
        visn_feats = self.visual_encoder(v, s)
        att = self.v_att(v, q_emb)
        if v_mask is None:
            att = nn.functional.softmax(att, 1)
        else:
            att = mask_softmax(att, v_mask)

        v_emb = (att * v).sum(1)  # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        norm_repr_ori = F.normalize(joint_repr, dim=-1)
        norm_repr_positive_v = F.normalize(v_repr, dim=-1)
        norm_repr_positive_q = F.normalize(q_repr, dim=-1)
        # norm_repr_ori = joint_repr
        # norm_repr_positive_v = v_repr
        # norm_repr_positive_q = q_repr
        # q_coor = self.c_q(q_repr)
        # v_coor = self.c_q(v_repr)
        # logit_new = q_coor * v_coor
        logits = self.classifier(joint_repr)
        q_pred_1 = self.c_q(q_emb)
        v_pred_1 = self.c_v(v_emb)
        if mode == 'first':
            q_coor = self.c_q(q_emb)
            v_coor = self.c_v(v_emb)
            # loss_q = self.coor_loss(q_coor, labels)
            # loss_v = self.coor_loss(v_coor, labels)
            loss_q = F.binary_cross_entropy_with_logits(q_coor, labels)
            loss_v = F.binary_cross_entropy_with_logits(v_coor, labels)
            # loss = self.debias_loss_fn(joint_repr, logits, bias, labels)

            # return logits, loss, w_emb, loss_q, loss_v
            return loss_q, loss_v
        else:
            if labels is not None:
                cl_loss_v = self.criterion(norm_repr_ori, norm_repr_positive_v, batch_size=q.shape[0])
                cl_loss_q = self.criterion(norm_repr_ori, norm_repr_positive_q, batch_size=q.shape[0])
                ref_logits_q = torch.sigmoid(q_pred_1) + bias
                ref_logits_v = torch.sigmoid(v_pred_1) + bias
                y_gradient_q = 2 * labels * torch.sigmoid(-2 * labels * ref_logits_q)
                y_gradient_v = 2 * labels * torch.sigmoid(-2 * labels * ref_logits_v)
                if coor_data == 'q':
                    q_pred = self.q_net(q_repr)
                    coor_data = joint_repr - q_pred
                    joint_repr_2 = norm_repr_ori - norm_repr_positive_q
                    logits_new = self.classifier(coor_data)
                    logits_new_2 = self.classifier(joint_repr_2)
                    loss_new = self.debias_loss_fn(joint_repr, logits_new, bias, y_gradient_q + y_gradient_v)
                    loss_new_2 = self.debias_loss_fn(joint_repr, logits_new_2, bias, y_gradient_q + y_gradient_v)
                elif coor_data == 'v':
                    v_pred = self.v_net_2(v_repr)
                    coor_data = joint_repr - v_pred
                    joint_repr_2 = norm_repr_ori - norm_repr_positive_v
                    logits_new = self.classifier(coor_data)
                    logits_new_2 = self.classifier(joint_repr_2)
                    loss_new = self.debias_loss_fn(joint_repr, logits_new, bias, y_gradient_q + y_gradient_v)
                    loss_new_2 = self.debias_loss_fn(joint_repr, logits_new_2, bias, y_gradient_q + y_gradient_v)
                else:
                    # loss_1 = self.debias_loss_fn(joint_repr, logits, bias, labels)
                    loss = self.debias_loss_fn(joint_repr, logits, bias, y_gradient_q + y_gradient_v)
                    # loss_q = F.binary_cross_entropy_with_logits(q_pred_1, labels)
                    # loss_v = F.binary_cross_entropy_with_logits(v_pred_1, labels)
                    loss_q = self.debias_loss_fn(joint_repr, q_pred_1, bias, y_gradient_q + y_gradient_v)
                    loss_v = self.debias_loss_fn(joint_repr, v_pred_1, bias, y_gradient_q + y_gradient_v)
                    return logits, loss + loss_q + loss_v, w_emb

                # y_gradient = 2 * labels * torch.sigmoid(-2 * labels * bias)
                # loss_q = F.binary_cross_entropy_with_logits(q_pred_1, y_gradient)
                # # ref_logits = F.softmax(torch.sigmoid(q_pred) + bias, dim=-1)

                # loss_new = self.debias_loss_fn(joint_repr, logits_new, bias, y_gradient)
                # loss_new_2 = self.debias_loss_fn(joint_repr, logits_new_2, bias, y_gradient)
                # loss = self.debias_loss_fn(joint_repr, logits, bias, y_gradient) + loss_new
                # loss_q_1 = self.debias_loss_fn(joint_repr, q_pred_1, bias, y_gradient)
                # loss_v_1 = self.debias_loss_fn(joint_repr, v_pred_1, bias, y_gradient)
                # loss_1 = F.binary_cross_entropy_with_logits(logits, y_gradient)
                # loss += loss_q_1 + loss_v_1 + loss_1 + F.binary_cross_entropy_with_logits(logits_new, y_gradient) + \
                #         self.debias_loss_fn(joint_repr, logits, bias, y_gradient) + loss_q + loss_new_2

                loss = self.debias_loss_fn(joint_repr, logits, bias, y_gradient_q + y_gradient_v)
                # loss_q = F.binary_cross_entropy_with_logits(q_pred_1, labels)
                # loss_v = F.binary_cross_entropy_with_logits(v_pred_1, labels)
                loss_q = self.debias_loss_fn(joint_repr, q_pred_1, bias, y_gradient_q + y_gradient_v)
                loss_v = self.debias_loss_fn(joint_repr, v_pred_1, bias, y_gradient_q + y_gradient_v)

                total_loss = loss + loss_q + loss_v + loss_new + loss_new_2

                return logits, total_loss, w_emb
            else:
                loss = None
            return logits, loss, att
            # return logits, loss, att


def build_baseline0(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = Attention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, 2 * num_hid, dataset.num_ans_candidates, 0.5)
    return BaseModel(w_emb, q_emb, v_att, q_net, v_net, classifier)


def build_baseline0_newatt(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = NewAttention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([q_emb.num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    v_net_2 = FCNet([q_emb.num_hid, num_hid])
    # lxmert = LxmertModel.from_pretrained('/home/lqw/lqw_pytorch/Baseline_code/lxmert/pytorch_model.bin', return_dict=True)
    classifier = SimpleClassifier(
        num_hid, num_hid * 2, dataset.num_ans_candidates, 0.5)
    c_q = MLP(input_dim=q_emb.num_hid, dimensions=[1024, 1024, dataset.num_ans_candidates])
    c_v = MLP(input_dim=dataset.v_dim, dimensions=[1024, 1024, dataset.num_ans_candidates])
    visual_encoder = VisualEncoder(dataset.v_dim, num_hid, 0.1)
    return BaseModel(w_emb, q_emb, v_att, q_net, v_net, v_net_2, classifier, c_q, c_v, visual_encoder)