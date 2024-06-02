import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from attention import Attention, NewAttention
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier, SimpleLinearNet
from fc import FCNet, MLP
from adver import Generator, Discriminator
from torch.nn import functional as F
import numpy as np

eps = 1e-3


def mask_softmax(x, mask):
    mask = mask.unsqueeze(2).float()
    x2 = torch.exp(x - torch.max(x))
    x3 = x2 * mask
    epsilon = 1e-5
    x3_sum = torch.sum(x3, dim=1, keepdim=True) + epsilon
    x4 = x3 / x3_sum.expand_as(x3)
    return x4


class BaseModel(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, classifier, c_1, c_2):
        super(BaseModel, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier
        self.debias_loss_fn = None
        # self.bias_scale = torch.nn.Parameter(torch.from_numpy(np.ones((1, ), dtype=np.float32)*1.2))
        self.bias_lin = torch.nn.Linear(1024, 1)
        self.generate = Generator()
        self.discriminate = Discriminator()
        self.linearnet1 = SimpleLinearNet()
        self.linearnet2 = SimpleLinearNet()
        self.c_1 = c_1
        self.c_2 = c_2
        self.coor_loss = nn.CrossEntropyLoss()

    def forward(self, v, s, q, labels, bias, v_mask):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        w_emb = self.w_emb(q)
        q_emb, q_hidden = self.q_emb(w_emb)  # [batch, q_dim]

        att = self.v_att(v, q_emb)
        att_2 = 1 - att
        att_new = self.generate(att)
        logit_pro, joint_repr_pro, att_pro = self.discriminate(att_new, v, q_emb)
        logit_pro_2, joint_repr_pro_2, att_pro_2 = self.discriminate(att, v, q_emb)
        if v_mask is None:
            att = nn.functional.softmax(att, 1)
            att_2 = nn.functional.softmax(att_2, 1)
        else:
            att = mask_softmax(att, v_mask)

        v_emb = (att * v).sum(1)  # [batch, v_dim]
        v_emb_2 = (att_2 * v).sum(1)

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        v_repr_2 = self.v_net(v_emb_2)

        joint_repr = v_repr * q_repr
        joint_repr_2 = v_repr_2 * q_repr
        logits = self.classifier(joint_repr)
        logits_2 = self.classifier(joint_repr_2)

        q_pred = self.c_1(q_emb.detach())

        q_out = self.c_2(q_pred)

        if labels is not None:
            loss = self.debias_loss_fn(joint_repr, logits, bias, labels)
            y_gradient = 2 * labels * torch.sigmoid(-2 * labels * bias)
            loss_q = F.binary_cross_entropy_with_logits(q_out, y_gradient)
            # ref_logits = F.softmax(torch.sigmoid(q_pred) + bias, dim=-1)
            ref_logits = torch.sigmoid(q_pred) + bias
            y_gradient = 2 * labels * torch.sigmoid(-2 * labels * ref_logits)
            w_1 = self.linearnet1(att.sum(2))
            w_2 = self.linearnet2(att_new.sum(2))
            e_x = torch.sigmoid(q_pred) + bias

            r_1 = w_1 / (e_x + eps) + (1 - w_1) / (1 - e_x + eps)
            r_2 = w_2 / (e_x + eps) + (1 - w_2) / (1 - e_x + eps)
            # ate = r_1 * ((w_1 / (e_x + eps) * logits) - ((1 - w_1) / (1 - e_x + eps) * logits)) \
            #       - r_2 * (w_2 / (e_x + eps) * logits_2 - (1 - w_2) / (1 - e_x + eps) * logits_2)
            # ate = (logits - logits_2) / (e_x + eps)
            ate = (logits / (e_x+eps) - logits_2 / (1-e_x+eps))
            ate = nn.functional.softmax(ate, 1)
            kl_loss = labels * ate
            kl_loss = kl_loss.sum(1).mean()

            loss_att = F.kl_div(F.log_softmax(att_2), F.softmax(att), reduction='mean')
            # loss_att_2 = F.kl_div(F.log_softmax(att_new), F.softmax(att), reduction='mean')
            # loss_logit2 = F.binary_cross_entropy_with_logits(logits_2, y_gradient)
            # loss_logit2_pro = self.debias_loss_fn(joint_repr, logits_2, bias, labels)
            # loss_dill = self.diloss(logit_pro.float(), logits.float(), labels)
            loss_3 = F.binary_cross_entropy_with_logits(logit_pro, y_gradient)
            loss_4 = F.binary_cross_entropy_with_logits(logit_pro_2, y_gradient)
            loss_5 = F.binary_cross_entropy_with_logits(logits_2, y_gradient)
            loss_6 = F.binary_cross_entropy_with_logits(logits_2, labels)
            loss += F.binary_cross_entropy_with_logits(logits, y_gradient) \
                    + loss_q + loss_att + loss_3 + loss_4 + loss_5 + loss_6 - kl_loss \
                    + F.binary_cross_entropy_with_logits(ate, y_gradient)
            return logits, loss, w_emb, att, att_2, loss_att
        else:
            loss = None
            return logits, loss, w_emb, att, att_2
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
    classifier = SimpleClassifier(
        num_hid, num_hid * 2, dataset.num_ans_candidates, 0.1)
    c_1 = MLP(input_dim=q_emb.num_hid, dimensions=[1024, 1024, dataset.num_ans_candidates])
    c_2 = nn.Linear(dataset.num_ans_candidates, dataset.num_ans_candidates)
    return BaseModel(w_emb, q_emb, v_att, q_net, v_net, classifier, c_1, c_2)
