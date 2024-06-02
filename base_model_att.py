import torch
import torch.nn as nn
from attention import Attention, NewAttention, SelfAttention, CrossAttention
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier, VisualEncoder
from fc import FCNet, MLP
import numpy as np
from torch.nn import functional as F
from vqa_debias_loss_functions import LearnedMixin

num_cross_layers = 2


def mask_softmax(x, mask):
    mask = mask.unsqueeze(2).float()
    x2 = torch.exp(x - torch.max(x))
    x3 = x2 * mask
    epsilon = 1e-5
    x3_sum = torch.sum(x3, dim=1, keepdim=True) + epsilon
    x4 = x3 / x3_sum.expand_as(x3)
    return x4


class BaseModel(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, classifier, c_1, c_2,
                 visual_encoder, cross_att):
        super(BaseModel, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier
        self.debias_loss_fn = None
        self.bias_lin = torch.nn.Linear(1024, 1)
        self.c_1 = c_1
        self.c_2 = c_2
        self.vision_lin = torch.nn.Linear(1024, 1)
        self.question_lin = torch.nn.Linear(1024, 1)
        # self.bce = nn.BCELoss()
        self.visual_encoder = visual_encoder
        self.cross_att_layers = nn.ModuleList(
            [cross_att for _ in range(num_cross_layers)]
        )
        self.dense = nn.Linear(1024, 1024)
        self.activation = nn.Tanh()

    def forward(self, v, s, q, labels, bias, v_mask):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        w_emb = self.w_emb(q)
        q_emb, q_hidden = self.q_emb(w_emb)  # [batch, q_dim]
        visn_feats = self.visual_encoder(v, s)
        # lang_feats, visn_feats = self.cross_att_layers[0](q_hidden, visn_feats, "self")
        lang_feats, visn_feats = self.cross_att_layers[0](q_hidden, visn_feats)

        for layer_module in self.cross_att_layers:
            lang_feats, visn_feats = layer_module(lang_feats, visn_feats)
            first_token_feat = lang_feats[:, 0].unsqueeze(1).repeat(1, 36, 1)
            visn_feats = visn_feats + first_token_feat
        # for layer_module in self.cross_att_layers:
        #     lang_feats, visn_feats = layer_module(q_hidden, visn_feat)
        #
        #

        # att_new = self.v_att(visn_feats, lang_feats.sum(1))
        att = self.v_att(v, q_emb)

        if v_mask is None:
            att = nn.functional.softmax(att, 1)
            # att_new = nn.functional.softmax(att_new, 1)
        else:
            att = mask_softmax(att, v_mask)

        v_emb = (att * v).sum(1)  # [batch, v_dim]
        # v_emb_new = (att_new * v).sum(1)  # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        # v_repr = self.v_net(v_emb) + visn_feats.sum(1)
        v_repr = self.v_net(v_emb)
        # v_repr_new = self.v_net(v_emb_new)

        joint_repr = v_repr * q_repr
        # joint_repr_new = v_repr_new * q_repr
        # joint_repr = joint_repr + lang_feats[:, 0]
        logit = lang_feats[:, 0]
        # logit = self.dense(logit)
        # logit = self.activation(logit)
        logit = self.classifier(logit)
        # joint_repr = self.dense(joint_repr)
        # joint_repr = self.activation(joint_repr)
        logits = self.classifier(joint_repr)
        # logits_new = self.classifier(joint_repr_new)
        logits = logits + logit
        q_pred = self.c_1(q_emb.detach())
        v_pred = self.c_1(v_repr.detach())

        q_out = self.c_2(q_pred)

        if labels is not None:
            y_gradient = 2 * labels * torch.sigmoid(-2 * labels * bias)
            loss_q = F.binary_cross_entropy_with_logits(q_out, y_gradient)
            ref_logtis = torch.sigmoid(q_pred) + bias
            y_gradient = 2 * labels * torch.sigmoid(-2 * labels * ref_logtis)
            loss_att = F.binary_cross_entropy_with_logits(logits, y_gradient)
            # loss_att_new = F.binary_cross_entropy_with_logits(logits_new, y_gradient)
            # loss_att_2 = F.binary_cross_entropy_with_logits(logit, y_gradient)
            loss = self.debias_loss_fn(joint_repr, logits, bias, labels) + loss_att + loss_q
            # logits = logits + logit
            # loss += self.debias_loss_fn(joint_repr, logits, bias, labels)
                   # + self.debias_loss_fn(lang_feats[:, 0], logit, bias, labels) + loss_att_2
            return logits, loss, w_emb
        else:
            loss = None
            return logits, loss, w_emb
            # return logits, loss, w_emb, att, v_pred




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
    # v_att = NewAttention(dataset.v_dim, q_emb.num_hid, num_hid)
    v_att = NewAttention(dataset.v_dim, q_emb.num_hid, num_hid)
    # v_att_2 = NewAttention(dataset.v_dim, q_emb.num_hid, num_hid)
    v_att_2 = NewAttention(num_hid, q_emb.num_hid, num_hid)
    q_att = SelfAttention(q_emb.num_hid, num_hid)
    q_net = FCNet([q_emb.num_hid, num_hid])
    q_net_2 = FCNet([q_emb.num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, num_hid * 2, dataset.num_ans_candidates, 0.1)

    c_1 = MLP(input_dim=q_emb.num_hid, dimensions=[1024, 1024, dataset.num_ans_candidates])
    c_2 = nn.Linear(dataset.num_ans_candidates, dataset.num_ans_candidates)
    visual_encoder = VisualEncoder(dataset.v_dim, num_hid, 0.1)
    cross_att = CrossAttention(num_hid, dropout=0.1)
    return BaseModel(w_emb, q_emb, v_att, q_net, v_net, classifier, c_1, c_2, visual_encoder,
                     cross_att)
