import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from model.SGRAF import SGRAF


def KL(alpha, c):
    beta = torch.ones((1, c)).cuda()
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl


def mse_loss(label, alpha, c, lambda2):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    m = alpha / S
    A = torch.sum((label - m) ** 2, dim=1, keepdim=True)
    B = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
    alp = E * (1 - label) + 1
    C = lambda2 * KL(alp, c)
    return (A + B) + C


class DECL(nn.Module):
    def __init__(self, opt):
        super(DECL, self).__init__()
        self.opt = opt
        self.grad_clip = opt.grad_clip
        self.lambda1 = opt.lambda1
        self.lambda2 = opt.lambda2
        self.similarity_model = SGRAF(opt)
        self.mu = opt.mu
        self.params = list(self.similarity_model.params)
        self.optimizer = torch.optim.Adam(self.params, lr=opt.learning_rate)
        self.step = 0

    def state_dict(self):
        return self.similarity_model.state_dict()

    def load_state_dict(self, state_dict):
        self.similarity_model.load_state_dict(state_dict)

    def train_start(self):
        """switch to train mode"""
        self.similarity_model.train_start()

    def val_start(self):
        """switch to valuate mode"""
        self.similarity_model.val_start()

    def get_alpha(self, images, captions, lengths):
        img_embs, cap_embs, cap_lens = self.similarity_model.forward_emb(images, captions, lengths)
        sims, evidences, sims_tanh = self.similarity_model.forward_sim(img_embs, cap_embs, cap_lens, 'not sims')
        sum_e = evidences + evidences.t()
        norm_e = sum_e / torch.sum(sum_e, dim=1, keepdim=True)
        alpha_i2t = evidences + 1
        alpha_t2i = evidences.t() + 1
        return alpha_i2t, alpha_t2i, norm_e, sims_tanh, sims

    def RDH_loss(self, scores, neg=None):
        if neg is None:
            neg = self.mu
        margin = self.opt.margin
        diagonal = scores.diag().view(scores.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        cost_s = (margin + scores - d1).clamp(min=0)
        cost_im = (margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        mask = mask.to(cost_s.device)
        cost_s, cost_im = cost_s.masked_fill_(mask, 0), cost_im.masked_fill_(mask, 0)

        top_neg_row = torch.topk(cost_s, k=neg, dim=1).values
        top_neg_column = torch.topk(cost_im.t(), k=neg, dim=1).values
        return (top_neg_row.sum(dim=1) + top_neg_column.sum(dim=1)) / neg  # (K,1)

    def warmup_batch(self, images, captions, lengths):
        self.step += 1
        batch_length = images.size(0)
        neg = max(int(self.opt.batch_size - self.opt.eta * self.step), self.mu)
        if batch_length < neg:
            neg = batch_length - 1
        alpha_i2t, alpha_t2i, _, sims_tanh, _ = self.get_alpha(images, captions, lengths)
        self.optimizer.zero_grad()
        batch_labels = torch.eye(batch_length).cuda().long()
        loss_edl = mse_loss(batch_labels, alpha_i2t, batch_length, self.lambda2)
        loss_edl += mse_loss(batch_labels, alpha_t2i, batch_length, self.lambda2)
        loss_edl = torch.mean(loss_edl)

        loss_rdh = self.RDH_loss(sims_tanh, neg=neg)
        loss_rdh = loss_rdh.sum() * self.lambda1

        loss = loss_edl + loss_rdh
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()
        self.logger.update('Step', self.step)
        self.logger.update('H_n', neg)  # The hardness
        self.logger.update('Lr', self.optimizer.param_groups[0]['lr'])
        self.logger.update('Loss_edl', loss_edl.item(), batch_length)
        self.logger.update('Loss_rdh', loss_rdh.item(), batch_length)
        self.logger.update('Loss', loss.item(), batch_length)

    def train_batch(self, images, captions, lengths, preds):
        self.step += 1
        neg = max(int(self.opt.batch_size - self.opt.eta * self.step), self.mu)
        batch_length = images.size(0)
        if batch_length < neg:
            neg = batch_length - 1
        alpha_i2t, alpha_t2i, _, sims_tanh, _ = self.get_alpha(images, captions, lengths)
        self.optimizer.zero_grad()
        preds = preds.cuda()
        self.optimizer.zero_grad()
        batch_labels = torch.eye(batch_length)
        n_idx = (1 - preds).nonzero().view(1, -1)[0].tolist()
        c_idx = preds.nonzero().view(1, -1)[0].tolist()
        for i in n_idx:
            batch_labels[i][i] = 0
        batch_labels = batch_labels.cuda().long()
        loss_edl = mse_loss(batch_labels, alpha_i2t, batch_length, self.lambda2)
        loss_edl += mse_loss(batch_labels, alpha_t2i, batch_length, self.lambda2)
        loss_edl = torch.mean(loss_edl)
        if len(c_idx) == 0:
            loss_rdh = torch.tensor(0).cuda()
        else:
            loss_rdh = self.RDH_loss(sims_tanh, neg=neg)
            loss_rdh = loss_rdh[c_idx]
            loss_rdh = loss_rdh.sum() * self.lambda1
        loss = loss_edl + loss_rdh
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()
        self.logger.update('Step', self.step)
        self.logger.update('H_n', neg)  # The hardness
        self.logger.update('Lr', self.optimizer.param_groups[0]['lr'])
        self.logger.update('Loss_edl', loss_edl.item(), batch_length)
        self.logger.update('Loss_rdh', loss_rdh.item(), batch_length)
        self.logger.update('Loss', loss.item(), batch_length)
