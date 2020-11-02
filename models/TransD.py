import os
import json
from numpy.core.fromnumeric import shape
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseModule(nn.Module):

    def __init__(self):
        super(BaseModule, self).__init__()
        self.zero_const = nn.Parameter(torch.Tensor([0]))
        self.zero_const.requires_grad = False
        self.pi_const = nn.Parameter(torch.Tensor([3.14159265358979323846]))
        self.pi_const.requires_grad = False

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(os.path.join(path)))
        self.eval()

    def save_checkpoint(self, path):
        torch.save(self.state_dict(), path)

    def load_parameters(self, path):
        f = open(path, "r")
        parameters = json.loads(f.read())
        f.close()
        for i in parameters:
            parameters[i] = torch.Tensor(parameters[i])
        self.load_state_dict(parameters, strict=False)
        self.eval()

    def save_parameters(self, path):
        f = open(path, "w")
        f.write(json.dumps(self.get_parameters("list")))
        f.close()

    def get_parameters(self, mode="numpy", param_dict=None):
        all_param_dict = self.state_dict()
        if param_dict == None:
            param_dict = all_param_dict.keys()
        res = {}
        for param in param_dict:
            if mode == "numpy":
                res[param] = all_param_dict[param].cpu().numpy()
            elif mode == "list":
                res[param] = all_param_dict[param].cpu().numpy().tolist()
            else:
                res[param] = all_param_dict[param]
        return res

    def set_parameters(self, parameters):
        for i in parameters:
            parameters[i] = torch.Tensor(parameters[i])
        self.load_state_dict(parameters, strict=False)
        self.eval()


class Model(BaseModule):

    def __init__(self, ent_tot, rel_tot):
        super(Model, self).__init__()
        self.ent_tot = ent_tot
        self.rel_tot = rel_tot

    def forward(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError


class TransD(Model):

    def __init__(self, ent_tot, rel_tot, dim_e=100, dim_r=100, p_norm=1, norm_flag=True, margin=None, epsilon=None):
        super(TransD, self).__init__(ent_tot, rel_tot)

        self.dim_e = dim_e
        self.dim_r = dim_r
        self.margin = margin
        self.epsilon = epsilon
        self.norm_flag = norm_flag
        self.p_norm = p_norm

        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim_r)
        self.ent_transfer = nn.Embedding(self.ent_tot, self.dim_e)
        self.rel_transfer = nn.Embedding(self.rel_tot, self.dim_r)

        if margin == None or epsilon == None:
            nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
            nn.init.xavier_uniform_(self.ent_transfer.weight.data)
            nn.init.xavier_uniform_(self.rel_transfer.weight.data)
        else:
            self.ent_embedding_range = nn.Parameter(
                torch.Tensor([(self.margin + self.epsilon) / self.dim_e]), requires_grad=False
            )
            self.rel_embedding_range = nn.Parameter(
                torch.Tensor([(self.margin + self.epsilon) / self.dim_r]), requires_grad=False
            )
            nn.init.uniform_(
                tensor=self.rel_embeddings.weight.data,
                a=-self.rel_embedding_range.item(),
                b=self.rel_embedding_range.item()
            )
            nn.init.uniform_(
                tensor=self.ent_transfer.weight.data,
                a=-self.ent_embedding_range.item(),
                b=self.ent_embedding_range.item()
            )
            nn.init.uniform_(
                tensor=self.rel_transfer.weight.data,
                a=-self.rel_embedding_range.item(),
                b=self.rel_embedding_range.item()
            )
        if margin != None:
            self.margin = nn.Parameter(torch.Tensor([margin]))
            self.margin.requires_grad = False
            self.margin_flag = True
        else:
            self.margin_flag = False

        self.loss = MarginLoss(margin=margin)

    def _resize(self, tensor, axis, size):
        shape = tensor.size()
        osize = shape[axis]
        if osize == size:
            return tensor
        if (osize > size):
            return torch.narrow(tensor, axis, 0, size)
        paddings = []
        for i in range(len(shape)):
            if i == axis:
                paddings = [0, size - osize] + paddings
            else:
                paddings = [0, 0] + paddings
        print(paddings)
        return F.pad(tensor, paddings=paddings, mode="constant", value=0)

    def _calc(self, h, t, r, mask, mode):
        if self.norm_flag:
            h = F.normalize(h, 2, -1)
            r = F.normalize(r, 2, -1)
            t = F.normalize(t, 2, -1)
        if mode != 'normal':
            h = h.view(-1, r.shape[0], h.shape[-1])
            t = t.view(-1, r.shape[0], t.shape[-1])
            r = r.view(-1, r.shape[0], r.shape[-1])
        if mode == 'head_batch':
            score = h + (r - t)
        else:
            score = (h + r) - t
        score = torch.norm(score, self.p_norm, -1).flatten()     # 3维变2维
        return score

    def _transfer(self, e, e_transfer, r_transfer):
        if e.shape[0] != r_transfer.shape[0]:
            e = e.view(-1, r_transfer.shape[0], e.shape[-1])
            e_transfer = e_transfer.view(-1, r_transfer.shape[0], e_transfer.shape[-1])
            r_transfer = r_transfer.view(-1, r_transfer.shape[0], r_transfer.shape[-1])
            e = F.normalize(
                self._resize(e, -1, r_transfer.size()[-1]) + torch.sum(e * e_transfer, -1, True) * r_transfer,
                p=2,
                dim=-1
            )
            return e.view(-1, e.shape[-1])
        else:
            return F.normalize(
                self._resize(e, -1, r_transfer.size()[-1]) + torch.sum(e * e_transfer, -1, True) * r_transfer,
                p=2,
                dim=-1
            )

    def forward(self, emb_h, emb_t, batch_h, batch_t, batch_r, mask):
        mode = "head_batch"
        r = self.rel_embeddings(batch_r)
        h_transfer = self.ent_transfer(batch_h)
        t_transfer = self.ent_transfer(batch_t)
        r_transfer = self.rel_transfer(batch_r)
        h = self._transfer(emb_h, h_transfer, r_transfer)   # 投影的过程
        t = self._transfer(emb_t, t_transfer, r_transfer)
        score = self._calc(h, t, r, mask, mode)
        if self.margin_flag:
            score = self.margin - score

        p_score = self._get_positive_score(score, mask)
        n_score = self._get_negative_score(score, mask)
        if p_score.shape[0] > n_score.shape[0]:
            n_score = torch.cat([n_score, n_score.mean().expand((p_score.shape[0] - n_score.shape[0],))], dim=0)
        elif n_score.shape[0] > p_score.shape[0]:
            p_score = torch.cat([p_score, p_score.mean().expand((n_score.shape[0] - p_score.shape[0],))], dim=0)
        loss_res = self.loss(p_score, n_score)

        return loss_res

    def _get_positive_score(self, score, mask):
        positive_score = score[mask.view(-1)]
        # positive_score = positive_score.view(-1, self.batch_size).permute(1, 0)
        return positive_score

    def _get_negative_score(self, score, mask):
        negative_score = score[~mask.view(-1)]
        # negative_score = negative_score.view(-1, self.batch_size).permute(1, 0)
        return negative_score

    def decode(self, emb_h, emb_t, batch_h, batch_t, batch_r, mask):
        mode = "head_batch"
        r = self.rel_embeddings(batch_r)
        h_transfer = self.ent_transfer(batch_h)
        t_transfer = self.ent_transfer(batch_t)
        r_transfer = self.rel_transfer(batch_r)
        h = self._transfer(emb_h, h_transfer, r_transfer)   # 投影的过程
        t = self._transfer(emb_t, t_transfer, r_transfer)
        score = self._calc(h, t, r, mask, mode)
        if self.margin_flag:
            score = self.margin - score
        p_score = self._get_positive_score(score, mask)
        n_score = self._get_negative_score(score, mask)
        if p_score.shape[0] > n_score.shape[0]:
            n_score = torch.cat([n_score, n_score.mean().expand((p_score.shape[0] - n_score.shape[0],))], dim=0)
        elif n_score.shape[0] > p_score.shape[0]:
            p_score = torch.cat([p_score, p_score.mean().expand((n_score.shape[0] - p_score.shape[0],))], dim=0)
        score = self.loss.predict(p_score, n_score)
        return score


class Loss(BaseModule):

    def __init__(self):
        super(Loss, self).__init__()


class MarginLoss(Loss):

    def __init__(self, adv_temperature=None, margin=6.0):
        super(MarginLoss, self).__init__()
        self.margin = nn.Parameter(torch.Tensor([margin]))
        self.margin.requires_grad = False
        if adv_temperature != None:
            self.adv_temperature = nn.Parameter(torch.Tensor([adv_temperature]))
            self.adv_temperature.requires_grad = False
            self.adv_flag = True
        else:
            self.adv_flag = False

    def get_weights(self, n_score):
        return F.softmax(-n_score * self.adv_temperature, dim=-1).detach()

    def forward(self, p_score, n_score):
        if self.adv_flag:
            return (self.get_weights(n_score) * torch.max(p_score - n_score, -self.margin)).sum(dim=-1).mean() + self.margin
        else:
            return (torch.max(p_score - n_score, -self.margin)).mean() + self.margin

    def predict(self, p_score, n_score):
        score = self.forward(p_score, n_score)
        return score.cpu().data.numpy()
