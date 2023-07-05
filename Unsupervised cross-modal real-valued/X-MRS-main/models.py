import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.models as models
import torch.nn.functional as F
import numpy as np
import math
import random

from utils import load_torch_model, load_model_state

# attention layer used in instruction encoding
class AttentionLayer(nn.Module):
    def __init__(self, input_dim, with_attention):
        super(AttentionLayer, self).__init__()
        self.with_attention = with_attention

        self.u = torch.nn.Parameter(torch.Tensor(input_dim))  # u = [2*hid_dim]
        torch.nn.init.normal_(self.u, mean=0, std=0.01)
        self.u.requires_grad = True

        self.fc = nn.Linear(input_dim, input_dim)
        torch.nn.init.normal_(self.fc.weight, mean=0, std=0.01)

    def forward(self, x, y1=None):
        # x = [BS, max_len, 2*hid_dim]
        # a trick used to find the mask for the softmax
        mask = (x != 0)
        mask = mask[:, :, 0]
        if y1 is not None:
            mask = (y1 != 0)
        h = torch.tanh(self.fc(x))  # h = [BS, max_len, 2*hid_dim]
        if self.with_attention == 1:  # softmax
        #     scores = h @ self.u  # scores = [BS, max_len], unnormalized importance
        #     masked_scores = scores.masked_fill((1 - mask).byte(), -1e32)
        #     alpha = F.softmax(masked_scores, dim=1)  # alpha = [BS, max_len], normalized importance
        # elif self.with_attention == 2:  # Transformer
            scores = h @ self.u / math.sqrt(h.shape[-1])  # scores = [BS, max_len], unnormalized importance
            # masked_scores = scores.masked_fill((1 - mask).byte(), -1e32)
            masked_scores = scores.masked_fill((~mask), -1e32)
            alpha = F.softmax(masked_scores, dim=1)  # alpha = [BS, max_len], normalized importance

        alpha = alpha.unsqueeze(-1)  # alpha = [BS, max_len, 1]
        out = x * alpha  # out = [BS, max_len, 2*hid_dim]
        out = out.sum(dim=1)  # out = [BS, 2*hid_dim]
        return out, alpha.squeeze(-1)


def norm(input, p=2, dim=1, eps=1e-12):
    return input / input.norm(p, dim, keepdim=True).clamp(min=eps).expand_as(input)


# Attention layer
class visualAttLayer(nn.Module):
    def __init__(self, inSize, attSize=1024):
        super(visualAttLayer, self).__init__()
        self.context_vector_size = [attSize, 1]
        self.w_proj = nn.Linear(in_features=inSize, out_features=attSize)
        torch.nn.init.normal_(self.w_proj.weight, mean=0, std=0.01)

        self.w_context_vector = nn.Parameter(torch.randn(self.context_vector_size))
        torch.nn.init.normal_(self.w_context_vector, mean=0, std=0.01)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x_size = x.size
        x = x.view(x_size(0), x_size(1), -1)
        x = x.permute(0, 2, 1)  # transpose_(1,2)
        Hw = torch.tanh(self.w_proj(x))
        w_score = self.softmax(Hw.matmul(self.w_context_vector) / np.power(Hw.size(2), 0.5))

        x = x.mul(w_score* (x_size(2)*x_size(3)))
        x = x.permute(0, 2, 1)  # transpose_(1,2)
        x = x.view(x_size(0), x_size(1), x_size(2), x_size(3))

        return x, w_score


# Vision embedding, attenton on resnet
class visionMLP(nn.Module):
    def __init__(self, opts):
        super(visionMLP, self).__init__()
        self.opts = opts
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]  # we do not use the last fc layer.
        self.visionMLP = nn.Sequential(*modules)


    def forward(self, x):
        w_score = []
        x = self.visionMLP(x)
        return x, w_score


class textmBERT_fulltxt(nn.Module):
    def __init__(self, opts):
        super(textmBERT_fulltxt, self).__init__()
        from transformers import BertModel
        self.opts = opts
        # attention: Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length)
        modelA = BertModel.from_pretrained('bert-base-multilingual-cased', output_attentions=True)
        config = modelA.config
        config.num_hidden_layers = opts.BERT_layers
        config.num_attention_heads = opts.BERT_heads
        self.mBERT = BertModel(config)
        pretrained_dict = {k: v for k, v in modelA.state_dict().items() if 'embeddings' in k}
        self.mBERT.load_state_dict(pretrained_dict, strict=False)
        del modelA


    def forward(self, x, txt_embs=None, return_attention=False):
        if txt_embs is None:
            emb = self.mBERT(x, x>0)
        else:
            emb = txt_embs
       
        if return_attention:
            return emb[1], emb[2]
        return emb[1], None


class textmBERT_fulltxtAWE(nn.Module):
    def __init__(self, opts):
        super(textmBERT_fulltxtAWE, self).__init__()
        from transformers import BertModel

        self.opts = opts
        modelA = BertModel.from_pretrained('bert-base-multilingual-cased')
        self.word_embeddings = modelA.embeddings.word_embeddings
        del modelA
        self.atten_layer = AttentionLayer(768, True)


    def forward(self, x, txt_embs=None, return_attention=False):
        if txt_embs is None:
            emb = [self.word_embeddings(x)]
        else:
            emb = txt_embs

        mask = (x>0).unsqueeze(2).repeat(1,1,768).float()
        counts = (mask.sum(2)>0).sum(1)
        counts = counts.unsqueeze(1).repeat(1,768).float()
        emb = (emb[0] * mask).sum(1)/counts
        return emb, None


# Main FoodSpaceNet
class FoodSpaceNet(nn.Module):
    def __init__(self, opts):
        super(FoodSpaceNet, self).__init__()
        self.opts = opts
        
        self.visionMLP = visionMLP(opts)
        self.visual_embedding = nn.Sequential(
            nn.BatchNorm1d(2048,),
            nn.Linear(2048, 1024, 1024),
            nn.Tanh())
        self.align_img = nn.Sequential(
            nn.BatchNorm1d(opts.embDim),
            nn.Tanh())

        if opts.textmodel == 'mBERT_fulltxt':
            self.textMLP =  textmBERT_fulltxt(opts)
            self.recipe_embedding = nn.Sequential(
                nn.BatchNorm1d(768,),
                nn.Linear(768, 1024, 1024),
                nn.Tanh())
        elif opts.textmodel == 'mBERT_fulltxtAWE':
            self.textMLP =  textmBERT_fulltxtAWE(opts)
            self.recipe_embedding = nn.Sequential(
                nn.BatchNorm1d(768,),
                nn.Linear(768, 1024, 1024),
                nn.Tanh())
        else:
            raise NotImplementedError
        
        self.align_rec = nn.Sequential(
            nn.BatchNorm1d(opts.embDim),
            nn.Tanh()
        )


        self.align = nn.Sequential(
            nn.Linear(1024, opts.embDim),
        )


    def forward(self, input, opts, txt_embs=None, return_visual_attention=False, return_text_attention=None):  # we need to check how the input is going to be provided to the model
        if not opts.no_cuda:
            if txt_embs is None:
                for i in range(len(input)):
                    input[i] = input[i].cuda()
            else:
                input[0] = input[0].cuda()
                for i in range(len(txt_embs)):
                    txt_embs[i] = txt_embs[i].cuda()
        x, y = input


        w_score, attention = None, None
        # visual embedding
        visual_emb, w_score = self.visionMLP(x)
        visual_emb = visual_emb.view(visual_emb.size(0), -1)
        visual_emb = self.visual_embedding(visual_emb)
        visual_emb = self.align(visual_emb)

        visual_emb = self.align_img(visual_emb)
        visual_emb = norm(visual_emb)


        # recipe embedding
        recipe_emb, attention = self.textMLP(y, return_attention=return_text_attention)
        if type(recipe_emb) is tuple: # we are assuming that if this is a tuple it also contains attention of tokens
            alpha = recipe_emb[1]
            recipe_emb = recipe_emb[0]
        recipe_emb = self.recipe_embedding(recipe_emb)
        recipe_emb = self.align(recipe_emb)
        recipe_emb = self.align_rec(recipe_emb)
        recipe_emb = norm(recipe_emb)

        
        return [visual_emb, recipe_emb, w_score, attention]


# Main FoodSpaceImageEncoder
class FoodSpaceImageEncoder(nn.Module):
    def __init__(self, opts):
        super(FoodSpaceImageEncoder, self).__init__()

        self.visionMLP = visionMLP(opts)
        self.visual_embedding = nn.Sequential(
            nn.BatchNorm1d(2048,),
            nn.Linear(2048, 1024, 1024),
            nn.Tanh())
        self.align_img = nn.Sequential(
            nn.BatchNorm1d(opts.embDim),
            nn.Tanh())
        self.align = nn.Sequential(
            nn.Linear(1024, opts.embDim),
        )

    def forward(self, input):  # we need to check how the input is going to be provided to the model
        # visual embedding
        visual_emb, attention = self.visionMLP(input)
        visual_emb = visual_emb.view(visual_emb.size(0), -1)
        visual_emb = self.visual_embedding(visual_emb)
        visual_emb = self.align(visual_emb)
        visual_emb = self.align_img(visual_emb)
        visual_emb = norm(visual_emb)

        return visual_emb


## a placeholder to store all submodules
class FoodSpaceImageEncoder_2(nn.Module):
    def __init__(self, visionMLP, visual_embedding, align_img, fc_align):
        super(FoodSpaceImageEncoder_2, self).__init__()
        self.visionMLP = visionMLP
        self.visual_embedding = visual_embedding
        self.align_img = align_img
        self.align = fc_align

    def forward(self, input):  # we need to check how the input is going to be provided to the model
        # visual embedding
        visual_emb = self.visionMLP(input)
        visual_emb = visual_emb.view(visual_emb.size(0), -1)
        visual_emb = self.visual_embedding(visual_emb)
        visual_emb = self.align(visual_emb)
        visual_emb = self.align_img(visual_emb)
        visual_emb = norm(visual_emb)

        return visual_emb


def extract_image_encoder_from_FoodSpaceNet(food_space_net, image_encoder_save_path):
    image_encoder = FoodSpaceImageEncoder_2(food_space_net.visionMLP, food_space_net.visual_embedding, food_space_net.align_img, food_space_net.fc_align)
    # save
    torch.save(image_encoder.state_dict(), image_encoder_save_path)


def load_image_encoder(model_path, opts, use_cuda=True):

    model = FoodSpaceImageEncoder(opts)

    ret = load_torch_model(model, model_path)
    if not ret:
        return None
    if use_cuda:
        model.cuda()
    else:
        model.cpu()
    model.eval()
    return model


def rank_i2t(random_seed, im_vecs, instr_vecs):
    random.seed(random_seed)

    # Ranker
    N = 1000
    N_runs = 10
    #idxs = range(N)

    glob_rank = []
    glob_recall = {1: 0.0, 5: 0.0, 10: 0.0}
    for i in range(N_runs):

        ids = random.sample(range(0, len(im_vecs)), N)
        im_sub = im_vecs[ids, :]
        instr_sub = instr_vecs[ids, :]

        med_rank = []
        recall = {1: 0.0, 5: 0.0, 10: 0.0}

        sims = np.dot(im_sub, instr_sub.T)
        for ii in range(N):
            sorting = np.argsort(sims[ii, :])[::-1].tolist()
            # where this index 'ii' is in the sorted list
            pos = sorting.index(ii)

            if pos < 1:
                recall[1] += 1
            if pos < 5:
                recall[5] += 1
            if pos < 10:
                recall[10] += 1

            # store the position
            med_rank.append(pos + 1)

        for k in recall:
            recall[k] = recall[k] / N

        med = np.median(med_rank)

        for k in recall:
            glob_recall[k] += recall[k]
        glob_rank.append(med)

    for k in glob_recall:
        glob_recall[k] = glob_recall[k] / N_runs

    return np.average(glob_rank), glob_recall


def rank_only(img_embs, rec_embs, mode="i2t"):
    assert mode in ["i2t", "t2i"], "unsupported cross modal ranking"
    assert img_embs.shape == rec_embs.shape
    N = img_embs.shape[0]
    ranks = []
    recall = {1: 0.0, 5: 0.0, 10: 0.0}
    if N <= 30000:
        if mode == "i2t":
            sims = np.dot(img_embs, rec_embs.T)
        else:
            sims = np.dot(rec_embs, img_embs.T)
        for i in range(N):
            # sort in descending order
            sorting = np.argsort(sims[i,:])[::-1].tolist()
            # where this index 'i' is in the sorted list
            pos = sorting.index(i)
            if pos == 0:
                recall[1] += 1
            if pos < 5:
                recall[5] += 1
            if pos < 10:
                recall[10] += 1
            ranks.append(pos+1)
    else:
        for i in range(N):
            if mode == "i2t":
                sims = np.dot(img_embs[i,:], rec_embs.T)
            else:
                sims = np.dot(rec_embs[i,:], img_embs.T)
            sorting = np.argsort(sims)[::-1].tolist()
            # where this index 'i' is in the sorted list
            pos = sorting.index(i)
            if pos == 0:
                recall[1] += 1
            if pos < 5:
                recall[5] += 1
            if pos < 10:
                recall[10] += 1
            ranks.append(pos+1)
    medRank = np.median(ranks)
    for k in recall:
        recall[k] = recall[k] / N
    dcg = np.array([1/np.log2(r+1) for r in ranks]).mean()

    return medRank, recall, dcg, ranks


def rank_3(img_embs, rec_embs, img_splits, rec_splits, mode="i2t"):
    assert img_splits is not None and rec_splits is not None
    assert len(img_splits) == len(rec_splits)
    N_folds = len(img_splits)

    global_recall = {1: 0.0, 5: 0.0, 10: 0.0}
    global_rank = []
    all_ranks = []
    global_dcg = []

    for i in range(N_folds):
        # sampling fold_size samples
        img_emb_sub = img_embs[img_splits[i],:]
        rec_emb_sub = rec_embs[rec_splits[i],:]

        medRank, recall, dcg, ranks = rank_only(img_emb_sub, rec_emb_sub, mode)

        global_rank.append(medRank)
        all_ranks.append(np.mean(ranks))
        global_dcg.append(dcg)
        for k in global_recall:
            global_recall[k] += recall[k]

    for k in global_recall:
        global_recall[k] = global_recall[k] / N_folds

    return np.average(global_rank), global_recall, np.mean(all_ranks), np.mean(global_dcg)