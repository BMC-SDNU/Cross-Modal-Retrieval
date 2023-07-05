# -----------------------------------------------------------
# "Universal Weighting Metric Learning for Cross-Modal Matching"
# Jiwei Wei, Xing Xu, Yang Yang, Yanli Ji, Zheng Wang, Heng Tao Shen
#  
# Writen by Jiwei Wei, 2020
# ---------------------------------------------------------------

class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self, opt, margin=0, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        
        self.opt = opt
        self.margin = margin
        self.max_violation = max_violation
    def polyloss(self,sim_mat):
        epsilon = 1e-5
        size=sim_mat.size(0)
        hh=sim_mat.t()
        label=torch.Tensor([i for i in range(size)])
  
        loss = list()
        for i in range(size):
            pos_pair_ = sim_mat[i][i]
            pos_pair_ = pos_pair_[pos_pair_ < 1 - epsilon]
            neg_pair_ = sim_mat[i][label!=label[i]]
    
            neg_pair = neg_pair_[neg_pair_ + self.margin > min(pos_pair_)]

            pos_pair=pos_pair_
            if len(neg_pair) < 1 or len(pos_pair) < 1:
                continue

            pos_loss =torch.clamp(0.2*torch.pow(pos_pair,2)-0.7*pos_pair+0.5, min=0)
            neg_pair=max(neg_pair)
            neg_loss = torch.clamp(0.9*torch.pow(neg_pair,2)-0.4*neg_pair+0.03,min=0)

            loss.append(pos_loss + neg_loss)
        for i in range(size):
            pos_pair_ = hh[i][i]
            pos_pair_ = pos_pair_[pos_pair_ < 1 - epsilon]
            neg_pair_ = hh[i][label!=label[i]]
    
            neg_pair = neg_pair_[neg_pair_ + self.margin > min(pos_pair_)]

            pos_pair=pos_pair_
            if len(neg_pair) < 1 or len(pos_pair) < 1:
                continue
            pos_loss =torch.clamp(0.2*torch.pow(pos_pair,2)-0.7*pos_pair+0.5,min=0)

            neg_pair=max(neg_pair)
            neg_loss = torch.clamp(0.9*torch.pow(neg_pair,2)-0.4*neg_pair+0.03,min=0)
            loss.append(pos_loss + neg_loss)
            
        if len(loss) == 0:
            return torch.zeros([], requires_grad=True)
    
        loss = sum(loss) / size
        return loss

    def forward(self, im, s, s_l):
        # compute image-sentence score matrix
        if self.opt.cross_attn == 't2i':
            scores = xattn_score_t2i(im, s, s_l, self.opt)
        elif self.opt.cross_attn == 'i2t':
            scores = xattn_score_i2t(im, s, s_l, self.opt)
        else:
            raise ValueError("unknown first norm type:", self.opt.raw_feature_norm)
        loss=self.polyloss(scores)
        return loss