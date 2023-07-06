from torch import nn
import torch
from torch.nn import functional as F


def _euclidean_distances(source, target, squared=False):
    """
    Compulate the 2D matrix of distances between all the source and target vectors.
    :param source: tensor of shape (batch_size, embed_dim)
    :param target: tensor of shape (*, embed_dim)
    :param squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                    If false, output is the pairwise euclidean distance matrix.
    :return: pairwise_distances: tensor of shape (batch_size, batch_size)
    """

    # Compute the pairwise distance matrix
    # shape (batch_size, batch_size)
    distances = torch.sum(torch.pow((source.unsqueeze(1) - target.unsqueeze(0)), 2), dim=-1)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = torch.clamp(distances, 0)

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = (torch.eq(distances, 0.0)).float()
        distances = distances + mask * 1e-16

        distances = torch.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)

    return distances


def _cos_distance(source, target):
    """
    Compulate the 2D matrix of cosine distance between all the source and target vectors.
    :param source: tensor of shape (batch_size, embed_dim)
    :param target: tensor of shape (*, embed_dim)
    :return: tensor of shape (batch_size, batch_size)
    """

    cos_sim = F.cosine_similarity(source.unsqueeze(1), target, dim=-1)

    # put everything >= 0
    distances = torch.clamp(1 - cos_sim, 0)

    return distances


def _get_anchor_triplet_mask(s_labels, t_labels):
    """
    Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.
    :param s_labels: tensor with shape (batch_size, label_num)
    :param t_labels: tensor with shape (batch_size, label_num)
    :return: positive mask and negative mask, `Tensor` with shape [batch_size, batch_size]
    """
    sim = (s_labels.mm(t_labels.t()) > 0).float()

    return sim, 1 - sim


def _get_triplet_mask(s_labels, t_labels):
    """
    Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid
    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]
    :param s_labels: tensor with shape (batch_size, label_num)
    :param t_labels: tensor with shape (batch_size, label_num)
    """
    sim = (s_labels.mm(t_labels.t()) > 0).float()

    # # Check that i, j and k are distinct
    # indices_not_equal = 1 - torch.eye(sim.shape[0]).to(s_labels.device)
    # i_not_equal_j = indices_not_equal.unsqueeze(2)
    # i_not_equal_k = indices_not_equal.unsqueeze(1)
    # j_not_equal_k = indices_not_equal.unsqueeze(0)
    #
    # distinct_indices = i_not_equal_j * i_not_equal_k * j_not_equal_k

    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    i_equal_j = sim.unsqueeze(2)
    i_equal_k = sim.unsqueeze(1)

    mask = i_equal_j * (1 - i_equal_k)

    # Combine the two masks
    # mask = distinct_indices * valid_labels

    return mask


class TripletAllLoss(nn.Module):
    def __init__(self, dis_metric='euclidean', squared=False, reduction='mean'):
        """
        Build the triplet loss over a batch of embeddings.
        We generate all the valid triplets and average the loss over the positive ones.
        :param margin:
        :param dis_metric: 'euclidean' or 'dp'(dot product)
        :param squared:
        :param reduction: 'mean' or 'sum' or 'none'
        """
        super(TripletAllLoss, self).__init__()

        self.dis_metric = dis_metric
        self.reduction = reduction
        self.squared = squared

    def forward(self, source, s_labels, target=None, t_labels=None, margin=0):
        if target is None:
            target = source
        if t_labels is None:
            t_labels = s_labels

        if self.dis_metric is 'euclidean':
            pairwise_dist = _euclidean_distances(source, target, self.squared)
        elif self.dis_metric is 'cos':
            pairwise_dist = _cos_distance(source, target)

        # shape (batch_size, batch_size, 1)
        anchor_positive_dist = pairwise_dist.unsqueeze(2)
        # shape (batch_size, 1, batch_size)
        anchor_negative_dist = pairwise_dist.unsqueeze(1)

        # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
        # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
        # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
        # and the 2nd (batch_size, 1, batch_size)
        if self.dis_metric is 'euclidean':
            triplet_loss = anchor_positive_dist - (1 - margin) * anchor_negative_dist
        else:
            triplet_loss = anchor_positive_dist - anchor_negative_dist + margin
        # triplet_loss = anchor_positive_dist - anchor_negative_dist

        # Put to zero the invalid triplets
        # (where label(a) != label(p) or label(n) == label(a) or a == p)
        mask = _get_triplet_mask(s_labels, t_labels)
        triplet_loss = mask * triplet_loss

        # Remove negative losses (i.e. the easy triplets)
        triplet_loss = triplet_loss.clamp(0)

        # Count number of positive triplets (where triplet_loss > 0)
        valid_triplets = triplet_loss.gt(1e-16).float()
        num_positive_triplets = valid_triplets.sum()

        if self.reduction == 'mean':
            # Get final mean triplet loss over the positive valid triplets
            triplet_loss = triplet_loss.sum() / (num_positive_triplets + 1e-16)
        elif self.reduction == 'sum':
            triplet_loss = triplet_loss.sum()

        return triplet_loss


class TripletHardLoss(nn.Module):
    def __init__(self, dis_metric='euclidean', squared=False, reduction='mean'):
        """
        Build the triplet loss over a batch of embeddings.
        For each anchor, we get the hardest positive and hardest negative to form a triplet.
        :param margin:
        :param dis_metric: 'euclidean' or 'dp'(dot product)
        :param squared:
        :param reduction: 'mean' or 'sum' or 'none'
        """
        super(TripletHardLoss, self).__init__()

        self.dis_metric = dis_metric
        self.reduction = reduction
        self.squared = squared

    def forward(self, source, s_labels, target=None, t_labels=None, margin=0):
        if target is None:
            target = source
        if t_labels is None:
            t_labels = s_labels

        # Get the pairwise distance matrix
        if self.dis_metric == 'euclidean':
            pairwise_dist = _euclidean_distances(source, target, squared=self.squared)
        elif self.dis_metric == 'cos':
            pairwise_dist = _cos_distance(source, target)

        # First, we need to get a mask for every valid positive (they should have same label)
        # and every valid negative (they should have different labels)
        mask_anchor_positive, mask_anchor_negative = _get_anchor_triplet_mask(s_labels, t_labels)

        # For each anchor, get the hardest positive
        # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
        anchor_positive_dist = mask_anchor_positive * pairwise_dist

        # shape (batch_size, 1)
        hardest_positive_dist, _ = anchor_positive_dist.max(dim=1, keepdim=True)

        # For each anchor, get the hardest negative
        # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
        max_anchor_negative_dist, _ = pairwise_dist.max(dim=1, keepdim=True)
        anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

        # shape (batch_size,)
        hardest_negative_dist, _ = anchor_negative_dist.min(dim=1, keepdim=True)

        # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
        triplet_loss = torch.clamp(hardest_positive_dist - hardest_negative_dist + margin, 0.0)

        # Get final mean triplet loss
        if self.reduction is 'mean':
            triplet_loss = triplet_loss.mean()
        elif self.reduction is 'sum':
            triplet_loss = triplet_loss.sum()

        return triplet_loss
