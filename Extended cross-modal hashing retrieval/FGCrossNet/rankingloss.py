import torch

def ranking_loss(labels, embeddings, margin, margin2, squared=False):
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)
    
    anchor_positive_dist = pairwise_dist.unsqueeze(2).unsqueeze(3)
    anchor_negative1_dist = pairwise_dist.unsqueeze(1).unsqueeze(3)
    negative1_negative2_dist = pairwise_dist.unsqueeze(0).unsqueeze(0)

    triplet_loss = (anchor_positive_dist - anchor_negative1_dist + margin) + (anchor_positive_dist - negative1_negative2_dist + margin2)

    mask = _get_triplet_mask(labels)
    triplet_loss = mask.float() * triplet_loss

    triplet_loss[triplet_loss < 0] = 0

    valid_triplets = triplet_loss[triplet_loss > 1e-16]
    num_positive_triplets = valid_triplets.size(0)
    num_valid_triplets = mask.sum()

    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets.float() + 1e-16)

    triplet_loss = triplet_loss.sum() / (num_positive_triplets + 1e-16)

    return triplet_loss, fraction_positive_triplets

def _pairwise_distances(embeddings, squared=False):
    dot_product = torch.matmul(embeddings, embeddings.t())
    square_norm = torch.diag(dot_product)
    distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)

    distances[distances < 0] = 0

    if not squared:
        mask = distances.eq(0).float()
        distances = distances + mask * 1e-16
        distances = (1.0 -mask) * torch.sqrt(distances)

    return distances

def _get_triplet_mask(labels):

    indices_equal = torch.eye(labels.size(0)).byte()
    indices_not_equal = ~indices_equal

    i_not_equal_j = indices_not_equal.unsqueeze(2).unsqueeze(3)
    i_not_equal_k = indices_not_equal.unsqueeze(1).unsqueeze(3)
    j_not_equal_k = indices_not_equal.unsqueeze(2).unsqueeze(0)

    i_not_equal_l = indices_not_equal.unsqueeze(1).unsqueeze(1)
    j_not_equal_l = indices_not_equal.unsqueeze(1).unsqueeze(0)
    k_not_equal_l = indices_not_equal.unsqueeze(0).unsqueeze(0)

    distinct_indices = (((((i_not_equal_j & i_not_equal_k) & j_not_equal_k) & i_not_equal_l) & j_not_equal_l) & k_not_equal_l)

    label_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    i_equal_j = label_equal.unsqueeze(2).unsqueeze(3)
    i_equal_k = label_equal.unsqueeze(1).unsqueeze(3)
    i_equal_l = label_equal.unsqueeze(1).unsqueeze(1)
    k_equal_l = label_equal.unsqueeze(0).unsqueeze(0)

    valid_labels = i_equal_j & ~i_equal_k & ~i_equal_l & ~k_equal_l

    return valid_labels & distinct_indices.cuda()