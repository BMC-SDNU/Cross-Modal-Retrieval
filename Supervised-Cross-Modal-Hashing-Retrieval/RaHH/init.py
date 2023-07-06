import numpy as np
from cvh import *
from loss_func import *
from update import *

def get_attr(image_fea, tag_fea, similarity):
    #The attribute of the input data
    #K #domain
    #N #data instances
    #dim #data feature dimension
    K = 2
    N = [int(np.size(image_fea, 1)), int(np.size(tag_fea, 1))]
    dim = [int(np.size(image_fea, 0)), int(np.size(tag_fea, 0))]

    return [K, N, dim]


def initialize(image_fea, tag_fea, similarity, bit):
    #Initialize the hash code for image and tag using CVH
    #Initialize the inter domain mapping matrix W as I
    #Initialize the matrix S

    [K, N, dim] = get_attr(image_fea, tag_fea, similarity)

    #the hash code length
    #hash_bit = [4, 4]
    hash_bit = bit

    #Hash code rp*mp
    [H_img, H_tag, A_img, A_tag] = cvh(similarity, image_fea, tag_fea,hash_bit)
    #Heterogeneous mapping by image_hash'*W = tag_hash
    W = np.eye(hash_bit[0], hash_bit[1])

    S_img = update_S(image_fea, H_img)
    S_tag = update_S(tag_fea, H_tag)
    S = [S_img, S_tag]

    R_p = np.eye(np.shape(image_fea)[1])
    R_q = np.eye(np.shape(tag_fea)[1])
    return [H_img, H_tag, W, S, R_p, R_q, A_img, A_tag]