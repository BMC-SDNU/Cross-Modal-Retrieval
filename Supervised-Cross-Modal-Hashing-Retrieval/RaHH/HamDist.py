#To calculate the Hamming Distance given two string
#import numpy as np
import scipy.spatial as sp

def HamDist(v1, v2):
    
    return sp.distance.hamming(v1,v2)

if __name__ == '__main__':
    
    print sp.distance.hamming([1,1,1],[1,-1,1])