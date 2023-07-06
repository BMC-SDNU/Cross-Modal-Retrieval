import numpy as np
import scipy.linalg as scialg
import load_data
import HamDist
import pylab as P

#Reference : Kumar, S., & Udupa, R. (2011). 
#Learning hash functions for cross-view similarity search. 
#Paper presented at the Proceedings of the Twenty-Second international joint conference 
#on Artificial Intelligence-Volume Volume Two.
#Input: X: n dim-dimensional instances.
#       W: n*n matrix to represent the similarity between each instance
#Output: The hash function for each view/domain/task. 
#For initialize RaHH. Can also be used as baseline.

def domain2view(fea_1, fea_2, similarity):
    #Transform the setting of data
    #the image and tag whose similarity is greater than 
    #threshold is chosen as the different view for each (concept)
    #After transforming,the I is used to represent
    threshold = 0.001
    indicator = similarity > threshold #For choosing the pair that will be used

    dim= [np.size(fea_1, 0), np.size(fea_2, 0)]
    
    num_x = np.sum(indicator)

    #print 'numx', num_x

    X_1 = np.zeros([num_x, dim[0]])
    X_2 = np.zeros([num_x, dim[1]])
    
    count = 0
    
    for i in range(np.size(similarity, 0)):
        for j in range(np.size(similarity,1)):
            
            if similarity[i, j] > threshold:
                
                X_1[count] = fea_1[:, i]
                X_2[count] = fea_2[:, j]
                count += 1
    
    return [X_1, X_2]



def hash_function(X_1, X_2):
    #Based two view X_1, and X_2
    #return the hash function each view A_1 and A_2
    
    eye_lambda = 1e-4
    [A1, A2] = train_CCA(X_1, X_2, eye_lambda)
    
    return [A1, A2]


def train_CCA(X_1, X_2, eye_lambda):
    #The CCA to train the obtian the hash function
    
    X_1t = np.transpose(X_1)
    X_2t = np.transpose(X_2)
    
    Cxx = np.dot(X_1t, X_1)
    Cyy = np.dot(X_2t, X_2)
    Cxy = np.dot(X_1t, X_2)
    Cyx = np.dot(X_2t, X_1)
     
    #avoid Sigularity
    Cxx = np.add(Cxx, np.multiply(eye_lambda, np.eye(np.shape(Cxx)[0])))
    Cyy = np.add(Cyy, np.multiply(eye_lambda, np.eye(np.shape(Cyy)[0])))

    A = np.dot(Cxy,np.dot(np.linalg.pinv(Cyy), Cyx))
    B = Cxx
    
    [eigval, eigvec] = scialg.eig(A, B)
    
    A1 = np.real(eigvec)
    eigval = np.diag(np.real(eigval))
    
    A2 = np.dot(np.dot(np.dot(np.linalg.pinv(Cyy), np.transpose(Cxy)), A1), np.linalg.pinv(eigval))
    
    return [A1, A2]
    

def cvh(image_tags_cross_similarity, image_features, tag_features, bit):
    
    [X_1, X_2] = domain2view(image_features, tag_features, image_tags_cross_similarity)
    
    [A_1, A_2] = hash_function(X_1, X_2)

    hash_1 = np.dot(np.transpose(A_1[:, 0:bit[0]]), image_features)
    hash_2 = np.dot(np.transpose(A_2[:, 0:bit[1]]), tag_features)
    
    hash_1 = np.sign(hash_1)
    hash_2 = np.sign(hash_2)

    return [hash_1, hash_2, A_1.transpose(), A_2.transpose()]
    
if __name__ == '__main__':
    
    [image_tags_cross_similarity, image_features, tag_features] = load_data.analysis()
    
    bit = [32, 32]
    [hash_1, hash_2] = cvh(image_tags_cross_similarity, image_features, tag_features,bit)
    
    print np.shape(hash_1)
    print np.shape(hash_2)
    print hash_1
    
    for i in range(np.shape(tag_features)[1]):
        print i,'  ', image_tags_cross_similarity[0][i], 'Distance:', HamDist.HamDist(hash_1[:,0], hash_2[:,i])