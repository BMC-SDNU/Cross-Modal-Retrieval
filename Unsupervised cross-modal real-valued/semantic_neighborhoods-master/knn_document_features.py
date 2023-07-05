import numpy as np
import pickle
import os
import sys
import json
import pdb, random
import time
from PIL import Image
from tqdm import tqdm
import nmslib

def main():
    doc_features = pickle.load(open('./doc2vec_features.pickle', 'rb'))
    files = list(doc_features.keys())
    features = np.stack(list(doc_features.values()), axis=0).astype(np.float32)
    del doc_features
    index = nmslib.init(method='hnsw', space='l2')
    index.addDataPointBatch(features)
    index.createIndex({'post': 2}, print_progress=True)
    print(f'Len feats {len(features)}')
    neighbors = index.knnQueryBatch(features, k=200, num_threads=48)
    # Create path->neighbors lists. Each path is paired with approximate neighbors
    pickle.dump({'paths' : files, 'neighbors' : neighbors}, open('document_feats_knn.pickle', 'wb'))

if __name__ == '__main__':
    main()
