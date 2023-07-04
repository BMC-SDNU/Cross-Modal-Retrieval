import torch
import numpy as np
import scipy
from scipy.spatial.distance import cdist


def fx_calc_map_label(image, text, label, corrflag = False, k = 0, dist_method='COS'):
  if dist_method == 'L2':
    dist = scipy.spatial.distance.cdist(image, text, 'euclidean')
  elif dist_method == 'COS':
    dist = scipy.spatial.distance.cdist(image, text, 'cosine')
  ord = dist.argsort()
  numcases = dist.shape[0]
  if k == 0:
    k = numcases
  res = []
  allcorr = []
  for i in range(numcases):
    order = ord[i]
    p = 0.0
    r = 0.0
    corr = 0
    for j in range(k):
      if label[i] == label[order[j]]:
        if j < 4:
          corr += 1
          if corr == 4:
            allcorr.append(i)
        r += 1
        p += (r / (j + 1))
    if r > 0:
      res += [p / r]
    else:
      res += [0]

  if corrflag and len(allcorr)>0:
    print(allcorr)
  return np.mean(res)