import os
import sys
import torch
import numpy as np
import scipy.spatial
def compute_mAP(img, vid, aud, txt):
        
	image = np.loadtxt(img, dtype = np.float32)
	video = np.loadtxt(vid, dtype = np.float32)
	audio = np.loadtxt(aud, dtype = np.float32)
	text = np.loadtxt(txt, dtype = np.float32)
	
	image_labels = np.loadtxt('./list/image/testlabel.txt')
	video_labels = np.loadtxt('./list/video/testlabel.txt')
	audio_labels = np.loadtxt('./list/audio/testlabel.txt')
	text_labels = np.loadtxt('./list/text/testlabel.txt')

	it = mAP(image, text, image_labels, text_labels)
	ia = mAP(image, audio, image_labels, audio_labels)
	iv = mAP(image, video, image_labels, video_labels)

	ti = mAP(text, image, text_labels, image_labels)
	ta = mAP(text, audio, text_labels, audio_labels)
	tv = mAP(text, video, text_labels, video_labels)

	ai = mAP(audio, image, audio_labels, image_labels)
	at = mAP(audio, text, audio_labels, text_labels)
	av = mAP(audio, video, audio_labels, video_labels)
	
	vi = mAP(video, image, video_labels, image_labels)
	vt = mAP(video, text, video_labels, text_labels)
	va = mAP(video, audio, video_labels, audio_labels)
    
	print('i2t mAP: %f' % it)
	print('i2a mAP: %f' % ia)
	print('i2v mAP: %f' % iv)
	print('t2i mAP: %f' % ti)
	print('t2a mAP: %f' % ta)
	print('t2v mAP: %f' % tv)
	print('a2i mAP: %f' % ai)
	print('a2t mAP: %f' % at)
	print('a2v mAP: %f' % av)
	print('v2i mAP: %f' % vi)
	print('v2t mAP: %f' % vt)
	print('v2a mAP: %f' % va)
	
	image_labels = np.expand_dims(image_labels, axis=1)
	video_labels = np.expand_dims(video_labels, axis=1)
	audio_labels = np.expand_dims(audio_labels, axis=1)
	text_labels = np.expand_dims(text_labels, axis=1)
	i2all = mAP(image, np.vstack((image,text,audio,video)), image_labels, np.vstack((image_labels,text_labels,audio_labels,video_labels)))
	t2all = mAP(text, np.vstack((image,text,audio,video)), text_labels, np.vstack((image_labels,text_labels,audio_labels,video_labels)))
	a2all = mAP(audio, np.vstack((image,text,audio,video)), audio_labels, np.vstack((image_labels,text_labels,audio_labels,video_labels)))
	v2all = mAP(video, np.vstack((image,text,audio,video)), video_labels, np.vstack((image_labels,text_labels,audio_labels,video_labels)))
	print('i2all mAP: %f' % i2all)
	print('t2all mAP: %f' % t2all)
	print('a2all mAP: %f' % a2all)
	print('v2all mAP: %f' % v2all)

def mAP(image, text, image_label, text_label):
	dist = scipy.spatial.distance.cdist(image, text, 'cosine')
	ord = dist.argsort()
	numcases = dist.shape[0]
	k = dist.shape[1]
	res = []
	for i in range(numcases):
		order = ord[i]
		p = 0.0
		r = 0.0
		for j in range(k):
			if image_label[i] == text_label[order[j]]:
				r += 1
				p += (r / (j + 1))
		if r > 0:
			res += [p / r]
		else:
			res += [0]
	return np.mean(res)
