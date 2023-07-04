import numpy as np
import scipy.io as sio
import scipy.spatial
import os

print 'load data! Wait please!'
dataset='twitter100k'
root='../../feature/'+dataset
image=sio.loadmat(root+'/test_image.mat')['image']
text=sio.loadmat(root+'/lda.mat')['text']
#text=sio.loadmat(root+'/text_word2vec_bow.mat')['text']
test_index=np.load(root+'/2000_test_index.npy')
nb_test=test_index.shape[0]
text=text[60000:]
print 'test.shape:',text.shape,'image.shape',image.shape
test_txt=text[test_index]
test_im=image[test_index]
index=np.arange(image.shape[0])
print 'test_txt.shape',test_txt.shape,'test_im.shape',test_im.shape
dist_method='cosine'
savepath='../../result/rank/'+dataset+os.sep
if not os.path.exists(savepath):
	os.mkdir(savepath)
for baseline in ['pls','cca','blm','mfa','lpp']:
	print baseline
	data=sio.loadmat('../../result/Wout/'+dataset+'/lda/'+baseline+'/Wout.mat')
	W_I=data['W_I']
	W_T=data['W_T']

	print 'im2txt'
	rank=np.zeros((nb_test,))
	t=np.dot(text,W_T)
	i=np.dot(test_im,W_I)
	print 'calculate dist'
        dist=scipy.spatial.distance.cdist(i,t,dist_method)
        print 'sort dist!'
        order=dist.argsort()
        for i in np.arange(nb_test):
		rank[i]=order[i,:].tolist().index(test_index[i])
        sio.savemat(savepath+baseline+'_im2txt_rank.mat',{'rank':rank})

	print 'txt2im'
	rank=np.zeros((nb_test,))
	t=np.dot(test_txt,W_T)
	i=np.dot(image,W_I)
	print 'calculate dist'
	dist=scipy.spatial.distance.cdist(t,i,dist_method)
	print 'sort dist!'
	order=dist.argsort()
	for i in np.arange(nb_test):
		rank[i]=order[i,:].tolist().index(test_index[i])
	sio.savemat(savepath+baseline+'_txt2im_rank.mat',{'rank':rank})
