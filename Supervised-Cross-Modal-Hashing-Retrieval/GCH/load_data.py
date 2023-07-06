import h5py
import numpy as np
from PIL import Image


def loading_data(path):
    print '******************************************************'
    print 'dataset:{0}'.format(path)
    print '******************************************************'
    file = h5py.File(path)
    images = file['images'][:].transpose(0,3,2,1)
    labels = file['LAll'][:].transpose(1,0)
    tags = file['YAll'][:].transpose(1,0)
    file.close()
    return images, tags, labels


def split_data(images, tags, labels, QUERY_SIZE, TRAINING_SIZE, DATABASE_SIZE):

	X = {}
	index_all = np.random.permutation(QUERY_SIZE+DATABASE_SIZE)
	ind_Q = index_all[0:QUERY_SIZE]
	ind_T = index_all[QUERY_SIZE:TRAINING_SIZE + QUERY_SIZE]
	ind_R = index_all[QUERY_SIZE:DATABASE_SIZE + QUERY_SIZE]

	X['query'] = images[ind_Q, :, :, :]
	X['train'] = images[ind_T, :, :, :]
	X['retrieval'] = images[ind_R, :, :, :]

	Y = {}
	Y['query'] = tags[ind_Q, :]
	Y['train'] = tags[ind_T, :]
	Y['retrieval'] = tags[ind_R, :]

	L = {}
	L['query'] = labels[ind_Q, :]
	L['train'] = labels[ind_T, :]
	L['retrieval'] = labels[ind_R, :]
	return X, Y, L


class LoadData():
    def __init__(self, path, dataset='NUS-WIDE'):
        if 'NUS-WIDE' in dataset:
            seg = 'NUS-WIDE/'
            wepath = '/home/libsource/Cross_Modal_Datasets/NUS-WIDE/'
        else:
            print "Dataset:{0}".format(dataset)
        # ImgPath = os.path.join(path, 'Img-R-21.h5')
        # F1 = h5py.File(ImgPath, 'r')
        # self.train_x_list = []
        # self.retrieval_x_list = []
        # self.query_x_list = []
        # train_x_list = F1['ImgTrain'][:]
        # for trainidx in train_x_list:
        #     train = trainidx.split(seg)[1]
        #     newpath = os.path.join(wepath, train)
        #     self.train_x_list.append(newpath)
        # self.train_x_list = np.array(self.train_x_list)
        # retrieval_x_list = F1['ImgDataBase'][:]
        # for retrievaligx in retrieval_x_list:
        #     retrieval = retrievaligx.split(seg)[1]
        #     newpath = os.path.join(wepath, retrieval)
        #     self.retrieval_x_list.append(newpath)
        # self.retrieval_x_list = np.array(self.retrieval_x_list)
        # query_x_list = F1['ImgQuery'][:]
        # for queryidx in query_x_list:
        #     query = queryidx.split(seg)[1]
        #     newpath = os.path.join(wepath, query)
        #     self.query_x_list.append(newpath)
        # self.query_x_list = np.array(self.query_x_list)
        # TagPath = os.path.join(path, 'Tag-R-21.h5')
        # F1.close()
        # F2 = h5py.File(TagPath, 'r')
        # self.train_y = F2['TagTrain'][:]
        # self.retrieval_y = F2['TagDataBase'][:]
        # self.query_y = F2['TagQuery'][:]
        # LabPath = os.path.join(path, 'Lab-R-21.h5')
        # F2.close()
        # F3 = h5py.File(LabPath, 'r')
        # self.train_L = F3['LabTrain'][:]
        # self.retrieval_L = F3['LabDataBase'][:]
        # self.query_L = F3['LabQuery'][:]
        # F3.close()
        print '******************************************************'
        print 					'dataset:{0}'.format(path)
        print '******************************************************'

    def loadimg(self, pathList):
        crop_size = 224
        ImgSelect = np.ndarray([len(pathList), crop_size, crop_size, 3])
        count = 0
        for path in pathList:
            img = Image.open(path)
            # img = img.resize((224, 224))
            # Img = np.array(img)
            # if Img.shape[2] != 3:
            #     print 'This image is not a rgb picture: {0}'.format(path)
            #     print 'The shape of this image is {0}'.format(Img.shape)
            #     ImgSelect[count, :, :, :] = Img[:, :, 0:3].astype(np.float32)
            #     count += 1
            # else:
            #     ImgSelect[count, :, :, :] = Img.astype(np.float32)
            #     count += 1
        # **********************************************************************************************************
            # **********************************************************************************************************
            # Here, we fist resize the original iamge into M*224 or 224*M, M>224, then cut out the part of M-224 surround.
            xsize, ysize = img.size
            seldim = min(xsize, ysize)
            rate = 224.0 / seldim
            img = img.resize((int(xsize * rate), int(ysize * rate)))
            nxsize, nysize = img.size
            box = (nxsize / 2.0 - 112, nysize / 2.0 - 112, nxsize / 2.0 + 112, nysize / 2.0 + 112)
            img = img.crop(box)
            img = img.convert("RGB")
            img = img.resize((224, 224))
            img = np.array(img)
            if img.shape[2] != 3:
                print 'This image is not a rgb picture: {0}'.format(path)
                print 'The shape of this image is {0}'.format(img.shape)
                ImgSelect[count, :, :, :] = img[:, :, 0:3]
                count += 1
            else:
                ImgSelect[count, :, :, :] = img
                count += 1

        return ImgSelect
            # **********************************************************************************************************
            # nulArray = np.zeros([224,224,3])
            # seldim = max(xsize, ysize)
            # rate = 224.0 / seldim
            # nxsize = int(xsize * rate)
            # nysize = int(ysize * rate)
            # if nxsize %2 != 0:
            #     nxsize = int(xsize*rate) + 1
            # if nysize %2 != 0:
            #     nysize = int(ysize*rate) + 1
            # img = img.resize((nxsize, nysize))
            # nxsize, nysize = img.size
            # img = img.convert("RGB")
            # img = np.array(img)
            # nulArray[112-nysize/2 :112+nysize/2, 112-nxsize/2 :112+nxsize/2, :] = img
            # if nulArray.shape[2] != 3:
            #     print 'This image is not a rgb picture: {0}'.format(path)
            #     print 'The shape of this image is {0}'.format(nulArray.shape)
            #     ImgSelect[count, :, :, :] = nulArray[:, :, 0:3]
            #     count += 1
            # else:
            #     ImgSelect[count, :, :, :] = nulArray
            #     count += 1
