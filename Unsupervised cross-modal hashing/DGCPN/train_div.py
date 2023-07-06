import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from metric import compress, calculate_top_map, calculate_map, p_topK, calculate_map_1
from models import ImgNet, TxtNet
import os.path as osp
from load_data import get_loader, get_loader_wiki
import numpy as np
import pdb
import time
import logging


def calc_dis(query_L, retrieval_L, query_dis, top_k=32):
    num_query = query_L.shape[0]
    map = 0
    for iter in range(num_query):
        gnd = (np.dot(query_L[iter, :], retrieval_L.transpose()) > 0).astype(np.float32)
        tsum = np.sum(gnd)
        if tsum == 0:
            continue
        hamm = query_dis[iter]
        ind = np.argsort(hamm)[:top_k]
        gnd = gnd[ind]
        tsum = np.int(np.sum(gnd))
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map = map + np.mean(count / (tindex))
    map = map / num_query
    return map


class Session:
    def __init__(self, opt):
        self.opt = opt
        if opt.data_name == 'wiki':
            dataloader, data_train = get_loader_wiki('./', opt.batch_size)
        else:
            dataloader, data_train = get_loader(opt.data_name, opt.batch_size)
        # Data Loader (Input Pipeline)
        self.global_imgs, self.global_txts, self.global_labs = data_train
        self.global_imgs = F.normalize(torch.Tensor(self.global_imgs)).cuda()
        self.global_txts = F.normalize(torch.Tensor(self.global_txts)).cuda()
        self.global_labs = torch.Tensor(self.global_labs).cuda()
        self.gs, self.sa, self.ni = self.cal_similarity(self.global_imgs, self.global_txts)
        self.train_loader = dataloader['train']
        self.val_loader = dataloader['validation']
        self.test_loader = dataloader['query']
        self.database_loader = dataloader['database']
        self.databasev_loader = dataloader['databasev']

        txt_feat_len = self.global_txts.size(1)
        self.CodeNet_I = ImgNet(code_len=opt.bit, txt_feat_len=txt_feat_len)
        self.FeatNet_I = ImgNet(code_len=opt.bit, txt_feat_len=txt_feat_len)
        self.CodeNet_T = TxtNet(code_len=opt.bit, txt_feat_len=txt_feat_len)

        self.opt_I = torch.optim.SGD(self.CodeNet_I.parameters(), lr=opt.learning_rate, momentum=opt.momentum,
                                     weight_decay=opt.weight_decay)

        self.opt_T = torch.optim.SGD(self.CodeNet_T.parameters(), lr=opt.learning_rate, momentum=opt.momentum,
                                     weight_decay=opt.weight_decay)
                                   
        self.best = 0
        # pdb.set_trace()
        logger = logging.getLogger('train')
        logger.setLevel(logging.INFO)
        stream_log = logging.StreamHandler()
        stream_log.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        stream_log.setFormatter(formatter)
        logger.addHandler(stream_log)
        self.logger = logger

    def cal_similarity(self, F_I, F_T):
        batch_size = F_I.size(0)
        
        F_I = F.normalize(F_I)
        S_I = F_I.mm(F_I.t())
        F_T = F.normalize(F_T)
        S_T = F_T.mm(F_T.t())

        S_pair = self.opt.a1 * S_T + (1 - self.opt.a1) * S_I
        
        pro = F_T.mm(F_T.t()) * self.opt.a1 + F_I.mm(F_I.t()) * (1. - self.opt.a1)
        size = batch_size
        top_size = self.opt.knn_number
        m, n1 = pro.sort()
        pro[torch.arange(size).view(-1, 1).repeat(1, top_size).view(-1), n1[:, :top_size].contiguous().view(
            -1)] = 0.
        pro[torch.arange(size).view(-1), n1[:, -1:].contiguous().view(
            -1)] = 0.
        pro = pro / pro.sum(1).view(-1, 1)
        pro_dis = pro.mm(pro.t())
        pro_dis = pro_dis * self.opt.scale
        # pdb.set_trace()
        S = (S_pair * (1 - self.opt.a2) + pro_dis * self.opt.a2)
        S = S * 2.0 - 1
        
        return S, S_pair, pro_dis
                
    def loss_cal(self, code_I, code_T, S, I):
        B_I = F.normalize(code_I)
        B_T = F.normalize(code_T)

        BI_BI = B_I.mm(B_I.t())
        BT_BT = B_T.mm(B_T.t())
        BI_BT = B_I.mm(B_T.t())
        # pdb.set_trace()
        diagonal = BI_BT.diagonal()
        all_1 = torch.rand((BT_BT.size(0))).fill_(1).cuda()
        loss_pair = F.mse_loss(diagonal, self.opt.K * all_1)

        loss_dis_1 = F.mse_loss(BT_BT * (1-I), S * (1-I))
        loss_dis_2 = F.mse_loss(BI_BT * (1-I), S * (1-I))
        loss_dis_3 = F.mse_loss(BI_BI * (1-I), S * (1-I))

        loss_cons = F.mse_loss(BI_BT, BI_BI) + \
                    F.mse_loss(BI_BT, BT_BT) + \
                    F.mse_loss(BI_BI, BT_BT) + \
                    F.mse_loss(BI_BT, BI_BT.t())

        loss = loss_pair + (loss_dis_1 + loss_dis_2 + loss_dis_3) * self.opt.dw \
               + loss_cons * self.opt.cw
        # loss = loss_pair
        loss = loss

        return loss, (loss_pair, loss_dis_1, loss_dis_2, loss_dis_3, loss_cons, loss_cons)
        
    def train(self, epoch):
        self.CodeNet_I.cuda().train()
        self.CodeNet_T.cuda().train()
        top_mAP = 0.0
        num = 0.0
        # self.CodeNet_I.set_alpha(epoch)
        # self.CodeNet_T.set_alpha(epoch * 2.0)

        self.logger.info('Epoch [%d/%d], alpha for ImgNet: %.3f, alpha for TxtNet: %.3f' % (
            epoch + 1, self.opt.num_epochs, self.CodeNet_I.alpha, self.CodeNet_T.alpha))
        for idx, (img, txt, labels, index) in enumerate(self.train_loader):
            img = Variable(img.cuda())
            txt = Variable(torch.FloatTensor(txt.numpy()).cuda())
            
            batch_size = img.size(0)
            I = torch.eye(batch_size).cuda()
            _, code_I = self.CodeNet_I(img)
            _, code_T = self.CodeNet_T(txt)

            S = self.gs[index, :][:, index].cuda()

            loss, all_los = self.loss_cal(code_I, code_T, S, I)
            self.opt_I.zero_grad()
            self.opt_T.zero_grad()
            loss.backward(retain_graph=True)
            self.opt_I.step()
            self.opt_T.step()
            
            _, code_I = self.CodeNet_I(img)
            _, code_T = self.CodeNet_T(txt)

            loss_i, _ = self.loss_cal(code_I, code_T.sign().detach(), S, I)
            self.opt_I.zero_grad()
            loss_i.backward(retain_graph=True)
            self.opt_I.step()
            
            loss_t, _ = self.loss_cal(code_I.sign().detach(), code_T, S, I)         
            self.opt_T.zero_grad()
            loss_t.backward()
            self.opt_T.step()

            loss1, loss2, loss3, loss4, loss5, loss6 = all_los

            top_mAP += calc_dis(labels.cpu().numpy(), labels.cpu().numpy(), -S.cpu().numpy())

            num += 1.
            if (idx + 1) % (len(self.train_loader)) == 0:
                self.logger.info(
                    'Epoch [%d/%d], Iter [%d/%d] '
                    'Loss1: %.4f Loss2: %.4f Loss3: %.4f '
                    'Loss4: %.4f '
                    'Loss5: %.4f Loss6: %.4f '
                    'Total Loss: %.4f '
                    'mAP: %.4f'
                    % (
                        epoch + 1, self.opt.num_epochs, idx + 1,
                        len(self.train_loader) // self.opt.batch_size,
                        loss1.mean().item(), loss2.mean().item(), loss3.mean().item(),
                        loss4.item(),
                        code_T.abs().mean().item(), 
                        code_I.abs().mean().item(),
                        loss.item(),
                        top_mAP / num))

    def eval(self, step=0, num_epoch=0, last=False):
        # Change model to 'eval' mode (BN uses moving mean/var).
        self.CodeNet_I.eval().cuda()
        self.CodeNet_T.eval().cuda()
        if self.opt.EVAL == False:
            re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L = compress(self.databasev_loader, self.val_loader, self.CodeNet_I,
                                                              self.CodeNet_T)

            MAP_I2T = calculate_top_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L, topk=50)
            MAP_T2I = calculate_top_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L, topk=50)
            MAP_I2Ta = calculate_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L)
            MAP_T2Ia = calculate_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L)
            MAP_I2Ta_1 = calculate_map_1(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L)
            MAP_T2Ia_1 = calculate_map_1(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L)
            K = [1, 200, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
            self.logger.info('--------------------Evaluation: Calculate top MAP-------------------')
            self.logger.info('MAP@50 of Image to Text: %.3f, MAP of Text to Image: %.3f' % (MAP_I2T, MAP_T2I))
            self.logger.info('MAP@All of Image to Text: %.3f, MAP of Text to Image: %.3f' % (MAP_I2Ta, MAP_T2Ia))
            self.logger.info('MAP@All_1 of Image to Text: %.3f, MAP of Text to Image: %.3f' % (MAP_I2Ta_1, MAP_T2Ia_1))
            self.logger.info('--------------------------------------------------------------------')
            if MAP_I2Ta + MAP_T2Ia > self.best:
                num_epoch = 0
                self.save_checkpoints(step=step, best=True)
                self.best = MAP_T2Ia + MAP_I2Ta
                self.logger.info("#########is best:%.3f #########" % self.best)
            else:
                num_epoch += 1
        if self.opt.EVAL:
            re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L = compress(self.database_loader, self.test_loader, self.CodeNet_I,
                                                              self.CodeNet_T)
            MAP_I2T = calculate_top_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L, topk=5)
            MAP_T2I = calculate_top_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L, topk=5)
            MAP_I2Ta = calculate_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L)
            MAP_T2Ia = calculate_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L)
            MAP_I2Ta_1 = calculate_map_1(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L)
            MAP_T2Ia_1 = calculate_map_1(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L)
            if self.opt.data_name == 'wiki':
                K = [1, 200, 400, 500, 1000, 1500, 2000]
            else:
                K = [1, 200, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
            self.logger.info('--------------------Evaluation: Calculate top MAP-------------------')
            self.logger.info('MAP@50 of Image to Text: %.3f, MAP of Text to Image: %.3f' % (MAP_I2T, MAP_T2I))
            self.logger.info('MAP@All of Image to Text: %.3f, MAP of Text to Image: %.3f' % (MAP_I2Ta, MAP_T2Ia))
            self.logger.info('MAP@All_1 of Image to Text: %.3f, MAP of Text to Image: %.3f' % (MAP_I2Ta_1, MAP_T2Ia_1))
            self.logger.info('--------------------------------------------------------------------')
            retI2T = p_topK(qu_BI, re_BT, qu_L, re_L, K)
            retT2I = p_topK(qu_BT, re_BI, qu_L, re_L, K)
            self.logger.info(retI2T)
            self.logger.info(retT2I)
            now = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
            log_name = 'log/' + self.opt.data_name + '_' + \
                       str(self.opt.batch_size) + '_' + str(self.opt.bit) + '_' + \
                       str(self.opt.dw) + '_' + str(self.opt.cw) \
                       + '_' + str(self.opt.a1) + '_' + str(self.opt.a2) + '_' + \
                       str(self.opt.knn_number) + '_' + str(self.opt.scale) + \
                       '.txt'
            fi = open(log_name, 'a')
            fi.write('--------------------Evaluation: Calculate top MAP-------------------')
            fi.write('\n')
            fi.write('MAP@50 of Image to Text: %.3f, MAP of Text to Image: %.3f' % (MAP_I2T, MAP_T2I))
            fi.write('\n')
            fi.write('MAP@All of Image to Text: %.3f, MAP of Text to Image: %.3f' % (MAP_I2Ta, MAP_T2Ia))
            fi.write('\n')
            fi.write('MAP@All_1 of Image to Text: %.3f, MAP of Text to Image: %.3f' % (MAP_I2Ta_1, MAP_T2Ia_1))
            fi.write('\n')
            fi.write(str(retI2T.cpu().numpy()))
            fi.write('\n')
            fi.write(str(retT2I.cpu().numpy()))
            fi.close()

            log_name = 'result/' + self.opt.data_name + 'batch_size' + \
                       str(self.opt.batch_size) + 'bit' + str(self.opt.bit) + '.txt'
            
            parameter = 'dw:' + str(self.opt.dw) + ' ' + 'cw:' + str(self.opt.cw) \
                        + ' ' + 'a1:' + str(self.opt.a1) + ' ' + 'a2:' + str(self.opt.a2) + ' ' \
                        + 'knn:' + str(self.opt.knn_number) + ' ' + 'scale:' + \
                        str(self.opt.scale)
                      
            fi = open(log_name, 'a')
            fi.write(parameter)            
            fi.write('\n')
            fi.write('MAP@50 of Im2Te: %.3f, MAP of Te2Im: %.3f' % (MAP_I2T, MAP_T2I))
            fi.write('\n')
            fi.write('MAP@All of Im2Te: %.3f, MAP of Te2Im: %.3f' % (MAP_I2Ta, MAP_T2Ia))
            fi.write('\n')
            fi.write('MAP@All_1 of Im2Te: %.3f, MAP of Te2Im: %.3f' % (MAP_I2Ta_1, MAP_T2Ia_1))
            fi.write('\n')
            fi.close()            
            
        return num_epoch

    def save_checkpoints(self, step,
                         best=False):
        file_name = '%s_%d_bit_latest.pth' % (self.opt.data_name, self.opt.bit)
        if best:
            file_name = '%s_%d_bit_best_epoch.pth' % (self.opt.data_name, self.opt.bit)
        ckp_path = osp.join(self.opt.save_model_path, file_name)
        obj = {
            'ImgNet': self.CodeNet_I.state_dict(),
            'TxtNet': self.CodeNet_T.state_dict(),
            'step': step,
        }
        torch.save(obj, ckp_path)
        self.logger.info('**********Save the trained model successfully.**********')

    def load_checkpoints(self):
        file_name = '%s_%d_bit_best_epoch.pth' % (self.opt.data_name, self.opt.bit)
        ckp_path = osp.join(self.opt.save_model_path, file_name)
        try:
            obj = torch.load(ckp_path, map_location=lambda storage, loc: storage.cuda())
            self.logger.info('**************** Load checkpoint %s ****************' % ckp_path)
        except IOError:
            self.logger.error('********** No checkpoint %s!*********' % ckp_path)
            return
        self.CodeNet_I.load_state_dict(obj['ImgNet'])
        self.CodeNet_T.load_state_dict(obj['TxtNet'])
        self.logger.info('********** The loaded model has been trained for %d epochs.*********' % obj['step'])





