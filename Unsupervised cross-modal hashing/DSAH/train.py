import os.path as osp

import torch
import torch.nn.functional as F
from torch.autograd import Variable

import datasets
import settings
from metric import compress_wiki, compress, calculate_top_map, calculate_map, p_topK
from models import ImgNet, TxtNet

torch.manual_seed(1)
torch.cuda.manual_seed_all(1)


class Session:
    def __init__(self):
        self.logger = settings.logger

        torch.cuda.set_device(settings.GPU_ID)

        if settings.DATASET == "WIKI":
            self.train_dataset = datasets.WIKI(root=settings.DATA_DIR, train=True,
                                               transform=datasets.wiki_train_transform)
            self.test_dataset = datasets.WIKI(root=settings.DATA_DIR, train=False,
                                              transform=datasets.wiki_test_transform)
            self.database_dataset = datasets.WIKI(root=settings.DATA_DIR, train=True,
                                                  transform=datasets.wiki_test_transform)

        if settings.DATASET == "MIRFlickr":
            self.train_dataset = datasets.MIRFlickr(train=True, transform=datasets.mir_train_transform)
            self.test_dataset = datasets.MIRFlickr(train=False, database=False, transform=datasets.mir_test_transform)
            self.database_dataset = datasets.MIRFlickr(train=False, database=True,
                                                       transform=datasets.mir_test_transform)

        if settings.DATASET == "NUSWIDE":
            self.train_dataset = datasets.NUSWIDE(train=True, transform=datasets.nus_train_transform)
            self.test_dataset = datasets.NUSWIDE(train=False, database=False, transform=datasets.nus_test_transform)
            self.database_dataset = datasets.NUSWIDE(train=False, database=True, transform=datasets.nus_test_transform)

        # Data Loader (Input Pipeline)
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                        batch_size=settings.BATCH_SIZE,
                                                        shuffle=True,
                                                        num_workers=settings.NUM_WORKERS,
                                                        drop_last=True)

        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                                       batch_size=settings.BATCH_SIZE,
                                                       shuffle=False,
                                                       num_workers=settings.NUM_WORKERS)

        self.database_loader = torch.utils.data.DataLoader(dataset=self.database_dataset,
                                                           batch_size=settings.BATCH_SIZE,
                                                           shuffle=False,
                                                           num_workers=settings.NUM_WORKERS)
        txt_feat_len = datasets.txt_feat_len
        self.CodeNet_I = ImgNet(code_len=settings.CODE_LEN, txt_feat_len=txt_feat_len)
        self.FeatNet_I = ImgNet(code_len=settings.CODE_LEN, txt_feat_len=txt_feat_len)

        self.CodeNet_T = TxtNet(code_len=settings.CODE_LEN, txt_feat_len=txt_feat_len)

        if settings.DATASET == "WIKI":
            self.opt_I = torch.optim.SGD(self.CodeNet_I.parameters(), lr=settings.LR_IMG, momentum=settings.MOMENTUM,
                                         weight_decay=settings.WEIGHT_DECAY, nesterov=True)

        if settings.DATASET == "MIRFlickr" or settings.DATASET == "NUSWIDE":
            self.opt_I = torch.optim.SGD(self.CodeNet_I.parameters(), lr=settings.LR_IMG, momentum=settings.MOMENTUM,
                                         weight_decay=settings.WEIGHT_DECAY, nesterov=True)

        self.opt_T = torch.optim.SGD(self.CodeNet_T.parameters(), lr=settings.LR_TXT, momentum=settings.MOMENTUM,
                                     weight_decay=settings.WEIGHT_DECAY, nesterov=True)
        self.best = 0

    def train(self, epoch):
        self.FeatNet_I.cuda().eval()

        self.CodeNet_I.cuda().train()
        self.CodeNet_T.cuda().train()

        self.CodeNet_I.set_alpha(epoch)
        self.CodeNet_T.set_alpha(epoch)

        self.logger.info('Epoch [%d/%d], alpha for ImgNet: %.3f, alpha for TxtNet: %.3f' % (
            epoch + 1, settings.NUM_EPOCH, self.CodeNet_I.alpha, self.CodeNet_T.alpha))

        for idx, (img, txt, labels, _) in enumerate(self.train_loader):
            batch_size = img.size(0)
            img = Variable(img.cuda())
            txt = Variable(torch.FloatTensor(txt.numpy()).cuda())
            self.opt_I.zero_grad()
            self.opt_T.zero_grad()

            (_, F_I), _, _, _ = self.FeatNet_I(img)
            F_T = txt
            _, hid_I, code_I, decoded_t = self.CodeNet_I(img)
            _, hid_T, code_T, decoded_i = self.CodeNet_T(txt)
            F_I = F.normalize(F_I)
            S_I = F_I.mm(F_I.t())
            S_I = S_I * 2 - 1
            F_T = F.normalize(F_T)
            S_T = F_T.mm(F_T.t())
            S_T = S_T * 2 - 1

            B_I = F.normalize(code_I)
            B_T = F.normalize(code_T)

            BI_BI = B_I.mm(B_I.t())
            BT_BT = B_T.mm(B_T.t())
            BI_BT = B_I.mm(B_T.t())

            S_tilde = settings.ALPHA * S_I + (1 - settings.ALPHA) * S_T
            S = settings.K * S_tilde

            loss1 = F.mse_loss(BT_BT, S)
            loss2 = F.mse_loss(BI_BT, S)
            loss3 = F.mse_loss(BI_BI, S)
            loss31 = F.mse_loss(BI_BI, settings.K * S_I)
            loss32 = F.mse_loss(BT_BT, settings.K * S_T)

            diagonal = BI_BT.diagonal()
            all_1 = torch.rand((batch_size)).fill_(1).cuda()
            loss4 = F.mse_loss(diagonal, settings.K * all_1)
            loss5 = F.mse_loss(decoded_i, F_I)
            loss6 = F.mse_loss(decoded_t, F_T)
            loss7 = F.mse_loss(BI_BT, BI_BT.t())
            loss = 1 * loss1 + 1 * loss2 + 1 * loss3 + 1 * loss4 + 1 * loss5 + 1 * loss6 + 2 * loss7 + settings.ETA * (
                        loss31 + loss32)
            # MIRFlickr
            # 16 bit 863 846; 860 849; 856 841
            # 32 bit 877 860; 867 858; 873 856
            # 64 bit 889 883; 886 888; 895 881
            # 128bit 903 882; 897 881; 907 877; 901 885

            # Wiki
            # 128bit 435 662;432 661;434 667; 433，663;
            # 64 bit 440 658;438 660;433 660;
            # 32 bit 430 650;422 658;420 665;
            # 16 bit 394 617;416 644;416 639;

            # NUS-WIDE
            # 128bit
            # 64 bit
            # 32 bit
            # 16 bit
            loss.backward()
            self.opt_I.step()
            self.opt_T.step()
            if (idx + 1) % (len(self.train_dataset) // settings.BATCH_SIZE / settings.EPOCH_INTERVAL) == 0:
                self.logger.info(
                    'Epoch [%d/%d], Iter [%d/%d] '
                    'Loss1: %.4f Loss2: %.4f Loss3: %.4f '
                    'Loss4: %.4f '
                    'Loss5: %.4f Loss6: %.4f '
                    'Loss7: %.4f '
                    'Total Loss: %.4f'
                    % (
                        epoch + 1, settings.NUM_EPOCH, idx + 1,
                        len(self.train_dataset) // settings.BATCH_SIZE,
                        loss1.item(), loss2.item(), loss3.item(),
                        loss4.item(),
                        loss5.item(), loss6.item(),
                        loss7.item(),
                        loss.item()))

    def eval(self, step=0, last=False):

        # Change model to 'eval' mode (BN uses moving mean/var).
        self.CodeNet_I.eval().cuda()
        self.CodeNet_T.eval().cuda()
        if settings.DATASET == "WIKI":
            re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L = compress_wiki(self.database_loader, self.test_loader,
                                                                   self.CodeNet_I, self.CodeNet_T,
                                                                   self.database_dataset, self.test_dataset)
            K = [1, 200, 400, 500, 1000, 1500, 2000]
        if settings.DATASET == "MIRFlickr" or settings.DATASET == "NUSWIDE":
            re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L = compress(self.database_loader, self.test_loader, self.CodeNet_I,
                                                              self.CodeNet_T, self.database_dataset, self.test_dataset)
            K = [1, 200, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
        if settings.EVAL:
            MAP_I2T = calculate_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L)
            MAP_T2I = calculate_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L)
            self.logger.info('--------------------Evaluation: Calculate top MAP-------------------')
            self.logger.info('MAP of Image to Text: %.3f, MAP of Text to Image: %.3f' % (MAP_I2T, MAP_T2I))
            self.logger.info('--------------------------------------------------------------------')
            retI2T = p_topK(qu_BI, re_BT, qu_L, re_L, K)
            retT2I = p_topK(qu_BT, re_BI, qu_L, re_L, K)
            self.logger.info(retI2T)
            self.logger.info(retT2I)
        MAP_I2T = calculate_top_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L, topk=50)
        MAP_T2I = calculate_top_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L, topk=50)
        self.logger.info('--------------------Evaluation: Calculate top MAP-------------------')
        self.logger.info('MAP of Image to Text: %.3f, MAP of Text to Image: %.3f' % (MAP_I2T, MAP_T2I))
        self.logger.info('--------------------------------------------------------------------')
        if MAP_I2T + MAP_T2I > self.best and not settings.EVAL:
            self.save_checkpoints(step=step, best=True)
            self.best = MAP_T2I + MAP_I2T
            self.logger.info("#########is best:%.3f #########" % self.best)

    def save_checkpoints(self, step, file_name='%s_%d_bit_latest.pth' % (settings.DATASET, settings.CODE_LEN),
                         best=False):
        if best:
            file_name = '%s_%d_bit_best_epoch.pth' % (settings.DATASET, settings.CODE_LEN)
        ckp_path = osp.join(settings.MODEL_DIR, file_name)
        obj = {
            'ImgNet': self.CodeNet_I.state_dict(),
            'TxtNet': self.CodeNet_T.state_dict(),
            'step': step,
        }
        torch.save(obj, ckp_path)
        self.logger.info('**********Save the trained model successfully.**********')

    def load_checkpoints(self, file_name='%s_%d_bit_best_epoch.pth' % (settings.DATASET, settings.CODE_LEN)):
        ckp_path = osp.join(settings.MODEL_DIR, file_name)
        try:
            obj = torch.load(ckp_path, map_location=lambda storage, loc: storage.cuda())
            self.logger.info('**************** Load checkpoint %s ****************' % ckp_path)
        except IOError:
            self.logger.error('********** No checkpoint %s!*********' % ckp_path)
            return
        self.CodeNet_I.load_state_dict(obj['ImgNet'])
        self.CodeNet_T.load_state_dict(obj['TxtNet'])
        self.logger.info('********** The loaded model has been trained for %d epochs.*********' % obj['step'])


def main():
    sess = Session()

    if settings.EVAL == True:
        sess.load_checkpoints()
        sess.eval()

    else:
        for epoch in range(settings.NUM_EPOCH):
            # train the Model
            sess.train(epoch)
            # eval the Model
            if (epoch + 1) % settings.EVAL_INTERVAL == 0:
                sess.eval(step=epoch + 1)
            # save the model
        settings.EVAL = True
        sess.logger.info('---------------------------Test------------------------')
        sess.load_checkpoints()
        sess.eval()


if __name__ == '__main__':
    main()
