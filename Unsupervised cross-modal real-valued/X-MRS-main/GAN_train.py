import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from data_loader import get_loader
from GAN_args import get_parser, save_opts
from GAN_models import *
from torch.optim import lr_scheduler
from tqdm import tqdm
import torch.nn.functional as F
from torch.autograd import Variable
from datetime import datetime

from models import load_image_encoder, rank_i2t, extract_image_encoder_from_FoodSpaceNet
from utils import load_torch_model
from utils import exists, make_dir

# =============================================================================
parser = get_parser()
opts = parser.parse_args()
if opts.device is None:
    device = [0]
else:
    device = opts.device

if opts.W_IMG_IMG_COS_LOSS < 0.0001:
    print("Do not use cycle loss")
else:
    print("Use cycle loss")

language = {0: "en", 1: "de-en", 2: "ru-en", 3: "de", 4: "ru", 5: "fr"}
opts.encoder_model = glob.glob(os.path.join(opts.encoder_dir,'models','model_BEST_REC*_image_encoder*'))[0]
opts.embedding_file_train = glob.glob(os.path.join(opts.encoder_dir,'foodSpace*train_'+language[opts.language]+'*'))[0]  # Using English (EN) embeddings as default
opts.embedding_file_val = glob.glob(os.path.join(opts.encoder_dir,'foodSpace*val_'+language[opts.language]+'*'))[0]
opts.embedding_file_test = glob.glob(os.path.join(opts.encoder_dir,'foodSpace*test_'+language[opts.language]+'*'))[0]

#load models
netG = torch.nn.DataParallel(G_NET().cuda())
netD = torch.nn.DataParallel(D_NET128().cuda())

## load loss functions
weights_class = torch.Tensor(opts.numClasses).fill_(1)
weights_class[0] = 0
class_criterion = nn.CrossEntropyLoss(weight=weights_class).cuda()

GAN_criterion = nn.BCELoss().cuda()

nz = opts.Z_DIM
real_labels = Variable(torch.FloatTensor(opts.batch_size).fill_(1)).cuda()
fake_labels = Variable(torch.FloatTensor(opts.batch_size).fill_(0)).cuda()

model_list = [netG, netD]

optimizer_G = torch.optim.Adam(netG.parameters(), lr=opts.lr, betas=(0.5, 0.999), weight_decay=opts.weight_decay)

optimizer_D = torch.optim.Adam(netD.parameters(), lr=opts.lr, betas=(0.5, 0.999), weight_decay=opts.weight_decay)

label = list(range(0, opts.batch_size))
label.extend(label)
label = np.array(label)
label = torch.tensor(label).cuda().long()

#------------------------------------------------------------
# load retrieval model
image_encoder = load_image_encoder(opts.encoder_model, opts)
if not image_encoder:
    raise RuntimeError("no image encoder")
image_encoder = nn.DataParallel(image_encoder)
image_encoder.eval()

upsample_layer = nn.DataParallel(nn.Upsample(size=(224, 224), mode="bilinear", align_corners=True).cuda()).eval()
#------------------------------------------------------------

method = 'GAN'
save_folder = os.path.join(opts.encoder_dir, "GAN_runs", method + "_" + datetime.now().strftime("%Y%m%d-%H%M%S"))
make_dir(save_folder)
epoch_trace_f_file = os.path.join(save_folder, "trace_" + method + ".csv")
print(save_folder)

opts_file = os.path.join(save_folder, "training_options.json")
save_opts(opts, opts_file)

with open(epoch_trace_f_file, "w") as f:
    f.write("epoch,lr,I2R,R@1,R@5,R@10\n")


def main():
    if opts.resume_G and exists(opts.resume_G):
        ret = load_torch_model(netG, opts.resume_G)
        if not ret:
            raise IOError("pretrained G not loaded")
    if opts.resume_D and exists(opts.resume_D):
        ret = load_torch_model(netD, opts.resume_D)
        if not ret:
            raise IOError("pretrained D not loaded")

    # data preparation, loaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip()])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224)])

    cudnn.benchmark = True

    # preparing the training laoder
    train_loader = get_loader(opts.img_path, opts.data_path, 'train', opts.embedding_file_train, train_transform,
                              batch_size=opts.batch_size, shuffle=True,
                              num_workers=opts.workers, pin_memory=True)
    print('Training loader prepared.')

    # preparing validation loader
    val_loader = get_loader(opts.img_path, opts.data_path, 'val', opts.embedding_file_val, val_transform,
                              batch_size=opts.batch_size, shuffle=False,
                              num_workers=opts.workers, pin_memory=True, drop_last=False)
    print('Validation loader prepared.')

    best_val_i2t = {1: 0.0, 5: 0.0, 10: 0.0}
    best_epoch_i2t = 0
    best_medR = 10000.0

    lr_drop_times = 0

    for epoch in range(0, opts.epochs):

        lossG, lossD = train(train_loader, epoch)

        medR_i2t, recall_i2t = validate(val_loader)
        with open(epoch_trace_f_file, "a") as f:
            lr = optimizer_G.param_groups[0]['lr']
            f.write("{},{},{},{},{},{}\n".format \
                        (epoch, lr, medR_i2t, recall_i2t[1], recall_i2t[5], recall_i2t[10]))

        for keys in best_val_i2t:
            if recall_i2t[keys] > best_val_i2t[keys]:
                best_val_i2t = recall_i2t
                best_medR = medR_i2t
                best_epoch_i2t = epoch
                model_num = 1
                for model_n in model_list:
                    filename = save_folder + '/model_e%03d_v%d.pkl' % (epoch + 1, model_num)
                    torch.save(model_n.state_dict(), filename)
                    model_num += 1
                break
        print("best: ", best_epoch_i2t+1, best_medR, best_val_i2t)
        print('params lr: %f' % optimizer_G.param_groups[0]['lr'])

        if (epoch+1) % 100 == 0:
            model_num = 1
            for model_n in model_list:
                filename = save_folder + '/model_e%03d_v%d.pkl' % (epoch + 1, model_num)
                if not os.path.exists(filename):
                    filename = save_folder + '/model_e%03d_v%d_regular_check_point.pkl' % (epoch + 1, model_num)
                    torch.save(model_n.state_dict(), filename)
                model_num += 1

        if opts.auto_drop_lr:
            if lr_drop_times < opts.auto_drop_times and abs(best_epoch_i2t-epoch) > opts.auto_drop_after_epochs:
                lr_drop_times += 1
                optimizer_G.param_groups[0]['lr'] = optimizer_G.param_groups[0]['lr'] / 2  # 0.00001
                optimizer_D.param_groups[0]['lr'] = optimizer_D.param_groups[0]['lr'] / 2  # 0.00001
        else:
            if opts.drop_G_lr_epoch > 0 and epoch == opts.drop_G_lr_epoch:
                optimizer_G.param_groups[0]['lr'] = optimizer_G.param_groups[0]['lr'] / 10 #0.00001
            if opts.drop_D_lr_epoch > 0 and epoch == opts.drop_D_lr_epoch:
                optimizer_D.param_groups[0]['lr'] = optimizer_D.param_groups[0]['lr'] / 10 #0.00001

    model_num = 1
    for model_n in model_list:
        filename = save_folder + '/model_e%03d_v%d_last_epoch.pkl' % (opts.epochs, model_num)
        if not os.path.exists(filename):
            torch.save(model_n.state_dict(), filename)
        model_num += 1


def train_Dnet(real_imgs, fake_imgs, mu, label_class):
    real_logits = netD(real_imgs, mu.detach())
    fake_logits = netD(fake_imgs.detach(), mu.detach())

    lossD_real = GAN_criterion(real_logits[0], real_labels)
    lossD_fake = GAN_criterion(fake_logits[0], fake_labels)

    lossD_real_cond = class_criterion(real_logits[1], label_class)
    lossD_fake_cond = class_criterion(fake_logits[1], label_class)

    lossD = lossD_real + lossD_fake + lossD_real_cond + lossD_fake_cond
    return lossD


def KL_loss(mu, logvar):
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD


def train_Gnet(fake_imgs, mu, logvar, label_class, real_image_feat=None, w_cycle_image=None, rec_feat=None, w_cycle_recipe=None):
    fake_logits = netD(fake_imgs, mu)

    lossG_fake = GAN_criterion(fake_logits[0], real_labels)

    lossG_fake_cond = class_criterion(fake_logits[1], label_class)

    lossG = lossG_fake + lossG_fake_cond

    kl_loss = KL_loss(mu, logvar) * 2
    lossG1 = kl_loss + lossG
    lossG = lossG1

    # cycle loss
    if (real_image_feat is not None and w_cycle_image is not None and w_cycle_image > 0) or (rec_feat is not None and w_cycle_recipe is not None and w_cycle_recipe > 0):
        fake_imgs = upsample_layer(fake_imgs)
        #fake_imgs = F.interpolate(fake_imgs, size=(224,224), mode="bilinear", align_corners=True)
        fake_feat = image_encoder(fake_imgs)
    if (real_image_feat is not None and w_cycle_image is not None and w_cycle_image > 0):
        loss_img_img = nn.CosineEmbeddingLoss(0.3)(fake_feat, real_image_feat, torch.ones(fake_feat.shape[0]).cuda())
        lossG = lossG + w_cycle_image * loss_img_img
    else:
        loss_img_img = None
    if (rec_feat is not None and w_cycle_recipe is not None and w_cycle_recipe > 0):
        loss_img_recipe = nn.CosineEmbeddingLoss(0.3)(fake_feat, rec_feat, torch.ones(fake_feat.shape[0]).cuda())
        lossG = lossG + w_cycle_recipe * loss_img_recipe
    else:
        loss_img_recipe = None

    return lossG, [lossG1, loss_img_img, loss_img_recipe]


def train(train_loader, epoch):
    G_losses = AverageMeter()
    D_losses = AverageMeter()
    Img_Img_losses = AverageMeter()
    Img_Rec_losses = AverageMeter()
    G1_losses = AverageMeter()


    netG.train()
    netD.train()

    for i, data in enumerate(tqdm(train_loader)):
        #images = data[0]
        class_label = data[1].cuda()
        real_images = data[2].cuda()
        recipes_embs = data[3].cuda()
        if opts.W_IMG_IMG_COS_LOSS > 0:
            img_embs = data[4].cuda()
        else:
            img_embs = None
        ###############################################################
        # text2img
        ###############################################################
        #noise.data.normal_(0, 1)
        noise = torch.randn(recipes_embs.size(0), nz).cuda()
        fake_imgs, mu, logvar = netG(noise, recipes_embs)

        lossD = train_Dnet(real_images, fake_imgs, mu, class_label)

        D_losses.update(lossD.item(), data[0].size(0))

        optimizer_D.zero_grad()
        lossD.backward()
        optimizer_D.step()

        lossG, other_losses = train_Gnet(fake_imgs, mu, logvar, class_label, img_embs, opts.W_IMG_IMG_COS_LOSS, recipes_embs, opts.W_IMG_REC_COS_LOSS)

        G_losses.update(lossG.item(), data[1].size(0))
        G1_losses.update(other_losses[0].item(), data[1].size(0))
        if opts.W_IMG_IMG_COS_LOSS > 0:
            Img_Img_losses.update(other_losses[1].item(), data[1].size(0))
        if opts.W_IMG_REC_COS_LOSS > 0:
            Img_Rec_losses.update(other_losses[2].item(), data[1].size(0))

        optimizer_G.zero_grad()
        lossG.backward()
        optimizer_G.step()

    print(epoch)
    print('Epoch: {0}  '
          'G loss {G_loss.val:.4f} ({G_loss.avg:.4f}),  '
          'D loss {D_loss.val:.4f} ({D_loss.avg:.4f}),  '
          'G1 loss {G1_loss.val:.4f} ({G1_loss.avg:.4f}),  '
          'Cyc_1 loss {Cyc_1_loss.val:.4f} ({Cyc_1_loss.avg:.4f}),  '
          'Cyc_2 loss {Cyc_2_loss.val:.4f} ({Cyc_2_loss.avg:.4f})  '
        .format(epoch, G_loss=G_losses, D_loss=D_losses, G1_loss=G1_losses, Cyc_1_loss=Img_Img_losses, Cyc_2_loss=Img_Rec_losses))

    return lossG.item(), lossD.item()


def validate(val_loader):
    # switch to evaluate mode
    netG.eval()

    for i, data in enumerate(tqdm(val_loader)):
        recipes_embs = data[3].cuda()
        noise = torch.randn(recipes_embs.size(0), nz).cuda()
        with torch.no_grad():
            fake_images, _, _ = netG(noise, recipes_embs)
            fake_images = upsample_layer(fake_images) #F.interpolate(fake_images, size=(224, 224), mode="bilinear", align_corners=True)
            fake_feat = image_encoder(fake_images)

            if i == 0:
                data0 = fake_feat.data.cpu().numpy()
                data1 = recipes_embs.data.cpu().numpy()
            else:
                data0 = np.concatenate((data0, fake_feat.data.cpu().numpy()), axis=0)
                data1 = np.concatenate((data1, recipes_embs.data.cpu().numpy()), axis=0)

    medR_i2t, recall_i2t = rank_i2t(opts.seed, data0, data1)
    print('I2T Val medR {medR:.4f}\t'
          'Recall {recall}'.format(medR=medR_i2t, recall=recall_i2t))

    return medR_i2t, recall_i2t


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
