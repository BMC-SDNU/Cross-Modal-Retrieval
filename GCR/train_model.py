from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torchvision
import time
import copy

from evaluate import fx_calc_map_label
import numpy as np


def calc_loss(view1_feature, view2_feature, view1_predict, view2_predict, labels_1, labels_2, gama=0.5):

    term1 = ((view1_predict - labels_1.float()) ** 2).sum(1).sqrt().mean() + (
                (view2_predict - labels_2.float()) ** 2).sum(1).sqrt().mean()


    term2 = ((view1_feature - view2_feature) ** 2).sum(1).sqrt().mean()

    im_loss = term1 + gama * term2
    return im_loss

def calc_loss2(view1_feature, view2_feature, view1_predict, labels_1, tau=12):

    term1 = ((view1_predict - labels_1.float()) ** 2).sum(
        1).sqrt().mean()

    term2 = ((view1_feature - view2_feature) ** 2).sum(1).sqrt().mean()

    im_loss = term1 + tau*term2
    return im_loss


import scipy.sparse as sp


def normalize_adj(adj, mask=None):

    if mask is None:
        mask = adj

    rowsum = np.sum(adj, axis=1)  # D
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()  # D^-0.5
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)  # D^-0.5
    return np.matmul(np.matmul(d_mat_inv_sqrt, mask),
                     d_mat_inv_sqrt)

cons = 1
def generate_adj(labels):
    y_single = np.argmax(np.vstack((labels)),
                         1)
    y_single = y_single.reshape(y_single.shape[0], 1)
    mask_initial = np.matmul(y_single, np.ones([1, y_single.shape[0]], dtype=np.int32)) - \
                   np.matmul(np.ones([y_single.shape[0], 1], dtype=np.int32), np.transpose(y_single))
    adj = cons * (np.equal(mask_initial, np.zeros_like(mask_initial)).astype(np.float32) - np.identity(
        mask_initial.shape[0]).astype(np.float32)) + np.identity(mask_initial.shape[0]).astype(
        np.float32)
    mask = cons * (np.equal(mask_initial, np.zeros_like(mask_initial)).astype(np.float32) - np.identity(
        mask_initial.shape[0]).astype(np.float32)) + np.identity(mask_initial.shape[0]).astype(
        np.float32)
    mask = normalize_adj(adj, mask)
    adj = torch.from_numpy(mask)
    return  adj

def train_gse(model, data_loaders, optimizer, num_epochs=100):

    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 20)

        for phase in ['train']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects_img = 0
            running_corrects_txt = 0
            index = 0

            for imgs, txts, labels in data_loaders[phase]:
                index = index + 1


                if torch.sum(imgs != imgs) > 1 or torch.sum(txts != txts) > 1:
                    print("Data contains Nan.")

                # zero the parameter gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):

                    adj = generate_adj(labels)
                    if torch.cuda.is_available():
                        imgs = imgs.cuda()
                        txts = txts.cuda()
                        labels = labels.cuda()
                        adj = adj.cuda()

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # Forward
                    view1_feature, view2_feature, view1_predict, view2_predict = model(imgs, txts, adj,
                                                                                             adjflag=True)

                    loss = calc_loss(view1_feature, view2_feature, view1_predict,
                                     view2_predict, labels, labels)

                    img_preds = view1_predict
                    txt_preds = view2_predict

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item()
                running_corrects_img += torch.sum(torch.argmax(img_preds, dim=1) == torch.argmax(labels, dim=1))
                running_corrects_txt += torch.sum(torch.argmax(txt_preds, dim=1) == torch.argmax(labels, dim=1))

            epoch_loss = running_loss / len(data_loaders[phase].dataset)

            t_imgsa, t_txtsa, t_imgs, t_txts, t_labels = [], [], [], [], []
            with torch.no_grad():
                for imgs, txts, labels in data_loaders['test']:

                    adj = generate_adj(labels)

                    if torch.cuda.is_available():
                        imgs = imgs.cuda()
                        txts = txts.cuda()
                        labels = labels.cuda()
                        adj = adj.cuda()

                    t_view1_feature, t_view2_feature, _, _ = model(imgs, txts, adj, adjflag=False)
                    t_view1_featurea, t_view2_featurea, _, _ = model(imgs, txts, adj, adjflag=True)
                    t_imgs.append(t_view1_feature.cpu().numpy())
                    t_txts.append(t_view2_feature.cpu().numpy())
                    t_imgsa.append(t_view1_featurea.cpu().numpy())
                    t_txtsa.append(t_view2_featurea.cpu().numpy())
                    t_labels.append(labels.cpu().numpy())
            t_imgs = np.concatenate(t_imgs)
            t_txts = np.concatenate(t_txts)
            t_imgsa = np.concatenate(t_imgsa)
            t_txtsa = np.concatenate(t_txtsa)
            t_labels = np.concatenate(t_labels).argmax(1)
            img2text = fx_calc_map_label(t_imgs, t_txts, t_labels)
            txt2img = fx_calc_map_label(t_txts, t_imgs, t_labels)

            img2texta = fx_calc_map_label(t_imgsa, t_txtsa, t_labels)
            txt2imga = fx_calc_map_label(t_txtsa, t_imgsa, t_labels)

            print('{} Loss: {:.4f} Img2Txt: {:.4f}  Txt2Img: {:.4f}'.format(phase, epoch_loss, img2text, txt2img))
            print('{} Loss: {:.4f} Img2Txta: {:.4f}  Txt2Imga: {:.4f}'.format(phase, epoch_loss, img2texta, txt2imga))

            # deep copy the model
            if phase == 'train' and (img2texta + txt2imga) / 2. > best_acc:
                best_acc = (img2texta + txt2imga) / 2.
                best_model_wts = copy.deepcopy(model.state_dict())



    print('Best average ACC: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), "save/pre-model.pt")

    return model

def train_gan(model_genT, model_disT, model_gen, model_dis, model, data_loaders, optimizer_genT, optimizer_disT,
                optimizer_gen, optimizer_dis,  num_epochsGAN=100):
    best_model_wts_adv = copy.deepcopy(model_gen.state_dict())
    best_model_wts_advT = copy.deepcopy(model_genT.state_dict())
    best_acc_adv_text = 0.0
    best_acc_adv_image = 0.0

    model.eval()
    
    idx = 0
    for epoch in range(num_epochsGAN):
        print('Epoch {}/{}'.format(epoch + 1, num_epochsGAN))
        print('-' * 20)
        idx = idx + 1

        for phase in ['train', 'test']:
            if phase == 'train':
                # Set model to training mode

                model_genT.train()
                model_gen.train()
                model_disT.train()
                model_dis.train()
            else:
                # Set model to evaluate mode
                model_genT.eval()
                model_gen.eval()
                model_disT.eval()
                model_dis.eval()

            running_loss_g = 0.0
            running_loss_d = 0.0
            running_loss_gT = 0.0
            running_loss_dT = 0.0


            # Iterate over data.
            for imgs, txts, labels in data_loaders[phase]:

                if torch.sum(imgs != imgs) > 1 or torch.sum(txts != txts) > 1:
                    print("Data contains Nan.")



                with torch.set_grad_enabled(phase == 'train'):

                    adj = generate_adj(labels)

                    if torch.cuda.is_available():
                        imgs = imgs.cuda()
                        txts = txts.cuda()
                        labels = labels.cuda()
                        adj = adj.cuda()

                    #generate constraint signal
                    constraint1, constraint2, view1_predict, view2_predict = model(imgs, txts, adj, adjflag=True)

                    if True:
                        optimizer_dis.zero_grad()

                        genimg = model_gen(imgs)
                        score_f = model_dis(genimg)
                        gentxt = model_genT(txts)
                        score_r = model_dis(gentxt)

                        dloss = - (torch.log(score_r) + torch.log(1 - score_f))
                        dloss = dloss.sum().mean()


                        if phase == 'train':
                            dloss.backward()
                            running_loss_d += dloss.item()
                            optimizer_dis.step()

                    optimizer_gen.zero_grad()

                    if True:

                        genimg = model_gen(imgs)
                        score_f = model_dis(genimg)

                        predloss = calc_loss2(genimg, constraint1.detach(), model.shareClassifier(genimg),
                                              labels,3)

                        gloss = -torch.log(score_f) + 1.2 * torch.log(
                            predloss)
                        gloss = gloss.sum().mean()

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            gloss.backward()
                            running_loss_g += gloss.item()
                            optimizer_gen.step()

                    ######################
                    if True:
                        optimizer_disT.zero_grad()
                        gentxt = model_genT(txts)
                        score_f = model_disT(gentxt)
                        genimg = model_gen(imgs)
                        score_r = model_disT(genimg)

                        dlossT = - (torch.log(score_r) + torch.log(1 - score_f))
                        dlossT = dlossT.sum().mean()

                        if phase == 'train':
                            dlossT.backward()
                            running_loss_dT += dlossT.item()
                            optimizer_disT.step()

                    optimizer_genT.zero_grad()
                    if True:
                        gentxt = model_genT(txts)
                        score_f = model_disT(gentxt)

                        predloss = calc_loss2(gentxt, constraint2.detach(), model.shareClassifier(gentxt),
                                              labels,12)

                        glossT = -torch.log(score_f) + 1.0 * torch.log(
                            predloss)
                        glossT = glossT.sum().mean()

                        if phase == 'train':
                            glossT.backward()
                            running_loss_gT += glossT.item()
                            optimizer_genT.step()



            epoch_loss_g = running_loss_g / len(data_loaders[phase].dataset)
            epoch_loss_d = running_loss_d / len(data_loaders[phase].dataset)
            epoch_loss_gT = running_loss_gT / len(data_loaders[phase].dataset)
            epoch_loss_dT = running_loss_dT / len(data_loaders[phase].dataset)

            t_imgs, t_txts, t_labels = [], [], []
            t_imgs_adv = []
            t_txts_adv = []
            with torch.no_grad():
                for imgs, txts, labels in data_loaders['test']:

                    adj = generate_adj(labels)
                    if torch.cuda.is_available():
                        imgs = imgs.cuda()
                        txts = txts.cuda()
                        labels = labels.cuda()
                        adj = adj.cuda()
                    t_view1_feature, t_view2_feature, _, _= model(imgs, txts, adj, adjflag=True)
                    t_imgs.append(t_view1_feature.cpu().numpy())
                    t_txts.append(t_view2_feature.cpu().numpy())
                    t_labels.append(labels.cpu().numpy())
                    t_imgs_adv.append(model_gen(imgs).cpu().numpy())
                    t_txts_adv.append(model_genT(txts).cpu().numpy())
            t_imgs = np.concatenate(t_imgs)
            t_imgs_adv = np.concatenate(t_imgs_adv)
            t_txts_adv = np.concatenate(t_txts_adv)
            t_txts = np.concatenate(t_txts)
            t_labels = np.concatenate(t_labels).argmax(1)
            img2text = fx_calc_map_label(t_imgs, t_txts, t_labels)
            txt2img = fx_calc_map_label(t_txts, t_imgs, t_labels)

            print('epoch_loss_g----{} Loss_Image: {:.4f} Loss_Text: {:.4f} ori - Img2Txt: {:.4f}  Txt2Img: {:.4f}'.format(phase, epoch_loss_g,
                                                                                                  epoch_loss_gT,img2text, txt2img))

            img2text_adv = fx_calc_map_label(t_imgs_adv, t_txts_adv, t_labels)
            txt2img_adv = fx_calc_map_label(t_txts_adv, t_imgs_adv, t_labels)

            print('epoch_loss_d----{} Loss_Image: {:.4f} Loss_Text: {:.4f} adv Img2Txt: {:.4f}  Txt2Img: {:.4f}'.format(phase, epoch_loss_d,
                                                                                                epoch_loss_dT,img2text_adv,
                                                                                                txt2img_adv))

            # deep copy the gen model

            if phase == 'test':


                if img2text_adv+txt2img_adv > best_acc_adv_image+best_acc_adv_text:
                    best_acc_adv_image = img2text_adv
                    best_model_wts_adv = copy.deepcopy(model_gen.state_dict())

                    best_acc_adv_text = txt2img_adv
                    best_model_wts_advT = copy.deepcopy(model_genT.state_dict())



    best_acc_adv = (best_acc_adv_image + best_acc_adv_text) / 2.0
    print('Adv - Best average ACC: {:4f}'.format(best_acc_adv))

    # load best model weights
    model_gen.load_state_dict(best_model_wts_adv)
    model_genT.load_state_dict(best_model_wts_advT)

    torch.save(model_gen.state_dict(), "save/pre-model_gen.pt")
    torch.save(model_genT.state_dict(), "save/pre-model_genT.pt")

    return model_genT, model_gen

def train_model(model_genT, model_disT, model_gen, model_dis, model, data_loaders, optimizer_genT, optimizer_disT,
                optimizer_gen, optimizer_dis, optimizer, num_epochs=100,num_epochsGAN=100):

    ##################################

    model = train_gse(model, data_loaders, optimizer,num_epochs)

    ##################################

    model_genT, model_gen = train_gan(model_genT,model_disT,model_gen,model_dis,model,data_loaders,optimizer_genT,optimizer_disT,
              optimizer_gen,optimizer_dis,num_epochsGAN)


    return model_genT, model_gen, model
