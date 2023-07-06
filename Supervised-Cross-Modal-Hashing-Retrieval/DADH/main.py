import os
from torch import autograd
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset import Dataset
from config import opt
from models.dis_model import DIS
from models.gen_model import GEN
from triplet_loss import *
from torch.optim import Adam
from utils import calc_map_k, Visualizer
from datasets.data_handler import load_data, load_pretrain_model
import time
import pickle


def train(**kwargs):
    opt.parse(kwargs)
    opt.beta = opt.beta + 0.1

    if opt.vis_env:
        vis = Visualizer(opt.vis_env, port=opt.vis_port)

    if opt.device is None or opt.device is 'cpu':
        opt.device = torch.device('cpu')
    else:
        opt.device = torch.device(opt.device)

    images, tags, labels = load_data(opt.data_path, type=opt.dataset)
    train_data = Dataset(opt, images, tags, labels)
    train_dataloader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True)
    L = train_data.get_labels()
    L = L.to(opt.device)
    # test
    i_query_data = Dataset(opt, images, tags, labels, test='image.query')
    i_db_data = Dataset(opt, images, tags, labels, test='image.db')
    t_query_data = Dataset(opt, images, tags, labels, test='text.query')
    t_db_data = Dataset(opt, images, tags, labels, test='text.db')

    i_query_dataloader = DataLoader(i_query_data, opt.batch_size, shuffle=False)
    i_db_dataloader = DataLoader(i_db_data, opt.batch_size, shuffle=False)
    t_query_dataloader = DataLoader(t_query_data, opt.batch_size, shuffle=False)
    t_db_dataloader = DataLoader(t_db_data, opt.batch_size, shuffle=False)

    query_labels, db_labels = i_query_data.get_labels()
    query_labels = query_labels.to(opt.device)
    db_labels = db_labels.to(opt.device)

    pretrain_model = load_pretrain_model(opt.pretrain_model_path)

    generator = GEN(opt.dropout, opt.image_dim, opt.text_dim, opt.hidden_dim, opt.bit, pretrain_model=pretrain_model).to(opt.device)

    discriminator = DIS(opt.hidden_dim//4, opt.hidden_dim//8, opt.bit).to(opt.device)

    optimizer = Adam([
        # {'params': generator.cnn_f.parameters()},     ## froze parameters of cnn_f
        {'params': generator.image_module.parameters()},
        {'params': generator.text_module.parameters()},
        {'params': generator.hash_module.parameters()}
    ], lr=opt.lr, weight_decay=0.0005)

    optimizer_dis = {
        'feature': Adam(discriminator.feature_dis.parameters(), lr=opt.lr, betas=(0.5, 0.9), weight_decay=0.0001),
        'hash': Adam(discriminator.hash_dis.parameters(), lr=opt.lr, betas=(0.5, 0.9), weight_decay=0.0001)
    }

    tri_loss = TripletLoss(opt, reduction='sum')

    loss = []

    max_mapi2t = 0.
    max_mapt2i = 0.
    max_average = 0.

    mapt2i_list = []
    mapi2t_list = []
    train_times = []

    B_i = torch.randn(opt.training_size, opt.bit).sign().to(opt.device)
    B_t = B_i
    H_i = torch.zeros(opt.training_size, opt.bit).to(opt.device)
    H_t = torch.zeros(opt.training_size, opt.bit).to(opt.device)

    for epoch in range(opt.max_epoch):
        t1 = time.time()
        e_loss = 0
        for i, (ind, img, txt, label) in tqdm(enumerate(train_dataloader)):
            imgs = img.to(opt.device)
            txt = txt.to(opt.device)
            labels = label.to(opt.device)

            batch_size = len(ind)

            h_i, h_t, f_i, f_t = generator(imgs, txt)
            H_i[ind, :] = h_i.data
            H_t[ind, :] = h_t.data
            h_t_detach = generator.generate_txt_code(txt)

            #####
            # train feature discriminator
            #####
            D_real_feature = discriminator.dis_feature(f_i.detach())
            D_real_feature = -opt.gamma * torch.log(torch.sigmoid(D_real_feature)).mean()
            # D_real_feature = -D_real_feature.mean()
            optimizer_dis['feature'].zero_grad()
            D_real_feature.backward()

            # train with fake
            D_fake_feature = discriminator.dis_feature(f_t.detach())
            D_fake_feature = -opt.gamma * torch.log(torch.ones(batch_size).to(opt.device) - torch.sigmoid(D_fake_feature)).mean()
            # D_fake_feature = D_fake_feature.mean()
            D_fake_feature.backward()

            # train with gradient penalty
            alpha = torch.rand(batch_size, opt.hidden_dim//4).to(opt.device)
            interpolates = alpha * f_i.detach() + (1 - alpha) * f_t.detach()
            interpolates.requires_grad_()
            disc_interpolates = discriminator.dis_feature(interpolates)
            gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                      grad_outputs=torch.ones(disc_interpolates.size()).to(opt.device),
                                      create_graph=True, retain_graph=True, only_inputs=True)[0]
            gradients = gradients.view(gradients.size(0), -1)
            # 10 is gradient penalty hyperparameter
            feature_gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
            feature_gradient_penalty.backward()

            optimizer_dis['feature'].step()

            #####
            # train hash discriminator
            #####
            D_real_hash = discriminator.dis_hash(h_i.detach())
            D_real_hash = -opt.gamma * torch.log(torch.sigmoid(D_real_hash)).mean()
            optimizer_dis['hash'].zero_grad()
            D_real_hash.backward()

            # train with fake
            D_fake_hash = discriminator.dis_hash(h_t.detach())
            D_fake_hash = -opt.gamma * torch.log(torch.ones(batch_size).to(opt.device) - torch.sigmoid(D_fake_hash)).mean()
            D_fake_hash.backward()

            # train with gradient penalty
            alpha = torch.rand(batch_size, opt.bit).to(opt.device)
            interpolates = alpha * h_i.detach() + (1 - alpha) * h_t.detach()
            interpolates.requires_grad_()
            disc_interpolates = discriminator.dis_hash(interpolates)
            gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                      grad_outputs=torch.ones(disc_interpolates.size()).to(opt.device),
                                      create_graph=True, retain_graph=True, only_inputs=True)[0]
            gradients = gradients.view(gradients.size(0), -1)

            hash_gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
            hash_gradient_penalty.backward()

            optimizer_dis['hash'].step()

            loss_G_txt_feature = -torch.log(torch.sigmoid(discriminator.dis_feature(f_t))).mean()
            loss_adver_feature = loss_G_txt_feature

            loss_G_txt_hash = -torch.log(torch.sigmoid(discriminator.dis_hash(h_t_detach))).mean()
            loss_adver_hash = loss_G_txt_hash

            tri_i2t = tri_loss(h_i, labels, target=h_t, margin=opt.margin)
            tri_t2i = tri_loss(h_t, labels, target=h_i, margin=opt.margin)
            weighted_cos_tri = tri_i2t + tri_t2i

            i_ql = torch.sum(torch.pow(B_i[ind, :] - h_i, 2))
            t_ql = torch.sum(torch.pow(B_i[ind, :] - h_t, 2))
            loss_quant = i_ql + t_ql
            err = opt.alpha * weighted_cos_tri + \
                  opt.beta * loss_quant + opt.gamma * (loss_adver_feature + loss_adver_hash)

            optimizer.zero_grad()
            err.backward()
            optimizer.step()

            e_loss = err + e_loss

        P_i = torch.inverse(
                L.t() @ L + opt.lamb * torch.eye(opt.num_label, device=opt.device)) @ L.t() @ B_i
        # P_t = torch.inverse(
        #         L.t() @ L + opt.lamb * torch.eye(opt.num_label, device=opt.device)) @ L.t() @ B_t

        B_i = (L @ P_i + 0.5 * opt.mu * (H_i + H_t)).sign()
        # B_t = (L @ P_t + opt.mu * H_t).sign()
        loss.append(e_loss.item())
        print('...epoch: %3d, loss: %3.3f' % (epoch + 1, loss[-1]))
        delta_t = time.time() - t1

        if opt.vis_env:
            vis.plot('loss', loss[-1])

        # validate
        if opt.valid and (epoch + 1) % opt.valid_freq == 0:
            mapi2t, mapt2i = valid(generator, i_query_dataloader, i_db_dataloader, t_query_dataloader, t_db_dataloader,
                                   query_labels, db_labels)
            print('...epoch: %3d, valid MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (epoch + 1, mapi2t, mapt2i))

            mapi2t_list.append(mapi2t)
            mapt2i_list.append(mapt2i)
            train_times.append(delta_t)

            if 0.5 * (mapi2t + mapt2i) > max_average:
                max_mapi2t = mapi2t
                max_mapt2i = mapt2i
                max_average = 0.5 * (mapi2t + mapt2i)
                save_model(generator)

            if opt.vis_env:
                vis.plot('mapi2t', mapi2t)
                vis.plot('mapt2i', mapt2i)

        if epoch % 100 == 0:
            for params in optimizer.param_groups:
                params['lr'] = max(params['lr'] * 0.8, 1e-6)

    if not opt.valid:
        save_model(generator)

    print('...training procedure finish')
    if opt.valid:
        print('   max MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (max_mapi2t, max_mapt2i))
    else:
        mapi2t, mapt2i = valid(generator, i_query_dataloader, i_db_dataloader, t_query_dataloader, t_db_dataloader,
                               query_labels, db_labels)
        print('   max MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (mapi2t, mapt2i))

    path = 'checkpoints/' + opt.dataset + '_' + str(opt.bit)
    with open(os.path.join(path, 'result.pkl'), 'wb') as f:
        pickle.dump([train_times, mapi2t_list, mapt2i_list], f)


def valid(model, x_query_dataloader, x_db_dataloader, y_query_dataloader, y_db_dataloader,
          query_labels, db_labels):
    model.eval()

    qBX = generate_img_code(model, x_query_dataloader, opt.query_size)
    qBY = generate_txt_code(model, y_query_dataloader, opt.query_size)
    rBX = generate_img_code(model, x_db_dataloader, opt.db_size)
    rBY = generate_txt_code(model, y_db_dataloader, opt.db_size)

    mapi2t = calc_map_k(qBX, rBY, query_labels, db_labels)
    mapt2i = calc_map_k(qBY, rBX, query_labels, db_labels)

    model.train()
    return mapi2t.item(), mapt2i.item()


def test(**kwargs):
    opt.parse(kwargs)

    if opt.device is not None:
        opt.device = torch.device(opt.device)
    elif opt.gpus:
        opt.device = torch.device(0)
    else:
        opt.device = torch.device('cpu')

    pretrain_model = load_pretrain_model(opt.pretrain_model_path)

    generator = GEN(opt.dropout, opt.image_dim, opt.text_dim, opt.hidden_dim, opt.bit, pretrain_model=pretrain_model).to(opt.device)
    

    path = 'checkpoints/' + opt.dataset + '_' + str(opt.bit)
    load_model(generator, path)

    generator.eval()

    images, tags, labels = load_data(opt.data_path, opt.dataset)

    i_query_data = Dataset(opt, images, tags, labels, test='image.query')
    i_db_data = Dataset(opt, images, tags, labels, test='image.db')
    t_query_data = Dataset(opt, images, tags, labels, test='text.query')
    t_db_data = Dataset(opt, images, tags, labels, test='text.db')

    i_query_dataloader = DataLoader(i_query_data, opt.batch_size, shuffle=False)
    i_db_dataloader = DataLoader(i_db_data, opt.batch_size, shuffle=False)
    t_query_dataloader = DataLoader(t_query_data, opt.batch_size, shuffle=False)
    t_db_dataloader = DataLoader(t_db_data, opt.batch_size, shuffle=False)

    qBX = generate_img_code(generator, i_query_dataloader, opt.query_size)
    qBY = generate_txt_code(generator, t_query_dataloader, opt.query_size)
    rBX = generate_img_code(generator, i_db_dataloader, opt.db_size)
    rBY = generate_txt_code(generator, t_db_dataloader, opt.db_size)

    query_labels, db_labels = i_query_data.get_labels()
    query_labels = query_labels.to(opt.device)
    db_labels = db_labels.to(opt.device)

    mapi2t = calc_map_k(qBX, rBY, query_labels, db_labels)
    mapt2i = calc_map_k(qBY, rBX, query_labels, db_labels)
    print('...test MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (mapi2t, mapt2i))


def generate_img_code(model, test_dataloader, num):
    B = torch.zeros(num, opt.bit).to(opt.device)

    for i, input_data in tqdm(enumerate(test_dataloader)):
        input_data = input_data.to(opt.device)
        b = model.generate_img_code(input_data)
        idx_end = min(num, (i + 1) * opt.batch_size)
        B[i * opt.batch_size: idx_end, :] = b.data

    B = torch.sign(B)
    return B


def generate_txt_code(model, test_dataloader, num):
    B = torch.zeros(num, opt.bit).to(opt.device)

    for i, input_data in tqdm(enumerate(test_dataloader)):
        input_data = input_data.to(opt.device)
        b = model.generate_txt_code(input_data)
        idx_end = min(num, (i + 1) * opt.batch_size)
        B[i * opt.batch_size: idx_end, :] = b.data

    B = torch.sign(B)
    return B


def load_model(model, path):
    if path is not None:
        model.load(os.path.join(path, model.module_name + '.pth'))


def save_model(model):
    path = 'checkpoints/' + opt.dataset + '_' + str(opt.bit)
    model.save(model.module_name + '.pth', path, cuda_device=opt.device)


if __name__ == '__main__':
    import fire
    fire.Fire()


