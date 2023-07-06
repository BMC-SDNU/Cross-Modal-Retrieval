import os
import torch
from torch import nn
from torch import autograd
import numpy as np
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset import Dataset
from config import opt
from models import *
from torch.optim import Adamax
from utils import calc_map_k, pr_curve, p_topK, Visualizer
from datasets.data_handler import load_data, load_pretrain_model
import time
import pickle


def train(**kwargs):
    opt.parse(kwargs)

    if opt.vis_env:
        vis = Visualizer(opt.vis_env, port=opt.vis_port)

    if opt.device is None or opt.device is 'cpu':
        opt.device = torch.device('cpu')
    else:
        opt.device = torch.device(opt.device)

    images, tags, labels = load_data(opt.data_path, type=opt.dataset)

    train_data = Dataset(opt, images, tags, labels)
    train_dataloader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True)

    # valid or test data
    x_query_data = Dataset(opt, images, tags, labels, test='image.query')
    x_db_data = Dataset(opt, images, tags, labels, test='image.db')
    y_query_data = Dataset(opt, images, tags, labels, test='text.query')
    y_db_data = Dataset(opt, images, tags, labels, test='text.db')

    x_query_dataloader = DataLoader(x_query_data, opt.batch_size, shuffle=False)
    x_db_dataloader = DataLoader(x_db_data, opt.batch_size, shuffle=False)
    y_query_dataloader = DataLoader(y_query_data, opt.batch_size, shuffle=False)
    y_db_dataloader = DataLoader(y_db_data, opt.batch_size, shuffle=False)

    query_labels, db_labels = x_query_data.get_labels()
    query_labels = query_labels.to(opt.device)
    db_labels = db_labels.to(opt.device)

    if opt.load_model_path:
        pretrain_model = None
    elif opt.pretrain_model_path:
        pretrain_model = load_pretrain_model(opt.pretrain_model_path)

    model = AGAH(opt.bit, opt.tag_dim, opt.num_label, opt.emb_dim,
                lambd=opt.lambd, pretrain_model=pretrain_model).to(opt.device)

    load_model(model, opt.load_model_path)

    optimizer = Adamax([
        {'params': model.img_module.parameters(), 'lr': opt.lr},
        {'params': model.txt_module.parameters()},
        {'params': model.hash_module.parameters()},
        {'params': model.classifier.parameters()}
    ], lr=opt.lr * 10, weight_decay=0.0005)

    optimizer_dis = {
        'img': Adamax(model.img_discriminator.parameters(), lr=opt.lr * 10, betas=(0.5, 0.9), weight_decay=0.0001),
        'txt': Adamax(model.txt_discriminator.parameters(), lr=opt.lr * 10, betas=(0.5, 0.9), weight_decay=0.0001)
    }

    criterion_tri_cos = TripletAllLoss(dis_metric='cos', reduction='sum')
    criterion_bce = nn.BCELoss(reduction='sum')

    loss = []

    max_mapi2t = 0.
    max_mapt2i = 0.

    FEATURE_I = torch.randn(opt.training_size, opt.emb_dim).to(opt.device)
    FEATURE_T = torch.randn(opt.training_size, opt.emb_dim).to(opt.device)

    U = torch.randn(opt.training_size, opt.bit).to(opt.device)
    V = torch.randn(opt.training_size, opt.bit).to(opt.device)

    FEATURE_MAP = torch.randn(opt.num_label, opt.emb_dim).to(opt.device)
    CODE_MAP = torch.sign(torch.randn(opt.num_label, opt.bit)).to(opt.device)

    train_labels = train_data.get_labels().to(opt.device)

    mapt2i_list = []
    mapi2t_list = []
    train_times = []

    for epoch in range(opt.max_epoch):
        t1 = time.time()
        for i, (ind, x, y, l) in tqdm(enumerate(train_dataloader)):
            imgs = x.to(opt.device)
            tags = y.to(opt.device)
            labels = l.to(opt.device)

            batch_size = len(ind)

            h_x, h_y, f_x, f_y, x_class, y_class = model(imgs, tags, FEATURE_MAP)

            FEATURE_I[ind] = f_x.data
            FEATURE_T[ind] = f_y.data
            U[ind] = h_x.data
            V[ind] = h_y.data

            #####
            # train txt discriminator
            #####
            D_txt_real = model.dis_txt(f_y.detach())
            D_txt_real = -D_txt_real.mean()
            optimizer_dis['txt'].zero_grad()
            D_txt_real.backward()

            # train with fake
            D_txt_fake = model.dis_txt(f_x.detach())
            D_txt_fake = D_txt_fake.mean()
            D_txt_fake.backward()

            # train with gradient penalty
            alpha = torch.rand(batch_size, opt.emb_dim).to(opt.device)
            interpolates = alpha * f_y.detach() + (1 - alpha) * f_x.detach()
            interpolates.requires_grad_()
            disc_interpolates = model.dis_txt(interpolates)
            gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                      grad_outputs=torch.ones(disc_interpolates.size()).to(opt.device),
                                      create_graph=True, retain_graph=True, only_inputs=True)[0]
            gradients = gradients.view(gradients.size(0), -1)
            # 10 is gradient penalty hyperparameter
            txt_gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
            txt_gradient_penalty.backward()

            loss_D_txt = D_txt_real - D_txt_fake
            optimizer_dis['txt'].step()

            #####
            # train img discriminator
            #####
            D_img_real = model.dis_img(f_x.detach())
            D_img_real = -D_img_real.mean()
            optimizer_dis['img'].zero_grad()
            D_img_real.backward()

            # train with fake
            D_img_fake = model.dis_img(f_y.detach())
            D_img_fake = D_img_fake.mean()
            D_img_fake.backward()

            # train with gradient penalty
            alpha = torch.rand(batch_size, opt.emb_dim).to(opt.device)
            interpolates = alpha * f_x.detach() + (1 - alpha) * f_y.detach()
            interpolates.requires_grad_()
            disc_interpolates = model.dis_img(interpolates)
            gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                      grad_outputs=torch.ones(disc_interpolates.size()).to(opt.device),
                                      create_graph=True, retain_graph=True, only_inputs=True)[0]
            gradients = gradients.view(gradients.size(0), -1)
            # 10 is gradient penalty hyperparameter
            img_gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
            img_gradient_penalty.backward()

            loss_D_img = D_img_real - D_img_fake
            optimizer_dis['img'].step()

            #####
            # train generators
            #####
            # update img network (to generate txt features)
            domain_output = model.dis_txt(f_x)
            loss_G_txt = -domain_output.mean()

            # update txt network (to generate img features)
            domain_output = model.dis_img(f_y)
            loss_G_img = -domain_output.mean()

            loss_adver = loss_G_txt + loss_G_img

            loss1 = criterion_tri_cos(h_x, labels, target=h_y, margin=opt.margin)
            loss2 = criterion_tri_cos(h_y, labels, target=h_x, margin=opt.margin)

            theta1 = F.cosine_similarity(torch.abs(h_x), torch.ones_like(h_x).to(opt.device))
            theta2 = F.cosine_similarity(torch.abs(h_y), torch.ones_like(h_y).to(opt.device))
            loss3 = torch.sum(1 / (1 + torch.exp(theta1))) + torch.sum(1 / (1 + torch.exp(theta2)))

            loss_class = criterion_bce(x_class, labels) + criterion_bce(y_class, labels)

            theta_code_x = h_x.mm(CODE_MAP.t())  # size: (batch, num_label)
            theta_code_y = h_y.mm(CODE_MAP.t())
            loss_code_map = torch.sum(torch.pow(theta_code_x - opt.bit * (labels * 2 - 1), 2)) + \
                            torch.sum(torch.pow(theta_code_y - opt.bit * (labels * 2 - 1), 2))

            loss_quant = torch.sum(torch.pow(h_x - torch.sign(h_x), 2)) + torch.sum(torch.pow(h_y - torch.sign(h_y), 2))

            # err = loss1 + loss2 + loss3 + 0.5 * loss_class + 0.5 * (loss_f1 + loss_f2)
            err = loss1 + loss2 + opt.alpha * loss3 + opt.beta * loss_class + opt.gamma * loss_code_map + \
                  opt.eta * loss_quant + opt.mu * loss_adver

            optimizer.zero_grad()
            err.backward()
            optimizer.step()

            loss.append(err.item())

        CODE_MAP = update_code_map(U, V, CODE_MAP, train_labels)
        FEATURE_MAP = update_feature_map(FEATURE_I, FEATURE_T, train_labels)

        print('...epoch: %3d, loss: %3.3f' % (epoch + 1, loss[-1]))
        delta_t = time.time() - t1

        if opt.vis_env:
            vis.plot('loss', loss[-1])

        # validate
        if opt.valid and (epoch + 1) % opt.valid_freq == 0:
            mapi2t, mapt2i = valid(model, x_query_dataloader, x_db_dataloader, y_query_dataloader, y_db_dataloader,
                                   query_labels, db_labels, FEATURE_MAP)
            print('...epoch: %3d, valid MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (epoch + 1, mapi2t, mapt2i))

            mapi2t_list.append(mapi2t)
            mapt2i_list.append(mapt2i)
            train_times.append(delta_t)

            if opt.vis_env:
                d = {
                    'mapi2t': mapi2t,
                    'mapt2i': mapt2i
                }
                vis.plot_many(d)

            if mapt2i >= max_mapt2i and mapi2t >= max_mapi2t:
                max_mapi2t = mapi2t
                max_mapt2i = mapt2i
                save_model(model)
                path = 'checkpoints/' + opt.dataset + '_' + str(opt.bit)
                with torch.cuda.device(opt.device):
                    torch.save(FEATURE_MAP, os.path.join(path, 'feature_map.pth'))

        if epoch % 100 == 0:
            for params in optimizer.param_groups:
                params['lr'] = max(params['lr'] * 0.6, 1e-6)

    if not opt.valid:
        save_model(model)

    print('...training procedure finish')
    if opt.valid:
        print('   max MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (max_mapi2t, max_mapt2i))
    else:
        mapi2t, mapt2i = valid(model, x_query_dataloader, x_db_dataloader, y_query_dataloader, y_db_dataloader,
                               query_labels, db_labels, FEATURE_MAP)
        print('   max MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (mapi2t, mapt2i))

    path = 'checkpoints/' + opt.dataset + '_' + str(opt.bit)
    with open(os.path.join(path, 'result.pkl'), 'wb') as f:
        pickle.dump([train_times, mapi2t_list, mapt2i_list], f)


def update_code_map(U, V, M, L):
    CODE_MAP = M
    U = torch.sign(U)
    V = torch.sign(V)
    S = torch.eye(opt.num_label).to(opt.device) * 2 - 1

    Q = 2 * opt.bit * (L.t().mm(U + V) + S.mm(M))

    for k in range(opt.bit):
        ind = np.setdiff1d(np.arange(0, opt.bit), k)
        term1 = CODE_MAP[:, ind].mm(U[:, ind].t()).mm(U[:, k].unsqueeze(-1)).squeeze()
        term2 = CODE_MAP[:, ind].mm(V[:, ind].t()).mm(V[:, k].unsqueeze(-1)).squeeze()
        term3 = CODE_MAP[:, ind].mm(M[:, ind].t()).mm(M[:, k].unsqueeze(-1)).squeeze()
        CODE_MAP[:, k] = torch.sign(Q[:, k] - 2 * (term1 + term2 + term3))

    return CODE_MAP


def update_feature_map(FEAT_I, FEAT_T, L, mode='average'):
    if mode is 'average':
        feature_map_I = L.t().mm(FEAT_I) / L.sum(dim=0).unsqueeze(-1)
        feature_map_T = L.t().mm(FEAT_T) / L.sum(dim=0).unsqueeze(-1)
    else:
        assert mode is 'max'
        feature_map_I = (L.t().unsqueeze(-1) * FEAT_I).max(dim=1)[0]
        feature_map_T = (L.t().unsqueeze(-1) * FEAT_T).max(dim=1)[0]

    FEATURE_MAP = (feature_map_T + feature_map_I) / 2
    # normalization
    FEATURE_MAP = FEATURE_MAP / torch.sqrt(torch.sum(FEATURE_MAP ** 2, dim=-1, keepdim=True))
    return FEATURE_MAP


def valid(model, x_query_dataloader, x_db_dataloader, y_query_dataloader, y_db_dataloader,
          query_labels, db_labels, FEATURE_MAP):
    model.eval()
    qBX = generate_img_code(model, x_query_dataloader, opt.query_size, FEATURE_MAP)
    qBY = generate_txt_code(model, y_query_dataloader, opt.query_size, FEATURE_MAP)
    rBX = generate_img_code(model, x_db_dataloader, opt.db_size, FEATURE_MAP)
    rBY = generate_txt_code(model, y_db_dataloader, opt.db_size, FEATURE_MAP)

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

    model = AGAH(opt.bit, opt.tag_dim, opt.num_label, opt.emb_dim,
                lambd=opt.lambd, pretrain_model=pretrain_model).to(opt.device)

    path = 'checkpoints/' + opt.dataset + '_' + str(opt.bit)
    load_model(model, path)
    FEATURE_MAP = torch.load(os.path.join(path, 'feature_map.pth')).to(opt.device)

    model.eval()

    images, tags, labels = load_data(opt.data_path, opt.dataset)

    x_query_data = Dataset(opt, images, tags, labels, test='image.query')
    x_db_data = Dataset(opt, images, tags, labels, test='image.db')
    y_query_data = Dataset(opt, images, tags, labels, test='text.query')
    y_db_data = Dataset(opt, images, tags, labels, test='text.db')

    x_query_dataloader = DataLoader(x_query_data, opt.batch_size, shuffle=False)
    x_db_dataloader = DataLoader(x_db_data, opt.batch_size, shuffle=False)
    y_query_dataloader = DataLoader(y_query_data, opt.batch_size, shuffle=False)
    y_db_dataloader = DataLoader(y_db_data, opt.batch_size, shuffle=False)

    qBX = generate_img_code(model, x_query_dataloader, opt.query_size, FEATURE_MAP)
    qBY = generate_txt_code(model, y_query_dataloader, opt.query_size, FEATURE_MAP)
    rBX = generate_img_code(model, x_db_dataloader, opt.db_size, FEATURE_MAP)
    rBY = generate_txt_code(model, y_db_dataloader, opt.db_size, FEATURE_MAP)

    query_labels, db_labels = x_query_data.get_labels()
    query_labels = query_labels.to(opt.device)
    db_labels = db_labels.to(opt.device)

    p_i2t, r_i2t = pr_curve(qBX, rBY, query_labels, db_labels)
    p_t2i, r_t2i = pr_curve(qBY, rBX, query_labels, db_labels)

    K = [1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    pk_i2t = p_topK(qBX, rBY, query_labels, db_labels, K)
    pk_t2i = p_topK(qBY, rBX, query_labels, db_labels, K)

    path = 'checkpoints/' + opt.dataset + '_' + str(opt.bit)
    np.save(os.path.join(path, 'P_i2t.npy'), p_i2t.numpy())
    np.save(os.path.join(path, 'R_i2t.npy'), r_i2t.numpy())
    np.save(os.path.join(path, 'P_t2i.npy'), p_t2i.numpy())
    np.save(os.path.join(path, 'R_t2i.npy'), r_t2i.numpy())
    np.save(os.path.join(path, 'P_at_K_i2t.npy'), pk_i2t.numpy())
    np.save(os.path.join(path, 'P_at_K_t2i.npy'), pk_t2i.numpy())

    mapi2t = calc_map_k(qBX, rBY, query_labels, db_labels)
    mapt2i = calc_map_k(qBY, rBX, query_labels, db_labels)
    print('...test MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (mapi2t, mapt2i))


def generate_img_code(model, test_dataloader, num, FEATURE_MAP):
    B = torch.zeros(num, opt.bit).to(opt.device)

    for i, input_data in tqdm(enumerate(test_dataloader)):
        input_data = input_data.to(opt.device)
        b = model.generate_img_code(input_data, FEATURE_MAP)
        idx_end = min(num, (i + 1) * opt.batch_size)
        B[i * opt.batch_size: idx_end, :] = b.data

    B = torch.sign(B)
    return B


def generate_txt_code(model, test_dataloader, num, FEATURE_MAP):
    B = torch.zeros(num, opt.bit).to(opt.device)

    for i, input_data in tqdm(enumerate(test_dataloader)):
        input_data = input_data.to(opt.device)
        b = model.generate_txt_code(input_data, FEATURE_MAP)
        idx_end = min(num, (i + 1) * opt.batch_size)
        B[i * opt.batch_size: idx_end, :] = b.data

    B = torch.sign(B)
    return B


def calc_loss(loss):
    l = 0.
    for v in loss.values():
        l += v[-1]
    return l


def avoid_inf(x):
    return torch.log(1.0 + torch.exp(-torch.abs(x))) + torch.max(torch.zeros_like(x), x)


def load_model(model, path):
    if path is not None:
        model.load(os.path.join(path, model.module_name + '.pth'))


def save_model(model):
    path = 'checkpoints/' + opt.dataset + '_' + str(opt.bit)
    model.save(model.module_name + '.pth', path, cuda_device=opt.device)


def help():
    """
    打印帮助的信息： python file.py help
    """
    print('''========================::HELP::=========================
    usage : python file.py <function> [--args=value]
    <function> := train | test | help
    example:
            python {0} train --lr=0.01
            python {0} help
    avaiable args (default value):'''.format(__file__))
    for k, v in opt.__class__.__dict__.items():
        if not k.startswith('__') and str(k) != 'parse':
            print('            {0}: {1}'.format(k, v))
    print('========================::HELP::=========================')


if __name__ == '__main__':
    import fire
    fire.Fire()
