import numpy as np
import os
import torch

# import numpy as np
# import os
# # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# import torch
from utils.config import args
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
cudnn.benchmark = True
import nets as models
from utils.bar_show import progress_bar
from src.noisydataset import cross_modal_dataset
import src.utils as utils
import scipy
import scipy.spatial


best_acc = 0  # best test accuracy
start_epoch = 0

args.log_dir = os.path.join(args.root_dir, 'logs', args.log_name)
args.ckpt_dir = os.path.join(args.root_dir, 'ckpt', args.ckpt_dir)

os.makedirs(args.log_dir, exist_ok=True)
os.makedirs(args.ckpt_dir, exist_ok=True)

def load_dict(model, path):
    chp = torch.load(path)
    state_dict = model.state_dict()
    for key in state_dict:
        if key in chp['model_state_dict']:
            state_dict[key] = chp['model_state_dict'][key]
    model.load_state_dict(state_dict)

def main():
    print('===> Preparing data ..')
    train_dataset = cross_modal_dataset(args.data_name, args.noisy_ratio, 'train')
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        # sampler=sampler,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=False
    )

    valid_dataset = cross_modal_dataset(args.data_name, args.noisy_ratio, 'valid')
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
        drop_last=False
    )

    test_dataset = cross_modal_dataset(args.data_name, args.noisy_ratio, 'test')
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
        drop_last=False
    )

    print('===> Building Models..')
    multi_models = []
    n_view = len(train_dataset.train_data)
    for v in range(n_view):
        if v == args.views.index('Img'): # Images
            multi_models.append(models.__dict__['ImageNet'](input_dim=train_dataset.train_data[v].shape[1], output_dim=args.output_dim).cuda())
        elif v == args.views.index('Txt'): # Text
            multi_models.append(models.__dict__['TextNet'](input_dim=train_dataset.train_data[v].shape[1], output_dim=args.output_dim).cuda())
        else: # Default to use ImageNet
            multi_models.append(models.__dict__['ImageNet'](input_dim=train_dataset.train_data[v].shape[1], output_dim=args.output_dim).cuda())

    C = torch.Tensor(args.output_dim, args.output_dim)
    C = torch.nn.init.orthogonal(C, gain=1)[:, 0: train_dataset.class_num].cuda()
    C.requires_grad = True

    embedding = torch.eye(train_dataset.class_num).cuda()
    embedding.requires_grad = False

    parameters = [C]
    for v in range(n_view):
        parameters += list(multi_models[v].parameters())
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(parameters, lr=args.lr, momentum=0.9, weight_decay=args.wd)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(parameters, lr=args.lr, betas=[0.5, 0.999], weight_decay=args.wd)
    if args.ls == 'cos':
        lr_schedu = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_epochs, eta_min=0, last_epoch=-1)
    else:
        lr_schedu = optim.lr_scheduler.MultiStepLR(optimizer, [200, 400], gamma=0.1)

    if args.loss == 'CE':
        criterion = torch.nn.CrossEntropyLoss().cuda()
    elif args.loss == 'MCE':
        criterion = utils.MeanClusteringError(train_dataset.class_num, tau=args.tau).cuda()
    else:
        raise Exception('No such loss function.')

    summary_writer = SummaryWriter(args.log_dir)

    if args.resume:
        ckpt = torch.load(os.path.join(args.ckpt_dir, args.resume))
        for v in range(n_view):
            multi_models[v].load_state_dict(ckpt['model_state_dict_%d' % v])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch']
        print('===> Load last checkpoint data')
    else:
        start_epoch = 0
        print('===> Start from scratch')

    def set_train():
        for v in range(n_view):
            multi_models[v].train()

    def set_eval():
        for v in range(n_view):
            multi_models[v].eval()

    def cross_modal_contrastive_ctriterion(fea, tau=1.):
        batch_size = fea[0].shape[0]
        all_fea = torch.cat(fea)
        sim = all_fea.mm(all_fea.t())

        sim = (sim / tau).exp()
        sim = sim - sim.diag().diag()
        sim_sum1 = sum([sim[:, v * batch_size: (v + 1) * batch_size] for v in range(n_view)])
        diag1 = torch.cat([sim_sum1[v * batch_size: (v + 1) * batch_size].diag() for v in range(n_view)])
        loss1 = -(diag1 / sim.sum(1)).log().mean()

        sim_sum2 = sum([sim[v * batch_size: (v + 1) * batch_size] for v in range(n_view)])
        diag2 = torch.cat([sim_sum2[:, v * batch_size: (v + 1) * batch_size].diag() for v in range(n_view)])
        loss2 = -(diag2 / sim.sum(1)).log().mean()
        return loss1 + loss2

    def train(epoch):
        print('\nEpoch: %d / %d' % (epoch, args.max_epochs))
        set_train()
        train_loss, loss_list, correct_list, total_list = 0., [0.] * n_view, [0.] * n_view, [0.] * n_view

        for batch_idx, (batches, targets, index) in enumerate(train_loader):
            batches, targets = [batches[v].cuda() for v in range(n_view)], [targets[v].cuda() for v in range(n_view)]
            norm = C.norm(dim=0, keepdim=True)
            C.data = (C / norm).detach()

            for v in range(n_view):
                multi_models[v].zero_grad()
            optimizer.zero_grad()

            outputs = [multi_models[v](batches[v]) for v in range(n_view)]
            preds = [outputs[v].mm(C) for v in range(n_view)]
            losses = [criterion(preds[v], targets[v]) for v in range(n_view)]
            loss = sum(losses)
            loss = args.beta * loss + (1. - args.beta) * cross_modal_contrastive_ctriterion(outputs, tau=args.tau)
            if epoch >= 0:
                loss.backward()
                optimizer.step()
            train_loss += loss.item()

            for v in range(n_view):
                loss_list[v] += losses[v]
                _, predicted = preds[v].max(1)
                total_list[v] += targets[v].size(0)
                acc = predicted.eq(targets[v]).sum().item()
                correct_list[v] += acc
            progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | LR: %g'
                         % (train_loss / (batch_idx + 1), optimizer.param_groups[0]['lr']))

        train_dict = {('view_%d_loss' % v): loss_list[v] / len(train_loader) for v in range(n_view)}
        train_dict['sum_loss'] = train_loss / len(train_loader)
        summary_writer.add_scalars('Loss/train', train_dict, epoch)
        summary_writer.add_scalars('Accuracy/train', {'view_%d_acc': correct_list[v] / total_list[v] for v in range(n_view)}, epoch)

    def eval(data_loader, epoch, mode='test'):
        fea, lab = [[] for _ in range(n_view)], [[] for _ in range(n_view)]
        test_loss, loss_list, correct_list, total_list = 0., [0.] * n_view, [0.] * n_view, [0.] * n_view
        with torch.no_grad():
            if sum([data_loader.dataset.train_data[v].shape[0] != data_loader.dataset.train_data[0].shape[0] for v in range(len(data_loader.dataset.train_data))]) == 0:
                for batch_idx, (batches, targets, index) in enumerate(data_loader):
                    batches, targets = [batches[v].cuda() for v in range(n_view)], [targets[v].cuda() for v in range(n_view)]
                    outputs = [multi_models[v](batches[v]) for v in range(n_view)]
                    pred, losses = [], []
                    for v in range(n_view):
                        fea[v].append(outputs[v])
                        lab[v].append(targets[v])
                        pred.append(outputs[v].mm(C))
                        losses.append(criterion(pred[v], targets[v]))
                        loss_list[v] += losses[v]
                        _, predicted = pred[v].max(1)
                        total_list[v] += targets[v].size(0)
                        acc = predicted.eq(targets[v]).sum().item()
                        correct_list[v] += acc
                    loss = sum(losses)
                    test_loss += loss.item()
            else:
                pred, losses = [], []
                for v in range(n_view):
                    count = int(np.ceil(data_loader.dataset.train_data[v].shape[0]) / data_loader.batch_size)
                    for ct in range(count):
                        batch, targets = torch.Tensor(data_loader.dataset.train_data[v][ct * data_loader.batch_size: (ct + 1) * data_loader.batch_size]).cuda(), torch.Tensor(data_loader.dataset.noise_label[v][ct * data_loader.batch_size: (ct + 1) * data_loader.batch_size]).long().cuda()
                        outputs = multi_models[v](batch)

                        fea[v].append(outputs)
                        lab[v].append(targets)
                        pred.append(outputs.mm(C))
                        losses.append(criterion(pred[v], targets))
                        loss_list[v] += losses[v]
                        _, predicted = pred[v].max(1)
                        total_list[v] += targets.size(0)
                        acc = predicted.eq(targets).sum().item()
                        correct_list[v] += acc
                    loss = sum(losses)
                    test_loss += loss.item()

            fea = [torch.cat(fea[v]).cpu().detach().numpy() for v in range(n_view)]
            lab = [torch.cat(lab[v]).cpu().detach().numpy() for v in range(n_view)]
        test_dict = {('view_%d_loss' % v): loss_list[v] / len(data_loader) for v in range(n_view)}
        test_dict['sum_loss'] = test_loss / len(data_loader)
        summary_writer.add_scalars('Loss/' + mode, test_dict, epoch)

        summary_writer.add_scalars('Accuracy/' + mode, {('view_%d_acc' % v): correct_list[v] / total_list[v] for v in range(n_view)}, epoch)
        return fea, lab

    def multiview_test(fea, lab):
        MAPs = np.zeros([n_view, n_view])
        val_dict = {}
        print_str = ''
        for i in range(n_view):
            for j in range(n_view):
                if i == j:
                    continue
                MAPs[i, j] = fx_calc_map_label(fea[j], lab[j], fea[i], lab[i], k=0, metric='cosine')[0]
                key = '%s2%s' % (args.views[i], args.views[j])
                val_dict[key] = MAPs[i, j]
                print_str = print_str + key + ': %.3f\t' % val_dict[key]
        return val_dict, print_str

    def test(epoch):
            global best_acc
            set_eval()
            # switch to evaluate mode
            fea, lab = eval(train_loader, epoch, 'train')

            MAPs = np.zeros([n_view, n_view])
            train_dict = {}
            for i in range(n_view):
                for j in range(n_view):
                    MAPs[i, j] = fx_calc_map_label(fea[j], lab[j], fea[i], lab[i], k=0, metric='cosine')[0]
                    train_dict['%s2%s' % (args.views[i], args.views[j])] = MAPs[i, j]

            train_avg = MAPs.sum() / n_view / (n_view - 1.)
            train_dict['avg'] = train_avg
            summary_writer.add_scalars('Retrieval/train', train_dict, epoch)

            fea, lab = eval(valid_loader, epoch, 'valid')
            MAPs = np.zeros([n_view, n_view])
            val_dict = {}
            print_val_str = 'Validation: '

            for i in range(n_view):
                for j in range(n_view):
                    if i == j:
                        continue
                    MAPs[i, j] = fx_calc_map_label(fea[j], lab[j], fea[i], lab[i], k=0, metric='cosine')[0]
                    key = '%s2%s' % (args.views[i], args.views[j])
                    val_dict[key] = MAPs[i, j]
                    print_val_str = print_val_str + key +': %g\t' % val_dict[key]


            val_avg = MAPs.sum() / n_view / (n_view - 1.)
            val_dict['avg'] = val_avg
            print_val_str = print_val_str + 'Avg: %g' % val_avg
            summary_writer.add_scalars('Retrieval/valid', val_dict, epoch)

            fea, lab = eval(test_loader, epoch, 'test')
            MAPs = np.zeros([n_view, n_view])
            test_dict = {}
            print_test_str = 'Test: '
            for i in range(n_view):
                for j in range(n_view):
                    if i == j:
                        continue
                    MAPs[i, j] = fx_calc_map_label(fea[j], lab[j], fea[i], lab[i], k=0, metric='cosine')[0]
                    key = '%s2%s' % (args.views[i], args.views[j])
                    test_dict[key] = MAPs[i, j]
                    print_test_str = print_test_str + key + ': %g\t' % test_dict[key]

            test_avg = MAPs.sum() / n_view / (n_view - 1.)
            print_test_str = print_test_str + 'Avg: %g' % test_avg
            test_dict['avg'] = test_avg
            summary_writer.add_scalars('Retrieval/test', test_dict, epoch)

            print(print_val_str)
            if val_avg > best_acc:
                best_acc = val_avg
                print(print_test_str)
                print('Saving..')
                state = {}
                for v in range(n_view):
                    # models[v].load_state_dict(ckpt['model_state_dict_%d' % v])
                    state['model_state_dict_%d' % v] = multi_models[v].state_dict()
                for key in test_dict:
                    state[key] = test_dict[key]
                state['epoch'] = epoch
                state['optimizer_state_dict'] = optimizer.state_dict()
                state['C'] = C
                torch.save(state, os.path.join(args.ckpt_dir, '%s_%s_%d_best_checkpoint.t7' % ('MRL', args.data_name, args.output_dim)))
            return val_dict

    # test(1)
    best_prec1 = 0.
    lr_schedu.step(start_epoch)
    train(-1)
    results = test(-1)
    for epoch in range(start_epoch, args.max_epochs):
        train(epoch)
        lr_schedu.step(epoch)
        test_dict = test(epoch + 1)
        if test_dict['avg'] == best_acc:
            multi_model_state_dict = [{key: value.clone() for (key, value) in m.state_dict().items()} for m in multi_models]
            W_best = C.clone()

    print('Evaluation on Last Epoch:')
    fea, lab = eval(test_loader, epoch, 'test')
    test_dict, print_str = multiview_test(fea, lab)
    print(print_str)

    print('Evaluation on Best Validation:')
    [multi_models[v].load_state_dict(multi_model_state_dict[v]) for v in range(n_view)]
    fea, lab = eval(test_loader, epoch, 'test')
    test_dict, print_str = multiview_test(fea, lab)
    print(print_str)
    import scipy.io as sio
    save_dict = dict(**{args.views[v]: fea[v] for v in range(n_view)}, **{args.views[v] + '_lab': lab[v] for v in range(n_view)})
    save_dict['C'] = W_best.detach().cpu().numpy()
    sio.savemat('features/%s_%g.mat' % (args.data_name, args.noisy_ratio), save_dict)

def fx_calc_map_multilabel_k(train, train_labels, test, test_label, k=0, metric='cosine'):
    dist = scipy.spatial.distance.cdist(test, train, metric)
    ord = dist.argsort()
    numcases = dist.shape[0]
    if k == 0:
        k = numcases
    res = []
    for i in range(numcases):
        order = ord[i].reshape(-1)

        tmp_label = (np.dot(train_labels[order], test_label[i]) > 0)
        if tmp_label.sum() > 0:
            prec = tmp_label.cumsum() / np.arange(1.0, 1 + tmp_label.shape[0])
            total_pos = float(tmp_label.sum())
            if total_pos > 0:
                res += [np.dot(tmp_label, prec) / total_pos]
    return np.mean(res)

def fx_calc_map_label(train, train_labels, test, test_label, k=0, metric='cosine'):
    dist = scipy.spatial.distance.cdist(test, train, metric)

    ord = dist.argsort(1)

    numcases = train_labels.shape[0]
    if k == 0:
        k = numcases
    if k == -1:
        ks = [50, numcases]
    else:
        ks = [k]

    def calMAP(_k):
        _res = []
        for i in range(len(test_label)):
            order = ord[i]
            p = 0.0
            r = 0.0
            for j in range(_k):
                if test_label[i] == train_labels[order[j]]:
                    r += 1
                    p += (r / (j + 1))
            if r > 0:
                _res += [p / r]
            else:
                _res += [0]
        return np.mean(_res)

    res = []
    for k in ks:
        res.append(calMAP(k))
    return res

if __name__ == '__main__':
    main()

