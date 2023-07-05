import os
import gc
import glob
import random
import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from data_loader import foodSpaceLoader
from data_loader import error_catching_loader, my_collate
from args import get_parser
from models import FoodSpaceNet as FoodSpaceNet
from utils import PadToSquareResize, AverageMeter, SubsetSequentialSampler, cosine_distance, worker_init_fn, get_variable
from tqdm import tqdm
from shutil import copyfile
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


# =============================================================================
parser = get_parser()
opts = parser.parse_args()
opts.freeWordEmb = opts.freeWordEmb if opts.w2vTrain else False

# =============================================================================


def main():
    global epoch, opts
    random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed(opts.seed)
    torch.cuda.manual_seed_all(opts.seed)

    iter = 0

    # Track results
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    print('Saving to: ' + timestamp)
    logdir = "runs/" + timestamp + '__' + os.path.basename(__file__).replace('.py', '') + opts.experiment_sufix + "/"
    logdir = os.path.join(os.path.dirname(__file__), logdir)
    modeldir = logdir + "models/"
    opts.snapshots = modeldir
    if not os.path.isdir("runs/"): os.mkdir("runs/")
    if not os.path.isdir(logdir): os.mkdir(logdir)
    if not os.path.isdir(modeldir): os.mkdir(modeldir)
    copyfile('models.py', os.path.join(logdir,'models.py'))
    train_writer = SummaryWriter(logdir + '/train')
    train_writer_text = open(os.path.join(logdir, 'train.txt'), 'a')
    test_writer_text = open(os.path.join(logdir, 'test.txt'), 'a')

    with open(os.path.join(logdir, 'opts.txt'), 'w') as file:
        for arg in vars(opts):
            file.write(arg + ': ' + str(getattr(opts, arg)) + '\n')
    model = FoodSpaceNet(opts)

    if opts.model_init_path:
        print("=> initializing from '{}'".format(opts.model_init_path))
        checkpoint = torch.load(opts.model_init_path, map_location=lambda storage, loc: storage)
        if 'module.' in list(checkpoint['state_dict'].keys())[0]:
            tmp_dict = checkpoint['state_dict'].copy()
            for k, v in checkpoint['state_dict'].items():
                tmp_dict[k.replace('module.', '')] = tmp_dict.pop(k)
            checkpoint['state_dict'] = tmp_dict.copy()
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        print("=> initialized from '{}'".format(opts.model_init_path))
    if not opts.no_cuda:
        model.cuda()


    # data preparation, loaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # preparing the training loader
    train_data = foodSpaceLoader(transform=transforms.Compose([
                                  transforms.Resize(256),
                                  transforms.CenterCrop(256),
                                  transforms.RandomCrop(224),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  normalize,]),
                              partition='train',
                              loader=error_catching_loader,
                              opts=opts)
    print('Training loader prepared.')

    # preparing the valitadion loader
    transform=transforms.Compose([
                            PadToSquareResize(resize=256, padding_mode='reflect'),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            normalize])

    val_data = foodSpaceLoader(transform=transform,
                            partition='val',
                            loader=error_catching_loader,
                            opts=opts)
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=opts.batch_size, shuffle=False,
        num_workers=opts.workers, pin_memory=False, drop_last=False)
    print('Validation loader prepared.')

    # creating different parameter groups
    vision_params = [kv[1] for kv in model.named_parameters() if kv[0].split('.')[0] in ['visionMLP']]
    wordEmb_params = [kv[1] for kv in model.named_parameters() if 'textMLP.mBERT.embeddings' in kv[0] or 'word_embeddings' in kv[0]]
    text_params = [kv[1] for kv in model.named_parameters() if ('textMLP' in kv[0]) and ('embeddings' not in kv[0])]
    heads_params = [kv[1] for kv in model.named_parameters() if (kv[0].split('.')[0] not in ['visionMLP', 'textMLP'])]
    optimizer = torch.optim.Adam([
                {'params': heads_params}, # these correspond to the text and vision "heads" (layer before foodSpace and after resnet and text embedder)
                {'params': vision_params, 'lr': opts.lr*opts.freeVision, 'weight_decay': opts.weight_decay },  # resnet embeddings
                {'params': wordEmb_params, 'lr': opts.lr*opts.freeWordEmb, 'weight_decay': opts.weight_decay },# word embeddings
                {'params': text_params, 'lr': opts.lr*opts.freeText, 'weight_decay': opts.weight_decay } # text embedder params except word embeddings
        ], lr=opts.lr*opts.freeHeads, weight_decay=opts.weight_decay)

    model = torch.nn.DataParallel(model)

    if opts.resume:
        if os.path.isdir(opts.resume):
            model_path = glob.glob(os.path.join(opts.resume,'models/model_BEST_REC_*'))[0]
            print("=> loading checkpoint '{}'".format(model_path))
            checkpoint = torch.load(model_path)
            params_path = '/'.join(model_path.split('/')[:-2]+['opts.txt'])
            print("=> use parameters from '{}'".format(params_path))
            opts = get_variable(params_path, opts=opts, exclude=['gpu', 'no_cuda', 'warmup', 'decayLR'])
            opts.start_epoch = checkpoint['epoch'] + 1
            best_val = checkpoint['curr_val']
            best_rec = sum([v for k, v in checkpoint['curr_recall'].items()])
            model.load_state_dict(checkpoint['state_dict'])
            optimizer = checkpoint['optimizer']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(opts.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(opts.resume))
            best_val = float('inf')
            best_rec = 0
    else:
        best_val = float('inf')
        best_rec = 0
    since_improvement = 0

    print('There are %d parameter groups' % len(optimizer.param_groups))
    print('\tInitial heads params lr: %f' % optimizer.param_groups[0]['lr'])
    print('\tInitial vision params lr: %f' % optimizer.param_groups[1]['lr'])
    print('\tInitial wordEmb params lr: %f' % optimizer.param_groups[2]['lr'])
    print('\tInitial text params lr: %f' % optimizer.param_groups[3]['lr'])


    cudnn.benchmark = True

    for epoch in range(opts.start_epoch, opts.epochs):
        print('Started epoch {}/{}'.format(epoch+1, opts.epochs))
        print('Saving to: {}'.format(timestamp))
        inds = np.arange(len(train_data)//1).tolist()
        random.shuffle(inds)
        inds = np.asanyarray(inds)

        print('\tStarted training...')

        # preparing the training loader
        transform=transforms.Compose([transforms.RandomChoice([
                                        PadToSquareResize(resize=256, padding_mode='random'),
                                        transforms.Resize((256, 256))]),
                                    transforms.RandomRotation(10),
                                    transforms.RandomCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    normalize])
        train_data = foodSpaceLoader(transform=transform,
                                  partition='train',
                                  aug=opts.textAug,
                                  opts=opts)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=opts.batch_size,
                                                   shuffle=False,
                                                   sampler=SubsetSequentialSampler(inds),
                                                   num_workers=opts.workers,
                                                   pin_memory=False,
                                                   drop_last=True,
                                                   worker_init_fn=worker_init_fn,
                                                   collate_fn=my_collate)
        iter = train(train_loader, train_data, model,  optimizer, epoch, iter, train_writer, train_writer_text)
        
        
        if (epoch) % opts.valfreq == 0:
            curr_val, curr_recall = validate(val_loader, model, iter, train_writer)
            if curr_val > best_val:
                since_improvement += 1
            else:
                since_improvement = 0
            cum_recall = sum([v for k, v in curr_recall.items()])
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer,
                'since_improvement': since_improvement,
                'freeVision': opts.freeVision,
                'freeHeads': opts.freeHeads,
                'freeWordEmb': opts.freeWordEmb,
                'curr_val': curr_val,
                'curr_recall': curr_recall,
                'iteration': iter,
                'lr': opts.lr
                }
            filename = opts.snapshots + 'model.pth.tar'
            torch.save(state, filename)
            if curr_val < best_val:
                if glob.glob(opts.snapshots + 'model_BEST_VAL*'): os.remove(glob.glob(opts.snapshots + 'model_BEST_VAL*')[0])
                filename = opts.snapshots + 'model_BEST_VAL_e%03d_v-%.3f_cr-%.4f.pth.tar' % (
                state['epoch'], state['curr_val'], cum_recall)
                torch.save(state, filename)
            if cum_recall > best_rec:
                if glob.glob(opts.snapshots + 'model_BEST_REC*'): os.remove(glob.glob(opts.snapshots + 'model_BEST_REC*')[0])
                filename = opts.snapshots + 'model_BEST_REC_e%03d_v-%.3f_cr-%.4f.pth.tar' % (
                state['epoch'], state['curr_val'], cum_recall)
                torch.save(state, filename)
            best_val = min(curr_val, best_val)
            best_rec = max(cum_recall, best_rec)

            train_writer.add_scalar("val_medR", curr_val, iter)
            test_writer_text.write(str(iter)+','+str(curr_val)+'\n')
            test_writer_text.flush()
            train_writer.flush()
            print('\t** Validation: %f (best) - %d (since_improvement)' % (best_val, since_improvement))


        if str(epoch + 1) in str(opts.warmup).split(','):
            opts.freeVision = True
            opts.freeHeads = True
            opts.freeWordEmb = True if opts.w2vTrain else False
            optimizer.param_groups[0]['lr'] = opts.lr * opts.freeHeads
            optimizer.param_groups[1]['lr'] = opts.lr * opts.freeVision
            optimizer.param_groups[2]['lr'] = opts.lr * opts.freeWordEmb
            optimizer.param_groups[3]['lr'] = opts.lr * opts.freeText

            # opts.ohem = True


        if str(epoch + 1) in str(opts.decayLR).split(','):
            opts.freeVision = True
            opts.freeHeads = True
            opts.freeWordEmb = True if opts.w2vTrain else False
            opts.lr = opts.lr * 0.1
            optimizer.param_groups[0]['lr'] = opts.lr * opts.freeHeads
            optimizer.param_groups[1]['lr'] = opts.lr * opts.freeVision
            optimizer.param_groups[2]['lr'] = opts.lr * opts.freeWordEmb
            optimizer.param_groups[3]['lr'] = opts.lr * opts.freeText


def train(train_loader, train_data, model, optimizer, epoch, iter, train_writer, train_writer_text):
    losses = AverageMeter()
    dist_ap = AverageMeter()
    dist_an = AverageMeter()
    model.train()

    tmp = None
    for i, batch in enumerate(train_loader):
        input = batch[0]
        rec_ids = batch[1]

        iter += 1

        output = model(input, opts)

        loss, d_ap, d_an = calculate_loss(output, return_dp_dn=True)


        with torch.no_grad():
            losses.update(loss.item(), input[0].size(0))
            dist_ap.update(d_ap.item(), input[0].size(0))
            dist_an.update(d_an.item(), input[0].size(0))
            if i % 100 == 0:
                print('\t\t\tIteration: {0}\t'
                      'loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'pos {pos.avg:.4f}\t'
                      'neg {neg.avg:.4f}\t'
                      'heads ({headsLR}) - vision ({visionLR}) - wordEmb ({wordEmbLR}) - text ({textLR}))\t'.format(
                    i, loss=losses,
                    pos=dist_ap,
                    neg=dist_an,
                    headsLR=optimizer.param_groups[0]['lr'],
                    visionLR=optimizer.param_groups[1]['lr'],
                    wordEmbLR=optimizer.param_groups[2]['lr'],
                    textLR =optimizer.param_groups[3]['lr']))
                train_writer.add_scalar("loss", loss.item(), iter)
                train_writer.add_scalar("cos_loss", loss.item(), iter)
                train_writer.add_scalar("cos_loss", loss.item(), iter)
                train_writer_text.write(str(iter)+','+str(loss.item())+'\n')
                train_writer_text.flush()
            dist_ap.reset()
            dist_an.reset()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        del loss, output

    print('\t\tSubepoch: {0}\t'
              'cos_loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'recipe ({headsLR}) - vision ({visionLR}) - wordEmb ({wordEmbLR}) - text ({textLR}) \t'.format(
               epoch+1, loss=losses,
               headsLR=optimizer.param_groups[0]['lr'],
               visionLR=optimizer.param_groups[1]['lr'],
               wordEmbLR=optimizer.param_groups[2]['lr'],
               textLR =optimizer.param_groups[3]['lr']))
    return iter


def validate(val_loader, model, iter, train_writer):
    losses = AverageMeter()
    model.eval()
    for i, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
        input = batch[0]
        rec_ids = batch[1]
        iter += 1
        with torch.no_grad():
            output = model(input, opts)

            if i == 0:
                data0 = output[0].data.cpu().numpy()
                data1 = output[1].data.cpu().numpy()
                data2 = rec_ids
            else:
                data0 = np.concatenate((data0, output[0].data.cpu().numpy()), axis=0)
                data1 = np.concatenate((data1, output[1].data.cpu().numpy()), axis=0)
                data2 = np.concatenate((data2, rec_ids), axis=0)

            loss = calculate_loss(output)

            losses.update(loss.item(), input[0].size(0))

    medR, recall = rank(opts, data0, data1, data2)

    train_writer.add_scalar("val_loss", losses.avg, iter)
    train_writer.add_scalar("recall_1", recall[1], iter)
    train_writer.add_scalar("recall_5", recall[5], iter)
    train_writer.add_scalar("recall_10", recall[10], iter)
    train_writer.add_scalar("recall_1-5-10", recall[10]+recall[5]+recall[1], iter)
    print('\t* Val medR {medR:.4f}\tRecall {recall}'.format(medR=medR, recall=recall))

    return medR, recall


def rank(opts, img_embeds, rec_embeds, names):
    st = random.getstate()
    random.seed(opts.seed)
    idxs = np.argsort(names)
    names = names[idxs]
    idxs = range(opts.medr)

    glob_rank = []
    glob_recall = {1:0.0,5:0.0,10:0.0}
    for i in range(10):
        ids = random.sample(range(0,len(names)), opts.medr)
        img_sub = img_embeds[ids,:]
        rec_sub = rec_embeds[ids,:]

        if opts.embtype == 'image':
            sims = np.dot(img_sub,rec_sub.T)# im2recipe
        else:
            sims = np.dot(rec_sub,img_sub.T)# recipe2im

        med_rank = []
        recall = {1:0.0,5:0.0,10:0.0}
        for ii in idxs:
            # sort indices in descending order
            sorting = np.argsort(sims[ii,:])[::-1].tolist()

            # find where the index of the pair sample ended up in the sorting
            pos = sorting.index(ii)

            if (pos+1) == 1:
                recall[1] += 1
            if (pos+1) <= 5:
                recall[5] += 1
            if (pos+1) <= 10:
                recall[10] += 1

            med_rank.append(pos+1)

        for i in recall.keys():
            recall[i] = recall[i]/opts.medr

        med = np.median(med_rank)
        for i in recall.keys():
            glob_recall[i] += recall[i]
        glob_rank.append(med)

    for i in glob_recall.keys():
        glob_recall[i] = glob_recall[i]/10
    random.setstate(st)

    return np.average(glob_rank), glob_recall


def calculate_loss(output, return_dp_dn=False):
    
    if opts.intraModalLoss:
        label = list(range(0, output[0].shape[0]))
        label.extend(label)
        label = np.array(label)
        label = torch.tensor(label).cuda().long()
        feat = torch.cat((output[0], output[1]))
        dist_mat = cosine_distance(feat, feat)
        N = dist_mat.size(0)
        is_pos = label.expand(N, N).eq(label.expand(N, N).t())
        is_neg = label.expand(N, N).ne(label.expand(N, N).t())
    else:
        label = list(range(0, output[0].shape[0]))
        label = np.array(label)
        label = torch.tensor(label).cuda().long()
        dist_mat = cosine_distance(output[0], output[1])
        N = dist_mat.size(0)
        is_pos = label.expand(N, N).eq(label.expand(N, N).t())
        is_neg = label.expand(N, N).ne(label.expand(N, N).t())

    # `dist_ap` means distance(anchor, positive) both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, relative_p_inds = torch.max(dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    if opts.ohem:
        # `dist_an` means distance(anchor, negative) both `dist_an` and `relative_n_inds` with shape [N, 1]
        dist_an, relative_n_inds = torch.min(dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    else:
        dist_an = torch.mean(dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)

    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    cos_loss = torch.nn.functional.relu((dist_ap - dist_an) + opts.alpha).mean()  # margin triplet loss

    if return_dp_dn:
        return cos_loss, dist_ap.mean(), dist_an.mean()
    else:
        return cos_loss


def save_checkpoint(state):
    filename = opts.snapshots + 'model.pth.tar'
    torch.save(state, filename)




if __name__ == '__main__':
    main()


