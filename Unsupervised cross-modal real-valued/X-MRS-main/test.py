import random
import numpy as np
import torch
import os
import torchvision.transforms as transforms
from data_loader import foodSpaceLoader
from data_loader import error_catching_loader
from args import get_parser
from utils import get_variable, PadToSquareResize
from tqdm import tqdm
import glob
import numpy as np


# =============================================================================
parser = get_parser()
opts = parser.parse_args()
opts.gpu = list(map(int, opts.gpu.split(',')))
print('Using GPU(s): ' + ','.join([str(x) for x in opts.gpu]))
print(torch.cuda.is_available())
# =============================================================================


PARTITIONS = ['test', 'val', 'train']
print(PARTITIONS)

torch.manual_seed(opts.seed)
np.random.seed(opts.seed)
if not opts.no_cuda:
    torch.cuda.manual_seed(opts.seed)


def main():
    global opts

    opts = get_variable('/'.join(opts.model_init_path.split('/')[:-2]+['opts.txt']), opts=opts, exclude=['language','textinputs','medr', 'no_cuda', 'model_init_path', 'img_path', 'data_path'])
    opts.maxImgs = 1

    import importlib.util
    spec = importlib.util.spec_from_file_location("models", '/'.join(opts.model_init_path.split('/')[:-2]+['models.py']))
    models = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(models)

    model = models.FoodSpaceNet(opts)
    if not opts.no_cuda:
        model.cuda()

    print("=> loading checkpoint '{}'".format(opts.model_init_path))
    if opts.no_cuda:
        checkpoint = torch.load(glob.glob(opts.model_init_path)[0],map_location='cpu')
    else:
        checkpoint = torch.load(glob.glob(opts.model_init_path)[0])
    opts.start_epoch = checkpoint['epoch']
    if 'module.' in list(checkpoint['state_dict'].keys())[0]:
        tmp_dict = checkpoint['state_dict'].copy()
        for k, v in checkpoint['state_dict'].items():
            tmp_dict[k.replace('module.', '')] = tmp_dict.pop(k)
        checkpoint['state_dict'] = tmp_dict.copy()
    model.load_state_dict(checkpoint['state_dict'], strict=False)



    from models import FoodSpaceImageEncoder
    image_encoder = FoodSpaceImageEncoder(opts)
    image_encoder.cuda()
    image_encoder.load_state_dict(checkpoint['state_dict'], strict=False)
    torch.save(image_encoder.state_dict(), opts.model_init_path.replace('.pth.tar','_image_encoder.pth.tar'))
    print('Saved image encoder')


    test(model)


def test(model):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    for partition in PARTITIONS:
        data = foodSpaceLoader(transform=transforms.Compose([
                                   PadToSquareResize(resize=256, padding_mode='reflect'),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   normalize]),
                               partition=partition,
                               aug='',
                               language=opts.language,
                               loader=error_catching_loader,
                               opts=opts)
        loader = torch.utils.data.DataLoader(data,
                                             batch_size=100,
                                             shuffle=False,
                                             num_workers=opts.workers,
                                             pin_memory=True)

        model.eval()
        for i, batch in tqdm(enumerate(loader), total=len(loader)):
            input_data = batch[0]
            ids = batch[1]
            with torch.no_grad():
                output = model(input_data, opts, return_visual_attention=True, return_text_attention=True)

                if not 'img_embeds' in locals():
                    img_embeds = output[0].detach().cpu().numpy()
                    rec_embeds = output[1].detach().cpu().numpy()
                    rec_ids = ids
                else:
                    img_embeds = np.concatenate((img_embeds, output[0].detach().cpu().numpy()), 0)
                    rec_embeds = np.concatenate((rec_embeds, output[1].detach().cpu().numpy()), 0)
                    rec_ids = np.concatenate((rec_ids, ids), 0)

        language = {0: "en", 1: "de-en", 2: "ru-en", 3: "de", 4: "ru", 5: "fr"}
        savedir = os.path.abspath(os.path.join(os.path.dirname(opts.model_init_path),'..'))
        savefile = 'foodSpace_vectors'+'_'+\
                    savedir.split('/')[-1].split('_')[0]+'_'+\
                    os.path.basename(opts.model_init_path).replace('.pth.tar','_'+partition+'_'+language[opts.language]+'.npz')        
        print('saving vectors to: ',os.path.join(savedir,savefile))
        np.savez(os.path.join(savedir,savefile),
                 img_embeds=img_embeds,
                 rec_embeds=rec_embeds,
                 rec_ids=rec_ids)


        opts.embtype = 'image'
        ranks = rank(opts, img_embeds, rec_embeds, rec_ids, return_NNs=True)
        print('Partition: {}\n\timage-to-recipe ({}):\n\t\tMedR: {}\n\t\tR@1:  {}\n\t\tR@5:  {}\n\t\tR@10: {}'
                .format(partition.upper(), opts.medr, np.mean(ranks[1]), ranks[2][1], ranks[2][5], ranks[2][10]))
        
        opts.embtype = 'recipe'
        ranks = rank(opts, img_embeds, rec_embeds, rec_ids, return_NNs=True)
        print('\trecipe-to-image ({}):\n\t\tMedR: {}\n\t\tR@1:  {}\n\t\tR@5:  {}\n\t\tR@10: {}'
                .format(opts.medr, np.mean(ranks[1]), ranks[2][1], ranks[2][5], ranks[2][10]))

        del img_embeds


def rank(opts, img_embeds, rec_embeds, rec_ids, runs=10, return_NNs=False):
    st = random.getstate()
    random.seed(opts.seed)
    names = rec_ids
    idxs = np.argsort(names)
    names = names[idxs]
    idxs = range(opts.medr)
    per_run_rank = []
    all_rank = []
    glob_recall = {1: 0.0, 5: 0.0, 10: 0.0}
    NNs = {}
    for i in range(runs):
        NNs[i] = {}
        ids = random.sample(range(0, len(names)), opts.medr)
        im_sub = img_embeds[ids, :]
        instr_sub = rec_embeds[ids, :]
        if opts.embtype == 'image':
            sims = np.dot(im_sub, instr_sub.T)  # for im2recipe
        else:
            sims = np.dot(instr_sub, im_sub.T)  # for recipe2im
        med_rank = []
        recall = {1: 0.0, 5: 0.0, 10: 0.0}
        for ii in idxs:
            sim = sims[ii, :]
            sorting = np.argsort(sim)[::-1].tolist()
            pos = sorting.index(ii)
            if return_NNs: NNs[i][names[ids[ii]]] = {'NNs':[names[ids[k]] for k in sorting[:10]],'rank':pos}
            if (pos + 1) == 1:
                recall[1] += 1
            if (pos + 1) <= 5:
                recall[5] += 1
            if (pos + 1) <= 10:
                recall[10] += 1
            med_rank.append(pos + 1)

        for i in recall.keys():
            recall[i] = recall[i] / opts.medr

        all_rank.append(med_rank)
        med = np.median(med_rank)

        for i in recall.keys():
            glob_recall[i] += recall[i]
        per_run_rank.append(med)

    for i in glob_recall.keys():
        glob_recall[i] = glob_recall[i] / 10
    random.setstate(st)

    if return_NNs:
        return all_rank, per_run_rank, glob_recall, NNs
    return all_rank, per_run_rank, glob_recall


if __name__ == '__main__':
    main()

