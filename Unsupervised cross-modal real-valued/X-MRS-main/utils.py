import copy
import os
import random
import numbers
import time
import torch
import torchvision.transforms.functional
import numpy as np
import simplejson as json

from PIL import Image
import pathlib as plb
from torch.utils.data.sampler import Sampler
from torchvision import transforms


class SubsetSequentialSampler(Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


class PadToSquareResize(object):
    def __init__(self, resize=None, fill=0, padding_mode='constant', interpolation=transforms.InterpolationMode.BILINEAR):
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric', 'random']

        self.fill = fill
        self.padding_mode = padding_mode
        self.resize = resize
        self.interpolation = interpolation

    def __call__(self, img):
        if img.size[0] < img.size[1]:
            pad = img.size[1] - img.size[0]
            self.padding = (int(pad/2), 0, int(pad/2 + pad%2), 0)
        elif img.size[0] > img.size[1]:
            pad = img.size[0] - img.size[1]
            self.padding = (0, int(pad/2), 0, int(pad/2 + pad%2))
        else:
            self.padding = (0, 0, 0, 0)

        if self.padding_mode == 'random':
            pad_mode = random.choice(['constant', 'edge', 'reflect', 'symmetric'])
        else:
            pad_mode = self.padding_mode

        if self.resize is None:
            return torchvision.transforms.functional.pad(img, self.padding, self.fill, self.padding_mode)
        else:
            return torchvision.transforms.functional.resize(torchvision.transforms.functional.pad(img, self.padding, self.fill, pad_mode), self.resize, self.interpolation)


    def __repr__(self):
        return self.__class__.__name__ + '(padding={0}, fill={1}, padding_mode={2})'.\
            format(self.padding, self.fill, self.padding_mode)


class AverageMeter(object):
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


def get_variable(file, opts, exclude):
    with open(file) as f:
        for line in f:
            name, var = line.partition(":")[::2]
            if name.strip() in exclude:
                continue
            if name.strip() in opts:
                if var.strip().lower() in ('yes', 'true', 't', 'y'):
                    t = True
                elif var.strip().lower() in ('no', 'false', 'f', 'n'):
                    t = False
                else:
                    try:
                        t = int(var)
                    except:
                        if '.' in var:
                            try:
                                t = float(var)
                            except:
                                t = var.strip()
                        else:
                            t = var.strip()
                opts.__setattr__(name.strip(), t)
    return opts


def cosine_distance(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)


def worker_init_fn(worker_id):
    seed = worker_id
    np.random.seed(seed)


def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference


TicToc = TicTocGenerator()


def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )


def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)


def dspath(ext, ROOT, **kwargs):
    return os.path.join(ROOT,ext)


class Layer(object):
    L1 = 'layer1'
    L1_de = 'layer1_de'
    L1_de_en = 'layer1_de-en'
    L1_ru = 'layer1_ru'
    L1_ru_en = 'layer1_ru-en'
    L1_fr = 'layer1_fr'
    L2 = 'layer2'
    L3 = 'layer3'
    INGRS = 'det_ingrs'
    GOODIES = 'goodies'

    @staticmethod
    def load(name, ROOT, **kwargs):
        with open(dspath(name + '.json',ROOT, **kwargs)) as f_layer:
            return json.load(f_layer)

    @staticmethod
    def merge(layers, ROOT,copy_base=False, **kwargs):
        layers = [l if isinstance(l, list) else Layer.load(l, ROOT, **kwargs) for l in layers]
        base = copy.deepcopy(layers[0]) if copy_base else layers[0]
        entries_by_id = {entry['id']: entry for entry in base}
        for layer in layers[1:]:
            for entry in layer:
                base_entry = entries_by_id.get(entry['id'])
                if not base_entry:
                    continue
                base_entry.update(entry)
        return base

    @staticmethod
    def merge2(layers_name, ROOT,copy_base=False, **kwargs):
        layers = [l if isinstance(l, list) else Layer.load(l, ROOT, **kwargs) for l in layers_name]
        base = copy.deepcopy(layers[0]) if copy_base else layers[0]
        entries_by_id = {entry['id']: entry for entry in base}
        for layer, name in zip(layers[1:],layers_name[1:]):
            for entry in layer:
                if name == 'det_ingrs':
                    entry['ingredients_ext'] = entry.pop('ingredients')
                base_entry = entries_by_id.get(entry['id'])
                if not base_entry:
                    continue
                base_entry.update(entry)
        return base


def tok(text, ts=False):
    if not ts:
        ts = [',','.',';','(',')','?','!','&','%',':','*','"']
    for t in ts:
        text = text.replace(t,' ' + t + ' ')
    return text


def untok(text, ts=False):
    if not ts:
        ts = [',','.',';','(',')','?','!','&','%',':','*','"']
    for t in ts:
        text = text.replace(' ' + t + ' ', t)
    return text


def get_filename(filepath):
    return str(plb.Path(filepath).stem)


def exists(filename):
    return plb.Path(filename).exists()


def make_dir(dirname):
    plb.Path(dirname).mkdir(parents=True, exist_ok=True)


def set_gpu(gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id) if gpu_id >= 0 else ""


def get_list_of_files(dir_path):
    root_path = plb.Path(dir_path)
    ret = []
    for filepath in root_path.iterdir():
        if filepath.is_file():
            ret.append(filepath.name)
    return ret


#------------------------------------------------------------------------------
def load_model_state(model, state_dict, strict=True):
    try:
        model.load_state_dict(state_dict)
        return True
    except Exception as e:
        pass
    # try to load on GPU
    try:
        print("Retry loading by moving model to GPU")
        model.cuda()
        model.load_state_dict(state_dict, strict=strict)
        return True
    except Exception as e:
        pass
    # try to load from parallel module
    try:
        print("Retry by loading parallel model")
        temp_state_dict = state_dict.copy()
        for k, v in state_dict.items():
            temp_state_dict[k.replace('module.', '')] = temp_state_dict.pop(k)
        model.load_state_dict(temp_state_dict, strict=strict)
        return True
    except Exception as e:
        print(e)
        print("Loading Failed")
        return False


def load_torch_model(model, filename, strict=True):
    try:
        saved_state = torch.load(filename)
        ret = load_model_state(model, saved_state, strict=strict)
        if not ret:
            ret = load_model_state(model, saved_state["state_dict"], strict=strict)
        return ret
    except Exception as e:
        print("Couldn't open save file")
        print(e)
        return False


def get_image_decoder():
    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.255]
    )
    return transforms.Compose([inv_normalize, transforms.ToPILImage()])


IMAGE_DECODER = None


def decode_image(input_tensor):
    global IMAGE_DECODER
    if not IMAGE_DECODER:
        IMAGE_DECODER = get_image_decoder()
    return IMAGE_DECODER(input_tensor)