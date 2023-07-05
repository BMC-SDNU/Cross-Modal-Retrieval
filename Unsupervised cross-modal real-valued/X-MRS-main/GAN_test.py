import os
import random
import glob
import numpy as np
import torch
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from GAN_args import get_parser
from GAN_models import *
from tqdm import tqdm
import pickle
import simplejson as json

from models import load_image_encoder, rank_i2t, rank_3, rank_only
from data_loader import get_loader, get_image_loader_from_dir, get_embedding_data_loader
from utils import load_torch_model, decode_image, make_dir, exists, get_filename

import PIL.ImageOps as imops
from recipe_info import get_recipe_info


parser = get_parser()
opts = parser.parse_args()

language = {0: "en", 1: "de-en", 2: "ru-en", 3: "de", 4: "ru", 5: "fr"}
opts.encoder_model = glob.glob(os.path.join(opts.encoder_dir,'models','model_BEST_REC*_image_encoder*'))[0]
opts.embedding_file_train = glob.glob(os.path.join(opts.encoder_dir,'foodSpace*train_'+language[opts.language]+'*'))[0]  # Using English (EN) embeddings as default
opts.embedding_file_val = glob.glob(os.path.join(opts.encoder_dir,'foodSpace*val_'+language[opts.language]+'*'))[0]
opts.embedding_file_test = glob.glob(os.path.join(opts.encoder_dir,'foodSpace*test_'+language[opts.language]+'*'))[0]

def load_model(model_path):
    netG = G_NET()
    if not load_torch_model(netG, model_path, strict=True):
        raise ValueError("The model file does not exist or is invalid")
    netG.cuda()
    netG.eval()
    return netG


# load retrieval image model
image_encoder = None

upsample_layer = nn.Upsample(size=(224, 224), mode="bilinear", align_corners=True).cuda().eval()


#------------------------------------------------------------------------------
def generate_images(model_path, rec_ids, rec_embs, batch_size=400, noise_mode="BATCH_RANDOM", fixed_noise=None, save_dir_root="./data/GAN_generated/", collate=False):
    netG = load_model(model_path)
    if not netG:
        print("Failed to load model")
        return
    model_name = get_filename(model_path)

    if noise_mode == "BATCH_RANDOM":
        pass
    elif noise_mode == "SINGLE_RANDOM":
        if fixed_noise is None:
            fixed_noise = torch.randn(1, opts.Z_DIM).cuda()
    elif noise_mode == "ZERO":
        fixed_noise = torch.zeros(1, opts.Z_DIM).cuda()
    else:
        raise ValueError("unsupported noise mode")

    val_loader = get_embedding_data_loader(rec_ids, rec_embs, batch_size, drop_last=False)

    for i, data in enumerate(tqdm(val_loader)):
        recipes = data[0].cuda()
        ids = data[1]
        if noise_mode == "BATCH_RANDOM":
            noise = torch.randn(recipes.size(0), opts.Z_DIM).cuda()
        elif noise_mode == "SINGLE_RANDOM" or noise_mode == "ZERO":
            noise = fixed_noise.expand(recipes.size(0), opts.Z_DIM)

        with torch.no_grad():
            fake_images, _, _ = netG(noise, recipes)
            fake_images = upsample_layer(fake_images)
            fake_images = fake_images.detach().cpu()

            if collate:
                save_fake_images_2(fake_images, ids, save_dir_root, model_name)
            else:
                save_fake_images(fake_images, ids, os.path.join(save_dir_root, model_name))


def main_generate_images(model_path, embedding_file, noise_mode="BATCH_RANDOM", fixed_noise=None, save_dir_root="./data/GAN_generated/", collate=False):
    """
    noise mode: BATCH_RANDOM; SINGLE_RANDOM; ZERO
    """
    # load generator
    netG = load_model(model_path)
    if not netG:
        print("Failed to load model")
        return
    model_name = get_filename(model_path)

    img_path = opts.img_path #"/home/SERILOCAL/hai.xuanpham/Data/im2recipe/data/images_bicubic"
    data_path = opts.data_path #"/home/SERILOCAL/hai.xuanpham/Data/im2recipe/data/data_merged_ingrs4"
    #embedding_file = opts.embedding_file #"/home/SERILOCAL/hai.xuanpham/Data/im2recipe/data/prebuilt_embeddings/" + models[model_name]["encoder"]

    #batch_size = opts.batch_size
    batch_size = 400 #100

    # data loader
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224)])
    val_loader = get_loader(img_path, data_path, 'test', embedding_file,
                            val_transform,
                            batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    if noise_mode == "BATCH_RANDOM":
        pass
    elif noise_mode == "SINGLE_RANDOM":
        if fixed_noise is None:
            fixed_noise = torch.randn(1, opts.Z_DIM).cuda()
    elif noise_mode == "ZERO":
        fixed_noise = torch.zeros(1, opts.Z_DIM).cuda()
    else:
        raise ValueError("unsupported noise mode")

    for i, data in enumerate(tqdm(val_loader)):
        recipes = data[3].cuda()
        ids = data[5]
        if noise_mode == "BATCH_RANDOM":
            noise = torch.randn(recipes.size(0), opts.Z_DIM).cuda()
        elif noise_mode == "SINGLE_RANDOM" or noise_mode == "ZERO":
            noise = fixed_noise.expand(recipes.size(0), opts.Z_DIM)

        with torch.no_grad():
            fake_images, _, _ = netG(noise, recipes)
            fake_images = upsample_layer(fake_images)
            fake_images = fake_images.detach().cpu()

            if collate:
                save_fake_images_2(fake_images, ids, save_dir_root, model_name)
            else:
                save_fake_images(fake_images, ids, os.path.join(save_dir_root, model_name))


def save_fake_image(fake_image, filename):
    image = decode_image(fake_image)
    image = imops.autocontrast(image)
    image.save(filename)


def save_fake_images(fake_images, IDs, save_dir):
    make_dir(save_dir)
    N = len(IDs)
    for i, ID in enumerate(IDs):
        recipe = get_recipe_info(ID, opts.data_path, opts.img_path)
        if not recipe:
            continue
        filename = os.path.join(save_dir, recipe.get_recipe_filename() + ".jpg")
        save_fake_image(fake_images[i], filename)


def save_fake_images_2(fake_images, IDs, save_dir, model_name):
    N = len(IDs)
    for i, ID in enumerate(IDs):
        recipe = get_recipe_info(ID, opts.data_path, opts.img_path)
        if not recipe:
            continue
        dir_name = os.path.join(save_dir, recipe.get_recipe_filename())
        make_dir(dir_name)
        filename = os.path.join(dir_name, model_name + ".jpg")
        save_fake_image(fake_images[i], filename)
        recipe.save_recipe(dir_name)


#-------------------------------------------------------------------------------
def extract_image_embeddings(img_dir, emb_dir):
    # load image encoder
    image_encoder = load_image_encoder(opts.encoder_model, opts, use_cuda=True)

    # load image list
    image_transforms = transforms.Compose([transforms.Resize(224),
                                           #transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225])

                                           ])
    image_loader = get_image_loader_from_dir(img_dir, image_transforms=image_transforms, batch_size=300)

    for i, input in enumerate(tqdm(image_loader)):
        img = input[0].cuda()
        ids = input[1]

        with torch.no_grad():
            img_embs = image_encoder(img)
            if i == 0:
                data0 = img_embs.data.cpu().numpy()
                rec_ids = ids
            else:
                data0 = np.concatenate((data0, img_embs.data.cpu().numpy()), axis=0)
                rec_ids.extend(ids)
    # save the embedding
    make_dir(emb_dir)
    emb_path = os.path.join(emb_dir, "img_embed.npz")
    np.savez(emb_path, img_embeds=data0, rec_ids=np.array(rec_ids))


#-------------------------------------------------------------------------------
def load_embedding_both(fake_emb_path, original_emb_path):
    # load original embedding
    original_embs = np.load(original_emb_path)
    # load fake image embedding
    fake_img_embs = np.load(fake_emb_path)
    # match IDs
    original_map = {}
    org_rec_IDs = list(original_embs["rec_ids"])
    for i in range(len(org_rec_IDs)):
        original_map[org_rec_IDs[i]] = i
    fake_map = {}
    fake_rec_IDs = list(fake_img_embs["rec_ids"])
    for i in range(len(fake_rec_IDs)):
        fake_map[fake_rec_IDs[i]] = i
    return original_map, org_rec_IDs, original_embs["img_embeds"], original_embs["rec_embeds"], fake_map, fake_rec_IDs, fake_img_embs["img_embeds"]


def get_random_splits(rec_IDs, N_folds=10, fold_size=1000):
    N = len(rec_IDs)
    st = random.getstate()
    random.seed(1234)

    assert N_folds >= 1
    if fold_size <= 0:
        fold_size = len(rec_IDs) // N_folds

    splits = []
    for i in range(N_folds):
        idxs = random.sample(range(N), fold_size)
        splits.append(idxs)

    random.setstate(st)
    return splits


def calculate_rank_random(fake_emb_path, original_emb_path, save_dir, N_folds=10, fold_size=1000, load_splits=True):
    org_map, org_ids, org_img, org_rec, fake_map, fake_ids, fake_img = load_embedding_both(fake_emb_path, original_emb_path)

    if "val" in original_emb_path:
        split_file = "./data/GAN_val_splits.pkl"
    else:
        split_file = "./data/GAN_test_splits.pkl"
    if load_splits and exists(split_file):
        with open(split_file, "rb") as fp:
            splits = pickle.load(fp)
        fake_splits = splits["fake"]
        org_splits = splits["full"]
    else:
        fake_splits = get_random_splits(fake_ids, N_folds, fold_size)
        org_splits = []
        for sps in fake_splits:
            org_split = []
            for idx in sps:
                id = fake_ids[idx]
                full_idx = org_map[id]
                org_split.append(full_idx)
            org_splits.append(org_split)
        splits = {"fake": fake_splits, "full": org_splits}
        with open(split_file, "wb") as fp:
            pickle.dump(splits, fp)

    # calculate retrieval
    print("fake i -> t")
    medR_1, recall_1, avgR_1, dcg_1 = rank_3(fake_img, org_rec, img_splits=fake_splits,
                                             rec_splits=org_splits, mode="i2t")
    print("t -> fake i")
    medR_2, recall_2, avgR_2, dcg_2 = rank_3(fake_img, org_rec, img_splits=fake_splits,
                                             rec_splits=org_splits, mode="t2i")
    print("fake i -> i")
    medR_3, recall_3, avgR_3, dcg_3 = rank_3(fake_img, org_img, img_splits=fake_splits,
                                             rec_splits=org_splits, mode="i2t")
    print("i -> fake i")
    medR_4, recall_4, avgR_4, dcg_4 = rank_3(fake_img, org_img, img_splits=fake_splits,
                                             rec_splits=org_splits, mode="t2i")

    ret = {}
    ret["fi2t"] = [medR_1, recall_1, avgR_1, dcg_1]
    ret["t2fi"] = [medR_2, recall_2, avgR_2, dcg_2]
    ret["fi2i"] = [medR_3, recall_3, avgR_3, dcg_3]
    ret["i2fi"] = [medR_4, recall_4, avgR_4, dcg_4]

    save_file = os.path.join(save_dir, "rank_random_result.csv")
    i = 1
    while exists(save_file):
        save_file = os.path.join(save_dir, "rank_random_result_{:d}.csv".format(i))
        i += 1

    with open(save_file, "w") as txt_file:
        header = "mode, medR , r@1, r@5, r@10, avgR, DCG\n"
        txt_file.write(header)
        for key in ["fi2t", "t2fi", "fi2i", "i2fi"]:
            x = ret[key]
            line_txt = "{:.2f}, {:.4f}, {:.4f}, {:.4f}, {:.2f}, {:.4f}".format(x[0], x[1][1], x[1][5], x[1][10], x[2], x[3])
            txt_file.write(line_txt)
    print(ret)
    return ret


def find_smallest_number(numbers):
    min_n = min(numbers)
    indices = []
    for i, n in enumerate(numbers):
        if n == min_n:
            indices.append(i)
    return min_n, indices


def calculate_rank_full(fake_emb_path, original_emb_path, save_dir):
    org_map, org_ids, org_img, org_rec, fake_map, fake_ids, fake_img = load_embedding_both(fake_emb_path, original_emb_path)
    full_split = []
    for id in fake_ids:
        full_idx = org_map[id]
        full_split.append(full_idx)
    new_org_img = org_img[full_split]
    new_org_rec = org_rec[full_split]

    # calculate retrieval
    print("fake i -> t")
    medR_1, recall_1, dcg_1, ranks_1 = rank_only(fake_img, new_org_rec, mode="i2t")
    print("t -> fake i")
    medR_2, recall_2, dcg_2, ranks_2 = rank_only(fake_img, new_org_rec, mode="t2i")
    print("fake i -> i")
    medR_3, recall_3, avgR_3, ranks_3 = rank_only(fake_img, new_org_img, mode="i2t")
    print("i -> fake i")
    medR_4, recall_4, avgR_4, ranks_4 = rank_only(fake_img, new_org_img, mode="t2i")

    ret = {}
    ret["fi2t"] = [medR_1, recall_1, ranks_1]
    ret["t2fi"] = [medR_2, recall_2, ranks_2]
    ret["fi2i"] = [medR_3, recall_3, ranks_3]
    ret["i2fi"] = [medR_4, recall_4, ranks_4]

    save_file = os.path.join(save_dir, "rank_full_result.csv")
    lines = []
    with open(save_file, "w") as txt_file:
        header = "mode, medR , r@1, r@5, r@10\n"
        txt_file.write(header)
        for key in ["fi2t", "t2fi", "fi2i", "i2fi"]:
            x = ret[key]
            line_txt = "{:s}, {:.2f}, {:.4f}, {:.4f}, {:.4f}\n".format(key, x[0], x[1][1], x[1][5], x[1][10])
            txt_file.write(line_txt)
            lines.append(line_txt)
    print(lines)
    # save ranks
    with open(os.path.join(save_dir, "rank_full_result.pkl"), "wb") as fp:
        pickle.dump(ret, fp)

    # find the best samples
    save_file = os.path.join(save_dir, "best_ranks.json")
    d = {}
    for key in ["fi2t", "t2fi", "fi2i", "i2fi"]:
        ranks = ret[key][2]
        min_rank, indices = find_smallest_number(ranks)
        d[key] = {"min_rank": min_rank, "indices": [[idx, id] for idx, id in zip(indices, [fake_ids[i] for i in indices])]}
    with open(save_file, "w") as fp:
        json.dump(d, fp, indent=4)

    return ret


#-------------------------------------------------------------------------------
def test_one_model(model_path, full_rank=False):
    # generate random noise
    print("\t\t ---", model_path, "---")

    if os.path.exists("./data/GAN_test_splits.pkl"):
        os.remove("./data/GAN_test_splits.pkl")
    if os.path.exists("./data/GAN_val_splits.pkl"):
        os.remove("./data/GAN_val_splits.pkl")

    print("\t generate image from random noise")
    save_root_dir = os.path.join(os.path.dirname(opts.trained_G_model_path),'GAN_generated_collated')
    main_generate_images(model_path, opts.embedding_file_test, noise_mode="BATCH_RANDOM", save_dir_root=save_root_dir, collate=True)
    # zero noise
    print("\t generate image from zero noise")
    save_root_dir = os.path.join(os.path.dirname(opts.trained_G_model_path),'GAN_zero_noise')
    main_generate_images(model_path, opts.embedding_file_test, noise_mode="ZERO", save_dir_root=save_root_dir, collate=False)
    # extract fake image embeddings
    print("\t extract fake image embeddings")
    image_root_dir = os.path.join(os.path.dirname(opts.trained_G_model_path),'GAN_zero_noise',os.path.basename(opts.trained_G_model_path).split('.')[0])
    emb_root_dir = os.path.join(os.path.dirname(opts.trained_G_model_path),'GAN_zero_noise_embeddings')
    extract_image_embeddings(image_root_dir, emb_root_dir)
    # calculate rank random
    print("\t calculate rank 1k random")
    emb_root_dir = os.path.join(os.path.dirname(opts.trained_G_model_path),'GAN_zero_noise_embeddings')
    calculate_rank_random(os.path.join(emb_root_dir, "img_embed.npz"), opts.embedding_file_test, os.path.dirname(opts.trained_G_model_path), load_splits=True)

    # calculate full rank
    if full_rank:
        print("\t calculate rank full set")
        calculate_rank_full(os.path.join(emb_root_dir, "img_embed.npz"), opts.embedding_file_test, os.path.dirname(opts.trained_G_model_path) )
    # generate image from pruned embedding
    print("----- DONE! -----")


#-------------------------------------------------------------------------------
def validate_sub(val_loader, netG, image_encoder, test_mode="ZERO", embed="rec", return_id=False):
    fixed_noise = torch.zeros(1, opts.Z_DIM).cuda()

    for i, data in enumerate(tqdm(val_loader)):
        recipes_embs = data[3]
        if return_id:
            rec_ids = data[5]
        if embed == "rec":
            input_embs = recipes_embs.cuda()
        else:
            input_embs = data[4].cuda()
        if test_mode == "ZERO":
            noise = fixed_noise.expand(input_embs.size(0), opts.Z_DIM)
        else:
            noise = torch.randn(input_embs.size(0), opts.Z_DIM).cuda()
        with torch.no_grad():
            fake_images, _, _ = netG(noise, input_embs)
            fake_images = upsample_layer(fake_images)
            fake_feat = image_encoder(fake_images)

            if i == 0:
                data0 = fake_feat.data.cpu().numpy()
                data1 = recipes_embs.data.cpu().numpy()
                if return_id:
                    data2 = rec_ids
            else:
                data0 = np.concatenate((data0, fake_feat.data.cpu().numpy()), axis=0)
                data1 = np.concatenate((data1, recipes_embs.data.cpu().numpy()), axis=0)
                if return_id:
                    data2 = np.concatenate((data2, rec_ids), 0)

    if return_id:
        return data0, data1, data2
    return data0, data1


def validate(val_loader, netG, image_encoder, test_both=False, test_mode="ZERO", compare_rec_embedding=None, embed="rec"):
    data0, data1 = validate_sub(val_loader, netG, image_encoder, test_mode, embed)

    if compare_rec_embedding is not None:
        medR_i2t, recall_i2t = rank_i2t(opts.seed, data0, compare_rec_embedding)
    else:
        medR_i2t, recall_i2t = rank_i2t(opts.seed, data0, data1)
    print('I2T Val medR {medR:.4f}\t'
          'Recall {recall}'.format(medR=medR_i2t, recall=recall_i2t))

    if test_both:
        if compare_rec_embedding is not None:
            medR_t2i, recall_t2i = rank_i2t(opts.seed, compare_rec_embedding, data0)
        else:
            medR_t2i, recall_t2i = rank_i2t(opts.seed, data1, data0)
        print('I2T Val medR {medR:.4f}\t'
              'Recall {recall}'.format(medR=medR_t2i, recall=recall_t2i))
        return medR_i2t, recall_i2t, medR_t2i, recall_t2i
    else:
        return medR_i2t, recall_i2t


def validate_one_model(model_path, partition="test", test_both=False, test_embed="rec", test_retrieve_image=False):
    if test_embed not in ["rec", "img"]:
        raise NotImplementedError("not supported type of input embedding: " + test_embed)

    netG = load_model(model_path)
    if not netG:
        print("Failed to load netG")
        return

    # load image encoder
    image_encoder = load_image_encoder(opts.encoder_model, opts, use_cuda=True)
    if not image_encoder:
        print("Failed to load image encoder")
        return

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224)])

    embedding_file = opts.embedding_file_test

    val_loader = get_loader(opts.img_path, opts.data_path, partition, embedding_file,
                            val_transform,
                            batch_size=300, shuffle=False,
                            num_workers=4, pin_memory=True, drop_last=False)

    if test_retrieve_image:
        embs = np.load(embedding_file)
        compare_embs = embs["img_embeds"]
        del embs
    else:
        compare_embs = None

    if test_both:
        medR1, recall1, medR2, recall2 = validate(val_loader, netG, image_encoder, test_both, embed=test_embed, compare_rec_embedding=compare_embs)
        print("I2T: ", medR1, recall1)
        print("T2I: ", medR2, recall2)
    else:
        medR, recall = validate(val_loader, netG, image_encoder, embed=test_embed, compare_rec_embedding=compare_embs)
        print("I2T: ", medR, recall)


#================================================================================


if __name__ == "__main__":
    G_model_path = opts.trained_G_model_path
    test_one_model(G_model_path)