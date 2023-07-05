import argparse
import json

def get_parser():
    parser = argparse.ArgumentParser(description='text2image GAN parameters')
    # general
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--device', nargs='+', type=int)
    parser.add_argument('--resume_G', type=str, default="", help="path to the trained G model")
    parser.add_argument('--resume_D', type=str, default="", help="path to the trained D model")
    parser.add_argument('--language', default=0, type=int, help='0: EN, 1: EN-DE-EN, 2: EN-RU-EN, 3: DE, 4: RU, 5: FR')

    # data
    parser.add_argument('--img_path', default='./data/Recipe1M/images')
    parser.add_argument('--data_path', default='./data/loader_data')
    parser.add_argument('--r1m_path', default='./data/Recipe1M')
    parser.add_argument('--encoder_dir', default='')
    parser.add_argument('--workers', default=5, type=int)
    parser.add_argument('--maxImgs', default=5, type=int)

    # model
    parser.add_argument('--batch_size', default=48, type=int)

    # im2recipe model
    parser.add_argument('--embDim', default=1024, type=int)
    parser.add_argument('--numClasses', default=1048, type=int)
    parser.add_argument('--imFeatDim', type=int, default=2048)

    #text-img
    parser.add_argument('--Z_DIM', type=int , default=100, help='noise dimension for image generation')
    parser.add_argument('--DF_DIM', type=int , default=64, help='D dimension')
    parser.add_argument('--GF_DIM', type=int , default=64, help='G dimension')
    parser.add_argument('--EMBEDDING_DIM', type=int , default=128, help='embedding dimension')
    parser.add_argument('--R_NUM', type=int , default=2, help='resudial unit number')
    parser.add_argument('--W_IMG_IMG_COS_LOSS', type=float, default=32.0, help='if use image-image cosine distance')
    parser.add_argument('--W_IMG_REC_COS_LOSS', type=float, default=32.0, help='if use image-recipe cosine distance')

    # training
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--drop_G_lr_epoch', type=int, default=30)
    parser.add_argument('--drop_D_lr_epoch', type=int, default=30)
    parser.add_argument('--auto_drop_lr', type=bool, default=False)
    parser.add_argument('--auto_drop_times', type=int, default=3)
    parser.add_argument('--auto_drop_after_epochs', type=int, default=10)

    # testing
    parser.add_argument('--trained_G_model_path', type=str, default="")



    return parser


def save_opts(args, filename):
    with open(filename, "w") as fp:
        json.dump(args.__dict__, fp, indent=4)
