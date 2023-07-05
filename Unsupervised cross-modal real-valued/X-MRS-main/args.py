import argparse


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_parser():
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--no_cuda', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--experiment_sufix', default='', type=str)

    # data
    parser.add_argument('--img_path', default='./data/Recipe1M/images')
    parser.add_argument('--data_path', default='./data/loader_data')#merged_ingrs4')
    parser.add_argument('--workers', default=15, type=int)
    parser.add_argument('--model_init_path', default='', type=str, help="Path to MODEL")
    parser.add_argument('--resume', default='', type=str, help="Path to training FOLDER")

    # FoodSpaceNet model
    parser.add_argument('--embDim', default=1024, type=int)          # Dim of FoodSpace
    parser.add_argument("--w2vInit", type=str2bool, nargs='?', const=True, default=True, help="Initialize word embeddings with w2v model?")
    parser.add_argument('--maxSeqlen', default=20, type=int)         # Used when building LMDB
    parser.add_argument('--maxInsts', default=20, type=int)          # Used when building LMDB
    parser.add_argument('--maxImgs', default=5, type=int)            # Used when building LMDB
    parser.add_argument('--textmodel', default='mBERT_fulltxt', type=str)  
    parser.add_argument('--textinputs', default='title,ingr,inst', type=str)  # Pieces of recipe to use. Only valid if textmodel='AWE'
    parser.add_argument("--textAug", type=str, default='english,de,ru,fr', help="Use text augmentation: 'english', 'de', 'ru' and/or 'fr'. 'english' uses back-translation from 'de' and 'ru'")
    parser.add_argument('--BERT_layers', default=2, type=int)           # 
    parser.add_argument('--BERT_heads', default=2, type=int)           # 

    # training
    parser.add_argument('--batch_size', default=20, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--valfreq', default=1,type=int)
    parser.add_argument("--w2vTrain", type=str2bool, nargs='?', const=True, default=True, help="Allow word embeddings to be trained?")
    parser.add_argument("--freeVision", type=str2bool, nargs='?', const=True, default=True, help="Train vision parameters?")
    parser.add_argument("--freeHeads", type=str2bool, nargs='?', const=True, default=True, help="Train model embedding heads?")
    parser.add_argument("--freeWordEmb", type=str2bool, nargs='?', const=True, default=True, help="Train word embedding parameters?")
    parser.add_argument("--freeText", type=str2bool, nargs='?', const=True, default=True, help="Train text encoder parameters?")
    parser.add_argument("--ohem", type=str2bool, nargs='?', const=True, default=True, help="Train with hard mining?")
    parser.add_argument("--intraModalLoss", type=str2bool   , nargs='?', const=True, default=True, help="Allow intra-modality modality pairs in loss?")
    parser.add_argument('--warmup', default='1', type=str, help="Train with the many epochs until all weights are relased and trained")
    parser.add_argument('--decayLR', default='20', type=str, help="Decay LR by 0.1 at these epochs")
    parser.add_argument('--alpha', default=0.3, type=float)
    parser.add_argument('--language', default=0, type=int, help='0: EN, 1: EN-DE-EN, 2: EN-RU-EN, 3: DE, 4: RU, 5: FR')

    # testing
    parser.add_argument('--embtype', default='image', type=str) # [image|recipe] query type
    parser.add_argument('--medr', default=1000, type=int)


    return parser




