import logging
import time
import argparse
import train_div

def main(opt):
    sess = train_div.Session(opt)
    num_epoch = 0

    if opt.EVAL == True:
        sess.load_checkpoints()
        sess.eval()

    else:
        for epoch in range(opt.num_epochs):
            # train the Model
            if epoch < 100:
                sess.train(epoch)
            else:
                sess.train(epoch)
            # eval the Model
            if (epoch + 1) % opt.EVAL_INTERVAL == 0:
                num_epoch = sess.eval(step=epoch + 1, num_epoch=num_epoch)
            if num_epoch > 10:
                break
        # save the model
        sess.opt.EVAL = True
        sess.logger.info('---------------------------Test------------------------')
        sess.load_checkpoints()
        sess.eval()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', default='flickr', help='flickr, nus, wiki')
    parser.add_argument('--save_model_path', default='checkpoint/', help='path to save_model')
    parser.add_argument('--num_epochs', default=100, type=int, help='Number of training epochs.')
    parser.add_argument('--batch_size', default=32, type=int, help='Size of a training mini-batch.')
    parser.add_argument('--learning_rate', default=0.005, type=float, help='Initial learning rate.')
    parser.add_argument('--weight_decay', default=0.0005, type=float, help='weight_decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='weight_decay')
    parser.add_argument('--EVAL', default=False, type=bool, help='train or test')
    parser.add_argument('--EVAL_INTERVAL', default=1, type=float, help='train or test')

    parser.add_argument('--bit', default=64, type=int, help='128, 64, 32,16')

    parser.add_argument('--dw', default=1, type=float, help='loss1-alpha')
    parser.add_argument('--cw', default=1, type=float, help='loss2-beta')

    parser.add_argument('--K', default=1.5, type=float, help='pairwise distance resize')

    parser.add_argument('--a1', default=0.01, type=float, help='1 order distance')
    parser.add_argument('--a2', default=0.3, type=float, help='2 order distance')

    '''
    in this code the knn number is different from paper.
    the relationship between code and paper:
    for flickr and nus: 5000 - knn number from paper = knn number from code
    for wiki 2100 - knn number from paper = knn number from code
    '''
    parser.add_argument('--knn_number', default=3000, type=int, help='1 order distance')

    parser.add_argument('--scale', default=4000, type=float, help='1 order distance')
    parser.add_argument('--minvalue', default=1, type=float, help='1 order distance')    
    opt = parser.parse_args()
    print(opt)

    main(opt)
