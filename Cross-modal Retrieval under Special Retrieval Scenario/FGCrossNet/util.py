import torch
import os
    
class AverageMeter(object):
    """Computes and stores the average and current value"""
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

class Log(object):
    def save_train_info(self, epoch, batch, maxbatch, losses):
        log_dir = './log'
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        
        log_file = os.path.join(log_dir, 'log_train.txt')
        if not os.path.exists(log_file):
            os.mknod(log_file)

        with open(log_file, 'a') as f:
            f.write('Train <==> Epoch: [{0}][{1}/{2}]\t'
                    'All Loss {loss.val:.4f} ({loss.avg:.4f})\n'.format(epoch, batch, maxbatch, loss = losses))