import os
from tqdm import tqdm
import torch

from torch import distributed as dist
from utils import get_logger, get_summary_writer


class TrainBase(object):

    def __init__(self,
                args,
                rank=0):
        
        self.args = args
        os.makedirs(args.save_dir, exist_ok=True)
        self._init_writer()
        self.logger.info(self.args)
        self.rank = rank

        self._init_dataset()
        self._init_model()

        self.global_step = 0
        # self.global_step_t = 0
        self.max_mapi2t = 0
        self.max_mapt2i = 0
        self.best_epoch_i = 0
        self.best_epoch_t = 0

    def _init_dataset(self):
        self.train_loader = None
        self.query_loader = None
        self.retrieval_loader = None
    
    def _init_model(self):
        self.model = None
        self.model_ddp = None
    
    def _init_writer(self):
        self.logger = get_logger(os.path.join(self.args.save_dir, "train.log" if self.args.is_train else "test.log"))
        self.writer = get_summary_writer(os.path.join(self.args.save_dir, "tensorboard"))

    def run(self):
        if self.args.is_train:
            self.train()
        else:
            self.test()

    def change_state(self, mode):

        if mode == "train":
            self.model.train()
        elif mode == "valid":
            self.model.eval()
    
    def get_code(self, data_loader, length: int):

        img_buffer = torch.empty(length, self.args.output_dim, dtype=torch.float).to(self.rank)
        text_buffer = torch.empty(length, self.args.output_dim, dtype=torch.float).to(self.rank)

        for image, text, label, index in tqdm(data_loader):
            image = image.to(self.rank, non_blocking=True)
            text = text.to(self.rank, non_blocking=True)
            index = index.numpy()
            image_hash = self.model.encode_image(image)
            text_hash = self.model.encode_text(text)

            img_buffer[index, :] = image_hash.data
            text_buffer[index, :] = text_hash.data
        
        return img_buffer, text_buffer# img_buffer.to(self.rank), text_buffer.to(self.rank)
    
    def hash_loss(self, a: torch.Tensor):
        return torch.mean(torch.sqrt(torch.sum(torch.pow(torch.sign(a) - a, 2), dim=1))) * 0.5
    
    def similarity_loss(self):
        raise NotImplementedError("Function of 'similarity_loss' doesn't implement.")
    
    def save_model(self, epoch):
        torch.save(self.model.state_dict(), os.path.join(self.args.save_dir, "model-" + str(epoch) + ".pth"))
        self.logger.info("save mode to {}".format(os.path.join(self.args.save_dir, "model-" + str(epoch) + ".pth")))
    
    def train(self):
        raise NotImplementedError("Function of 'train' doesn't implement.")
    
    def valid(self):
        raise NotImplementedError("Function of 'valid' doesn't implement.")

    def test(self):
        raise NotImplementedError("Function of 'test' doesn't implement.")

    def compute_loss(self):
        raise NotImplementedError("Function of 'compute_loss' doesn't implement.")
