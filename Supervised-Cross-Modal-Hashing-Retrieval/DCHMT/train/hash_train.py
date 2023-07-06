from torch.nn.modules import loss
from model.hash_model import DCMHT as DCMHT
import os
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import scipy.io as scio


from .base import TrainBase
from model.optimization import BertAdam
from utils import get_args, calc_neighbor, cosine_similarity, euclidean_similarity
from utils.calc_utils import calc_map_k_matrix as calc_map_k
from dataset.dataloader import dataloader

class Trainer(TrainBase):

    def __init__(self,
                rank=0):
        args = get_args()
        super(Trainer, self).__init__(args, rank)
        self.logger.info("dataset len: {}".format(len(self.train_loader.dataset)))
        self.run()

    def _init_model(self):
        self.logger.info("init model.")
        linear = False
        if self.args.hash_layer == "linear":
            linear = True

        self.logger.info("ViT+GPT!")
        HashModel = DCMHT
        self.model = HashModel(outputDim=self.args.output_dim, clipPath=self.args.clip_path,
                            writer=self.writer, logger=self.logger, is_train=self.args.is_train, linear=linear).to(self.rank)
        if self.args.pretrained != "" and os.path.exists(self.args.pretrained):
            self.logger.info("load pretrained model.")
            self.model.load_state_dict(torch.load(self.args.pretrained, map_location=f"cuda:{self.rank}"))
        
        self.model.float()
        self.optimizer = BertAdam([
                    {'params': self.model.clip.parameters(), 'lr': self.args.clip_lr},
                    {'params': self.model.image_hash.parameters(), 'lr': self.args.lr},
                    {'params': self.model.text_hash.parameters(), 'lr': self.args.lr}
                    ], lr=self.args.lr, warmup=self.args.warmup_proportion, schedule='warmup_cosine', 
                    b1=0.9, b2=0.98, e=1e-6, t_total=len(self.train_loader) * self.args.epochs,
                    weight_decay=self.args.weight_decay, max_grad_norm=1.0)
                
        print(self.model)

    def _init_dataset(self):
        self.logger.info("init dataset.")
        self.logger.info(f"Using {self.args.dataset} dataset.")
        self.args.index_file = os.path.join("./dataset", self.args.dataset, self.args.index_file)
        self.args.caption_file = os.path.join("./dataset", self.args.dataset, self.args.caption_file)
        self.args.label_file = os.path.join("./dataset", self.args.dataset, self.args.label_file)
        train_data, query_data, retrieval_data = dataloader(captionFile=self.args.caption_file, 
                                        indexFile=self.args.index_file, 
                                        labelFile=self.args.label_file, 
                                        maxWords=self.args.max_words,
                                        imageResolution=self.args.resolution,
                                        query_num=self.args.query_num,
                                        train_num=self.args.train_num,
                                        seed=self.args.seed)
        self.train_labels = train_data.get_all_label()
        self.query_labels = query_data.get_all_label()
        self.retrieval_labels = retrieval_data.get_all_label()
        self.args.retrieval_num = len(self.retrieval_labels)
        self.logger.info(f"query shape: {self.query_labels.shape}")
        self.logger.info(f"retrieval shape: {self.retrieval_labels.shape}")
        self.train_loader = DataLoader(
                dataset=train_data,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                pin_memory=True,
                shuffle=True
            )
        self.query_loader = DataLoader(
                dataset=query_data,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                pin_memory=True,
                shuffle=True
            )
        self.retrieval_loader = DataLoader(
                dataset=retrieval_data,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                pin_memory=True,
                shuffle=True
            )

    def train_epoch(self, epoch):
        self.change_state(mode="train")
        self.logger.info(">>>>>> epochs: %d/%d"%(epoch, self.args.epochs))
        all_loss = 0
        times = 0
        for image, text, label, index in self.train_loader:
            self.global_step += 1
            times += 1
            image.float()
            if self.args.dataset not in ["flickr25k", "coco", "nuswide"]:
                label = torch.ones([image.shape[0]], dtype=torch.int)
                label = label.diag()
            # print(text.dtype)
            # text.float()
            # label.float()
            image = image.to(self.rank, non_blocking=True)
            text = text.to(self.rank, non_blocking=True)
            # print("text shape:", text.shape)
            index = index.numpy()
            # print(text.shape)
            hash_img, hash_text = self.model(image, text)
            if self.args.hash_layer == "select":
                hash_img = torch.cat(hash_img, dim=-1) if isinstance(hash_img, list) else hash_img.view(hash_img.shape[0], -1)
                hash_text = torch.cat(hash_text, dim=-1)if isinstance(hash_text, list) else hash_text.view(hash_text.shape[0], -1)
            loss = self.compute_loss(hash_img, hash_text, label, epoch, times)
            all_loss += loss 


            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.logger.info(f">>>>>> [{epoch}/{self.args.epochs}] loss: {all_loss.data / (len(self.train_loader))}, lr: {'-'.join([str('%.9f'%itm) for itm in sorted(list(set(self.optimizer.get_lr())))])}")

    def train(self):
        self.logger.info("Start train.")

        for epoch in range(self.args.epochs):
            self.train_epoch(epoch)
            self.valid(epoch)
            self.save_model(epoch)

        self.logger.info(f">>>>>>> FINISHED >>>>>> Best epoch, I-T: {self.best_epoch_i}, mAP: {self.max_mapi2t}, T-I: {self.best_epoch_t}, mAP: {self.max_mapt2i}")

    def bayesian_loss(self, a: torch.Tensor, b: torch.Tensor, label_sim: torch.Tensor):
        
        s = torch.matmul(a, b.t())
        b_loss = -torch.mean(label_sim * s - torch.log(1 + torch.exp(s)))

        return b_loss
    
    def distribution_loss(self, a: torch.Tensor, b: torch.Tensor, label_sim: torch.Tensor):
        """
        """
        kl_divergence = torch.mean(a * torch.log(a / (b + 0.001)))
        print("mean", torch.mean(a - b))
        print("kl", kl_divergence)
        return kl_divergence


    def similarity_loss(self, a: torch.Tensor, b: torch.Tensor, label_sim: torch.Tensor, threshold=0.05):
        
        # $\vartheta$
        vartheta = self.args.vartheta
        if self.args.sim_threshold != 0:
            threshold = self.args.sim_threshold
        similarity = (1 - cosine_similarity(a, b)) if self.args.similarity_function == "cosine" else euclidean_similarity(a, b)
        
        positive_similarity = similarity * label_sim
        # 只要cosine为负值的全都算为计算正确了，因为优化到2确实很难。
        negative_similarity = similarity * (1 - label_sim)
        
        if self.args.similarity_function == "cosine":
            positive_similarity = positive_similarity.clip(threshold) - threshold
            negative_similarity = negative_similarity.clip(max=1.)
            negative_similarity = torch.tensor([1.]).expand_as(negative_similarity).to(self.rank) * (1 - label_sim) - negative_similarity
        elif self.args.similarity_function == "euclidean":
            # 有euclidean距离可知，当有一半长度的hash码不同时，其negative_similarity距离应该是长度（concat操作将outputdim翻倍），所以这里clip掉认为认定的值
            # 人为认定的最大值是一半长度的hash码不同。
            max_value = float(self.args.output_dim * 2 * vartheta) ** 0.5
            negative_similarity = negative_similarity.clip(max=max_value)
            negative_similarity = torch.tensor([max_value]).expand_as(negative_similarity).to(self.rank) * (1 - label_sim) - negative_similarity

        if self.args.loss_type == "l1":
            positive_loss = positive_similarity.mean()
            negative_loss = negative_similarity.mean()
        elif self.args.loss_type == "l2":
            positive_loss = torch.pow(positive_similarity, 2).mean()
            negative_loss = torch.pow(negative_similarity, 2).mean()
        else:
            raise ValueError("argument of loss_type is not support.")
        
        return similarity, positive_loss, negative_loss

    def make_hash_code(self, code: list) -> torch.Tensor:

        code = torch.stack(code)
        # print(code.shape)
        code = code.permute(1, 0, 2)
        hash_code = torch.argmax(code, dim=-1)
        hash_code[torch.where(hash_code == 0)] = -1
        hash_code = hash_code.float()

        return hash_code

    def get_code(self, data_loader, length: int):

        img_buffer = torch.empty(length, self.args.output_dim, dtype=torch.float).to(self.rank)
        text_buffer = torch.empty(length, self.args.output_dim, dtype=torch.float).to(self.rank)

        for image, text, label, index in tqdm(data_loader):
            image = image.to(self.rank, non_blocking=True)
            text = text.to(self.rank, non_blocking=True)
            index = index.numpy()
            image_hash = self.model.encode_image(image)
            image_hash = self.make_hash_code(image_hash)
            text_hash = self.model.encode_text(text)
            text_hash = self.make_hash_code(text_hash)
            # image_hash.to(self.rank)
            # text_hash.to(self.rank)
            img_buffer[index, :] = image_hash.data
            text_buffer[index, :] = text_hash.data
        
        return img_buffer, text_buffer# img_buffer.to(self.rank), text_buffer.to(self.rank)
        
    def our_loss(self, image, text, label, epoch, times):
        loss = 0

        label_sim = calc_neighbor(label, label)
        if image.is_cuda:
            label_sim = label_sim.to(image.device)
        intra_similarity, intra_positive_loss, intra_negative_loss = self.similarity_loss(image, text, label_sim)
        inter_similarity_i, inter_positive_loss_i, inter_negative_loss_i = self.similarity_loss(image, image, label_sim)
        inter_similarity_t, inter_positive_loss_t, inter_negative_loss_t = self.similarity_loss(text, text, label_sim)

        intra_similarity_loss = (intra_positive_loss + intra_negative_loss) if self.args.similarity_function == "euclidean" else (intra_positive_loss + intra_negative_loss)
        inter_similarity_loss = inter_positive_loss_t + inter_positive_loss_i + (inter_negative_loss_i + inter_negative_loss_t) if self.args.similarity_function == "euclidean" else inter_positive_loss_t + inter_positive_loss_i + inter_negative_loss_i + inter_negative_loss_t
        similarity_loss = inter_similarity_loss + intra_similarity_loss

        # if self.writer is not None:
        #     self.writer.add_scalar("intra similarity max", intra_similarity.max(), self.global_step)
        #     self.writer.add_scalar("intra similarity min", intra_similarity.min(), self.global_step)
        #     self.writer.add_scalar("intra positive loss", intra_positive_loss.data, self.global_step)
        #     self.writer.add_scalar("intra negative loss", intra_negative_loss.data, self.global_step)

        #     self.writer.add_scalar("inter image similarity max", inter_similarity_i.max(), self.global_step)
        #     self.writer.add_scalar("inter image similarity min", inter_similarity_i.min(), self.global_step)
        #     self.writer.add_scalar("inter image positive loss", inter_positive_loss_i.data, self.global_step)
        #     self.writer.add_scalar("inter image negative loss", inter_negative_loss_i.data, self.global_step)

        #     self.writer.add_scalar("inter text similarity max", inter_similarity_t.max(), self.global_step)
        #     self.writer.add_scalar("inter text similarity min", inter_similarity_t.min(), self.global_step)
        #     self.writer.add_scalar("inter text positive loss", inter_positive_loss_t.data, self.global_step)
        #     self.writer.add_scalar("inter text negative loss", inter_negative_loss_t.data, self.global_step)

        #     self.writer.add_scalar("intra similarity loss", intra_similarity_loss.data, self.global_step)
        #     self.writer.add_scalar("inter similarity loss", inter_similarity_loss.data, self.global_step)
        #     self.writer.add_scalar("similarity loss", similarity_loss.data, self.global_step)
        
        if self.args.hash_layer != "select":
            quantization_loss = (self.hash_loss(image) + self.hash_loss(text)) / 2
            loss = similarity_loss + quantization_loss
            if self.global_step % self.args.display_step == 0:
                self.logger.info(f">>>>>> Display >>>>>> [{epoch}/{self.args.epochs}], [{times}/{len(self.train_loader)}]: all loss: {loss.data}, "\
                    f"SIMILARITY LOSS, Intra, positive: {intra_positive_loss.data}, negitave: {intra_negative_loss.data}, sum: {intra_similarity_loss.data}, " \
                    f"Inter, image positive: {inter_positive_loss_i.data}, image negitave: {inter_negative_loss_i.data}, "\
                    f"text positive: {inter_positive_loss_t.data}, text negitave: {inter_negative_loss_t.data}, sum: {inter_similarity_loss.data}, "\
                    f"QUATIZATION LOSS, {quantization_loss.data}, "\
                    f"lr: {'-'.join([str('%.9f'%itm) for itm in sorted(list(set(self.optimizer.get_lr())))])}")
        else:
            loss = similarity_loss # + self.args.qua_gamma * (image_quantization_loss + text_quantization_loss)
            if self.global_step % self.args.display_step == 0:
                self.logger.info(f">>>>>> Display >>>>>> [{epoch}/{self.args.epochs}], [{times}/{len(self.train_loader)}]: all loss: {loss.data}, "\
                    f"SIMILARITY LOSS, Intra, positive: {intra_positive_loss.data}, negitave: {intra_negative_loss.data}, sum: {intra_similarity_loss.data}, " \
                    f"Inter, image positive: {inter_positive_loss_i.data}, image negitave: {inter_negative_loss_i.data}, "\
                    f"text positive: {inter_positive_loss_t.data}, text negitave: {inter_negative_loss_t.data}, sum: {inter_similarity_loss.data}, "\
                    # f"QUATIZATION LOSS, image: {image_quantization_loss.data}, text: {text_quantization_loss.data}, "\
                    f"lr: {'-'.join([str('%.9f'%itm) for itm in sorted(list(set(self.optimizer.get_lr())))])}")

        return loss
    
    def compute_loss(self, image, text, label, epoch, times):
        
        loss = self.our_loss(image, text, label, epoch, times)

        return loss

    def test(self, mode_name="i2t"):
        if self.args.pretrained == "":
            raise RuntimeError("test step must load a model! please set the --pretrained argument.")
        self.change_state(mode="valid")
        save_dir = os.path.join(self.args.save_dir, "PR_cruve")
        os.makedirs(save_dir, exist_ok=True)
        query_img, query_txt = self.get_code(self.query_loader, self.args.query_num) if self.args.hash_layer == "select" else super().get_code(self.query_loader, self.args.query_num)
        retrieval_img, retrieval_txt = self.get_code(self.retrieval_loader, self.args.retrieval_num) if self.args.hash_layer == "select" else super().get_code(self.retrieval_loader, self.args.retrieval_num)
        mAPi2t = calc_map_k(query_img, retrieval_txt, self.query_labels, self.retrieval_labels, None, self.rank)
        # print("map map")
        mAPt2i = calc_map_k(query_txt, retrieval_img, self.query_labels, self.retrieval_labels, None, self.rank)
        mAPi2i = calc_map_k(query_img, retrieval_img, self.query_labels, self.retrieval_labels, None, self.rank)
        mAPt2t = calc_map_k(query_txt, retrieval_txt, self.query_labels, self.retrieval_labels, None, self.rank)
        self.max_mapt2i = max(self.max_mapt2i, mAPt2i)
        self.logger.info(f">>>>>> MAP(i->t): {mAPi2t}, MAP(t->i): {mAPt2i}, MAP(t->t): {mAPt2t}, MAP(i->i): {mAPi2i}")

        query_img = query_img.cpu().detach().numpy()
        query_txt = query_txt.cpu().detach().numpy()
        retrieval_img = retrieval_img.cpu().detach().numpy()
        retrieval_txt = retrieval_txt.cpu().detach().numpy()
        query_labels = self.query_labels.numpy()
        retrieval_labels = self.retrieval_labels.numpy()

        result_dict = {
            'q_img': query_img,
            'q_txt': query_txt,
            'r_img': retrieval_img,
            'r_txt': retrieval_txt,
            'q_l': query_labels,
            'r_l': retrieval_labels
        }
        scio.savemat(os.path.join(save_dir, str(self.args.output_dim) + "-ours-" + self.args.dataset + "-" + mode_name + ".mat"), result_dict)
        self.logger.info(">>>>>> save all data!")


    def valid(self, epoch):
        self.logger.info("Valid.")
        self.change_state(mode="valid")
        query_img, query_txt = self.get_code(self.query_loader, self.args.query_num) if self.args.hash_layer == "select" else super().get_code(self.query_loader, self.args.query_num)
        retrieval_img, retrieval_txt = self.get_code(self.retrieval_loader, self.args.retrieval_num) if self.args.hash_layer == "select" else super().get_code(self.retrieval_loader, self.args.retrieval_num)
        # print("get all code")
        mAPi2t = calc_map_k(query_img, retrieval_txt, self.query_labels, self.retrieval_labels, None, self.rank)
        # print("map map")
        mAPt2i = calc_map_k(query_txt, retrieval_img, self.query_labels, self.retrieval_labels, None, self.rank)
        mAPi2i = calc_map_k(query_img, retrieval_img, self.query_labels, self.retrieval_labels, None, self.rank)
        mAPt2t = calc_map_k(query_txt, retrieval_txt, self.query_labels, self.retrieval_labels, None, self.rank)
        if self.max_mapi2t < mAPi2t:
            self.best_epoch_i = epoch
            self.save_mat(query_img, query_txt, retrieval_img, retrieval_txt, mode_name="i2t")
        self.max_mapi2t = max(self.max_mapi2t, mAPi2t)
        if self.max_mapt2i < mAPt2i:
            self.best_epoch_t = epoch
            self.save_mat(query_img, query_txt, retrieval_img, retrieval_txt, mode_name="t2i")
        self.max_mapt2i = max(self.max_mapt2i, mAPt2i)
        self.logger.info(f">>>>>> [{epoch}/{self.args.epochs}], MAP(i->t): {mAPi2t}, MAP(t->i): {mAPt2i}, MAP(t->t): {mAPt2t}, MAP(i->i): {mAPi2i}, \
                    MAX MAP(i->t): {self.max_mapi2t}, MAX MAP(t->i): {self.max_mapt2i}")

    def save_mat(self, query_img, query_txt, retrieval_img, retrieval_txt, mode_name="i2t"):

        save_dir = os.path.join(self.args.save_dir, "PR_cruve")
        os.makedirs(save_dir, exist_ok=True)

        query_img = query_img.cpu().detach().numpy()
        query_txt = query_txt.cpu().detach().numpy()
        retrieval_img = retrieval_img.cpu().detach().numpy()
        retrieval_txt = retrieval_txt.cpu().detach().numpy()
        query_labels = self.query_labels.numpy()
        retrieval_labels = self.retrieval_labels.numpy()

        result_dict = {
            'q_img': query_img,
            'q_txt': query_txt,
            'r_img': retrieval_img,
            'r_txt': retrieval_txt,
            'q_l': query_labels,
            'r_l': retrieval_labels
        }
        scio.savemat(os.path.join(save_dir, str(self.args.output_dim) + "-ours-" + self.args.dataset + "-" + mode_name + ".mat"), result_dict)
        self.logger.info(f">>>>>> save best {mode_name} data!")


