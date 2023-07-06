from torch import nn
import torch
import os

"""
Xie, De, et al. "Multi-Task Consistency-Preserving Adversarial Hashing for Cross-Modal Retrieval."
IEEE Transactions on Image Processing 29 (2020): 3626-3637.
DOI: 10.1109/TMM.2020.2969792
"""

class CPAH(nn.Module):
    def __init__(self, image_dim, text_dim, hidden_dim, hash_dim, label_dim):
        super(CPAH, self).__init__()
        self.module_name = 'CPAH'
        self.image_dim = image_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.hash_dim = hash_dim
        self.label_dim = label_dim

        class Unsqueezer(nn.Module):
            """
            Converts 2d input into 4d input for Conv2d layers
            """
            def __init__(self):
                super(Unsqueezer, self).__init__()

            def forward(self, x):
                return x.unsqueeze(1).unsqueeze(-1)

        self.image_module = nn.Sequential(
            nn.Linear(image_dim, hidden_dim, bias=True),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
        )
        self.text_module = nn.Sequential(
            nn.Linear(text_dim, hidden_dim, bias=True),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
        )

        self.hash_module = nn.ModuleDict({
            'img': nn.Sequential(
                nn.Linear(512, hash_dim, bias=True),
                nn.Tanh()
            ),
            'txt': nn.Sequential(
                nn.Linear(512, hash_dim, bias=True),
                nn.Tanh()
            ),
        })

        self.mask_module = nn.ModuleDict({
            'img': nn.Sequential(
                Unsqueezer(),
                nn.Conv1d(1, hidden_dim, kernel_size=(hidden_dim, 1), stride=(1, 1)),
                nn.Sigmoid()
            ),
            'txt': nn.Sequential(
                Unsqueezer(),
                nn.Conv1d(1, hidden_dim, kernel_size=(hidden_dim, 1), stride=(1, 1)),
                nn.Sigmoid()
            ),
        })

        # D (consistency adversarial loss)
        self.feature_dis = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 8, bias=True),
            nn.ReLU(True),
            nn.Linear(hidden_dim // 8, 1, bias=True)
        )

        # C (consistency classification)
        self.consistency_dis = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 8, bias=True),
            nn.ReLU(True),
            nn.Linear(hidden_dim // 8, 3, bias=True)
        )

        # classification
        self.classifier = nn.ModuleDict({
            'img': nn.Sequential(
                nn.Linear(hidden_dim, label_dim, bias=True),
                nn.Sigmoid()
            ),
            'txt': nn.Sequential(
                nn.Linear(hidden_dim, label_dim, bias=True),
                nn.Sigmoid()
            ),
        })

    def forward(self, r_img, r_txt):
        f_r_img = self.image_module(r_img)  # image feature
        f_r_txt = self.text_module(r_txt)  # text feature

        # MASKING
        mc_img = self.get_mask(f_r_img, 'img')  # modality common mask for img
        mc_txt = self.get_mask(f_r_txt, 'txt')  # modality common mask for txt
        mp_img = 1 - mc_img  # modality private mask for img
        mp_txt = 1 - mc_txt  # modality private mask for txt

        f_rc_img = f_r_img * mc_img  # modality common feature for img
        f_rc_txt = f_r_txt * mc_txt  # modality common feature for txt
        f_rp_img = f_r_img * mp_img  # modality private feature for img
        f_rp_txt = f_r_txt * mp_txt  # modality private feature for txt

        # HASHING

        h_img = self.get_hash(f_rc_img, 'img')  # img hash
        h_txt = self.get_hash(f_rc_txt, 'txt')  # txt hash

        return h_img, h_txt, f_rc_img, f_rc_txt, f_rp_img, f_rp_txt

    def get_mask(self, x, modality):
        return self.mask_module[modality](x).squeeze()

    def get_hash(self, x, modality):
        return self.hash_module[modality](x).squeeze()

    def generate_img_code(self, i):
        f_i = self.image_module(i)
        return self.hash_module['img'](f_i.detach())

    def generate_txt_code(self, t):
        f_t = self.text_module(t)
        return self.hash_module['txt'](f_t.detach())

    def load(self, path, use_gpu=False):
        if not use_gpu:
            self.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
        else:
            self.load_state_dict(torch.load(path))

    def save(self, name=None, path='./checkpoints', cuda_device=None):
        if not os.path.exists(path):
            os.makedirs(path)
        if cuda_device.type == 'cpu':
            torch.save(self.state_dict(), os.path.join(path, name))
        else:
            with torch.cuda.device(cuda_device):
                torch.save(self.state_dict(), os.path.join(path, name))
        return name

    def dis_D(self, f):
        return self.feature_dis(f).squeeze()

    def dis_C(self, f):
        return self.consistency_dis(f).squeeze()

    def dis_classify(self, f, modality):
        return self.classifier[modality](f).squeeze()
