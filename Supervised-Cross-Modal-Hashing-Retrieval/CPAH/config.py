import warnings


class Default(object):

    flag = 'ucm'

    batch_size = 64
    image_dim = 4096
    hidden_dim = 512
    modals = 2
    valid = True  # whether to use validation
    valid_freq = 40
    max_epoch = 80

    bit = 64  # hash code length
    lr = 0.0001  # initial learning rate

    device = 'cuda:0'
    # device = 'cpu'

    # hyper-parameters
    alpha = 0.1  # from paper's Fig. 4
    beta = 1  # from paper's Fig. 4

    proc = 'test'

    seed = 42
    dataset_train_split = 0.5  # part of all data, that will be used for training
    dataset_query_split = 0.2  # part of evaluation data, that will be used for query

    use_aug_data = False

    def data(self, flag):
        if flag == 'mir':
            self.dataset = 'flickr25k'
            self.data_path = './data/FLICKR-25K.mat'
            self.db_size = 18015
            self.num_label = 24
            self.query_size = 2000
            self.text_dim = 1386
            self.training_size = 10000
        if flag == 'nus':
            self.dataset = 'nus-wide'
            self.data_path = './data/NUS-WIDE-TC21.mat'
            self.db_size = 193734
            self.num_label = 21
            self.query_size = 2100
            self.text_dim = 1000
            self.training_size = 10000
        if flag == 'ucm':
            self.dataset = 'ucm'
            self.data_path = './data/UCM_resnet18_bert_sum_12.h5'
            self.db_size = 9450
            self.num_label = 21
            self.query_size = 1050
            self.image_dim = 512
            self.text_dim = 768
            self.training_size = 5250
            self.lr = 0.001

            self.image_emb_for_model = "./data/image_emb_{}_aug_center_crop_only.h5".format(flag.upper())
            self.caption_emb_for_model = "./data/caption_emb_{}_aug.h5".format(flag.upper())
            self.image_emb_aug_for_model = "./data/image_emb_{}_aug_aug_center.h5".format(flag.upper())
            self.caption_emb_aug_for_model = "./data/caption_emb_{}_aug_rb.h5".format(flag.upper())
            self.dataset_json_for_model = "./data/augmented_{}.json".format(flag.upper())
        if flag == 'rsicd':
            self.dataset = 'rsicd'
            self.data_path = './data/RSICD_resnet18_bert_sum_12.h5'
            self.db_size = 52000
            self.num_label = 31
            self.query_size = 2605
            self.text_dim = 768
            self.image_dim = 512
            self.training_size = 30000
            self.lr = 0.0001

            self.image_emb_for_model = "./data/image_emb_{}_aug_center_crop_only.h5".format(flag.upper())
            self.caption_emb_for_model = "./data/caption_emb_{}_aug.h5".format(flag.upper())
            self.image_emb_aug_for_model = "./data/image_emb_{}_aug_aug_center.h5".format(flag.upper())
            self.caption_emb_aug_for_model = "./data/caption_emb_{}_aug_rb.h5".format(flag.upper())
            self.dataset_json_for_model = "./data/augmented_{}.json".format(flag.upper())

    def parse(self, kwargs):
        """
        update configuration by kwargs.
        """
        for k, v in kwargs.items():
            if k == 'flag':
                self.data(v)
            if k == 'proc':
                self.proc = v
            if k == 'device':
                self.device = v
            if k == 'bit':
                self.bit = int(v)
            if not hasattr(self, k):
                warnings.warn("Warning: opt has no attribute %s" % k)
            setattr(self, k, v)

        print('Configuration:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__') and str(k) != 'parse' and str(k) != 'data':
                print('\t{0}: {1}'.format(k, getattr(self, k)))


opt = Default()
