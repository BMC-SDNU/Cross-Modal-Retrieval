import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.common.dtype as mstype


class BUFPostProcessing(nn.Cell):
    def __init__(self, config):
        super().__init__()
        img_emb_dim = config.img_emb_dim
        self.bs = config.bs
        self.img_seq = config.img_seq

        self.map = nn.SequentialCell(
            nn.Dense(img_emb_dim + 5, img_emb_dim),
            nn.ReLU(),
            nn.Dense(img_emb_dim, img_emb_dim)
        )
        self.transformer_encoder = nn.transformer.TransformerEncoder(config.bs, config.buf_num_layers, img_emb_dim, 2048, config.img_seq, 4, hidden_act='relu')
        self.fc = nn.Dense(img_emb_dim, config.emb_dim)

    def construct(self, visual_feats, boxes):
        ones = ops.Ones()
        cat = ops.Concat(2)
        cast = ops.Cast()

        area = (boxes[:, :, 2] - boxes[:, :, 0]) * (boxes[:, :, 3] - boxes[:, :, 1])
        area = area.expand_dims(2)
        s_infos = cast(cat((boxes, area)), mstype.float32)
        visual_feats = cat((visual_feats, s_infos))
        visual_feats = self.map(visual_feats)

        mask = ones((self.bs, self.img_seq, self.img_seq), mstype.float32)

        visual_feats = self.transformer_encoder(visual_feats, mask)[0]
        out = visual_feats.mean(0)

        out = self.fc(out)

        return out, visual_feats