import torch
import mindspore
from mindspore import Parameter, Tensor
from bert_model import BertConfig, BertModel
from vit import get_network, VitArgs


def update_torch_to_ms(torch_param_dict, ms_param_dict, torch_key, ms_key):
    """Updates mindspore param's data from torch param's data."""
 
    value = torch_param_dict[torch_key].cpu().numpy()
    value = Parameter(Tensor(value), name=ms_key)
    ms_param_dict[ms_key].set_data(value)

def update_torch_qkv_to_ms(torch_param_dict, ms_param_dict, torch_key, start, end, ms_key):
    """Updates mindspore param's data from torch param's data."""
    if end == -1:
        value = torch_param_dict[torch_key].cpu().numpy()[start:]
    else:
        value = torch_param_dict[torch_key].cpu().numpy()[start:end]
    value = Parameter(Tensor(value), name=ms_key)
    ms_param_dict[ms_key].set_data(value)

def bert_exe(bert_ms, bert_tc):
    for ms_key in bert_ms.keys():
        ms_key_tmp = ms_key.split('.')
        if ms_key_tmp[0] == 'bert_embedding_lookup':
            update_torch_to_ms(bert_tc, bert_ms, 'embeddings.word_embeddings.weight', ms_key)
        elif ms_key_tmp[0] == 'bert_embedding_postprocessor':
            if ms_key_tmp[1] == "token_type_embedding":
                update_torch_to_ms(bert_tc, bert_ms, 'embeddings.token_type_embeddings.weight', ms_key)
            elif ms_key_tmp[1] == "full_position_embedding":
                update_torch_to_ms(bert_tc, bert_ms, 'embeddings.position_embeddings.weight',
                                    ms_key)
            elif ms_key_tmp[1] =="layernorm":
                if ms_key_tmp[2]=="gamma":
                    update_torch_to_ms(bert_tc, bert_ms, 'embeddings.LayerNorm.weight',
                                        ms_key)
                else:
                    update_torch_to_ms(bert_tc, bert_ms, 'embeddings.LayerNorm.bias',
                                        ms_key)
        elif ms_key_tmp[0] == "bert_encoder":
            if ms_key_tmp[3] == 'attention':
                if ms_key_tmp[4] == 'output':
                    if ms_key_tmp[5] == 'dense':
                        update_torch_to_ms(bert_tc, bert_ms,
                                        'encoder.layer.' + ms_key_tmp[2] + '.attention.output.dense.'+ms_key_tmp[-1],
                                        ms_key)

                    elif ms_key_tmp[5]=='layernorm':
                        if ms_key_tmp[6]=='gamma':
                            update_torch_to_ms(bert_tc, bert_ms,
                                                'encoder.layer.' + ms_key_tmp[2] + '.attention.output.LayerNorm.weight',
                                                ms_key)
                        else:
                            update_torch_to_ms(bert_tc, bert_ms,
                                                'encoder.layer.' + ms_key_tmp[2] + '.attention.output.LayerNorm.bias',
                                                ms_key)
                else:
                    par = ms_key_tmp[5].split('_')[0]
                    update_torch_to_ms(bert_tc, bert_ms, 'encoder.layer.'+ms_key_tmp[2]+'.'+ms_key_tmp[3]+'.'
                                    +'self.'+par+'.'+ms_key_tmp[-1],
                                    ms_key)
                
            elif ms_key_tmp[3] == 'intermediate':
                update_torch_to_ms(bert_tc, bert_ms,
                                    'encoder.layer.' + ms_key_tmp[2] + '.intermediate.dense.'+ms_key_tmp[-1],
                                    ms_key)
            elif ms_key_tmp[3] == 'output':
                if ms_key_tmp[4] == 'dense':
                    update_torch_to_ms(bert_tc, bert_ms,
                                    'encoder.layer.' + ms_key_tmp[2] + '.output.dense.'+ms_key_tmp[-1],
                                    ms_key)
                else:
                    if ms_key_tmp[-1] == 'gamma':
                        update_torch_to_ms(bert_tc, bert_ms,
                                        'encoder.layer.' + ms_key_tmp[2] + '.output.LayerNorm.weight',
                                        ms_key)

                    else:
                        update_torch_to_ms(bert_tc, bert_ms,
                                        'encoder.layer.' + ms_key_tmp[2] + '.output.LayerNorm.bias',
                                        ms_key)

        if ms_key_tmp[0] == 'dense':
            if ms_key_tmp[1] == 'weight':
                update_torch_to_ms(bert_tc, bert_ms,
                                    'pooler.dense.weight',
                                    ms_key)
            else:
                update_torch_to_ms(bert_tc, bert_ms,
                                    'pooler.dense.bias',
                                    ms_key)


def vit_exe(vit_ms, vit_tc):
    for ms_key in vit_ms.keys():
        ms_key_tmp = ms_key.split('.')
        if ms_key_tmp[0] == 'cls':
            update_torch_to_ms(vit_tc, vit_ms, 'cls_token', ms_key)
        elif ms_key_tmp[0] == 'pos_embedding':
            update_torch_to_ms(vit_tc, vit_ms, 'pos_embed', ms_key)
        elif ms_key_tmp[0] == 'stem':
            if ms_key_tmp[-1] == 'weight':
                update_torch_to_ms(vit_tc, vit_ms, 'patch_embed.proj.weight', ms_key)
            elif ms_key_tmp[-1] == 'bias':
                update_torch_to_ms(vit_tc, vit_ms, 'patch_embed.proj.bias', ms_key)
        elif ms_key_tmp[0] == 'body':
            if ms_key_tmp[5] == '0':
                norm = '.norm1' if ms_key_tmp[3]=='0' else '.norm2'
                suffix = '.weight' if ms_key_tmp[-1]=='gamma' else '.bias'
                update_torch_to_ms(vit_tc, vit_ms, 'blocks.'+ms_key_tmp[2]+norm+suffix, ms_key)
            elif ms_key_tmp[6] == 'to_q':
                update_torch_qkv_to_ms(vit_tc, vit_ms, 'blocks.'+ms_key_tmp[2]+'.attn.qkv.'+ms_key_tmp[-1], 0, 768, ms_key)
            elif ms_key_tmp[6] == 'to_k':
                update_torch_qkv_to_ms(vit_tc, vit_ms, 'blocks.'+ms_key_tmp[2]+'.attn.qkv.'+ms_key_tmp[-1], 768,  1536, ms_key)
            elif ms_key_tmp[6] == 'to_v':
                update_torch_qkv_to_ms(vit_tc, vit_ms, 'blocks.'+ms_key_tmp[2]+'.attn.qkv.'+ms_key_tmp[-1], 1536,  -1, ms_key)
            elif ms_key_tmp[6] == 'to_out':
                update_torch_to_ms(vit_tc, vit_ms, 'blocks.'+ms_key_tmp[2]+'.attn.proj.'+ms_key_tmp[-1], ms_key)
            elif ms_key_tmp[6] == 'ff1':
                update_torch_to_ms(vit_tc, vit_ms, 'blocks.'+ms_key_tmp[2]+'.mlp.fc1.'+ms_key_tmp[-1], ms_key)
            elif ms_key_tmp[6] == 'ff2':
                update_torch_to_ms(vit_tc, vit_ms, 'blocks.'+ms_key_tmp[2]+'.mlp.fc2.'+ms_key_tmp[-1], ms_key)
        elif ms_key_tmp[0] == 'norm':
            suffix = '.weight' if ms_key_tmp[-1]=='gamma' else '.bias'
            update_torch_to_ms(vit_tc, vit_ms, 'norm'+suffix, ms_key)


# bert_tc = torch.load('/home/yuyang/works/scahn-ms/pretrained/bert_base_uncased.pth')
# bert = BertModel(BertConfig(), False)
# bert_ms = bert.parameters_dict()
# bert_exe(bert_ms, bert_tc)
# mindspore.save_checkpoint(bert, '/home/yuyang/works/scahn-ms/pretrained/bert_base_uncased.ckpt')

vit_tc = torch.load('/home/yuyang/works/scahn-ms/pretrained/mae_pretrain_vit_base.pth')['model']
vit = get_network('vit_base_patch16', VitArgs(224, 1024))
vit_ms = vit.parameters_dict()
vit_exe(vit_ms, vit_tc)
mindspore.save_checkpoint(vit, '/home/yuyang/works/scahn-ms/pretrained/mae_vit_base_16.ckpt')