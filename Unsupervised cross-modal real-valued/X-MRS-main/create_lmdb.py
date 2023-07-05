# !/usr/bin/env python
import pickle
import utils
import os
import lmdb
import shutil
from tqdm import *
from transformers import BertTokenizer, BertModel


MAX_NUM_IMG = 5
LANGUAGES = ['en','de-en','ru-en','de','ru','fr']
SAVE = True
SAVE_PATH = './data/loader_data'
ORIGINAL_DATA_PATH = './data/Recipe1M'



############### LOAD VOCABULARY AND TOKENIZER ################
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained('bert-base-multilingual-cased')

vectors = list(model.embeddings.word_embeddings.parameters())[0].detach().numpy()
vocab = list(tokenizer.get_vocab().keys())
wv = {}
for w,v in zip(vocab, vectors):
    wv[w] = v

ingr_vocab = {k: i for i, (k,v) in enumerate(wv.items())}
vocab_ingr = {i: k for i, (k,v) in enumerate(wv.items())}

if SAVE:
    if not os.path.isdir(SAVE_PATH): os.mkdir(SAVE_PATH)
    with open(os.path.join(SAVE_PATH,'vocab.pkl'),'wb') as f:
        pickle.dump(wv,f)
    with open( os.path.join(SAVE_PATH, 'ingr_vocab.pkl'), 'wb') as f:
        pickle.dump(ingr_vocab, f)
    with open( os.path.join(SAVE_PATH, 'vocab_ingr.pkl'), 'wb') as f:
        pickle.dump(vocab_ingr, f)

    

############### LOAD DATA ###############
print('Loading dataset.')
layers = ['layer2'] + ['layer1' if l=='en' else 'layer1'+'_'+l  for l in LANGUAGES] + ['det_ingrs']
dataset = utils.Layer.merge2(layers, ORIGINAL_DATA_PATH)

with open(os.path.join(ORIGINAL_DATA_PATH, 'classes1M.pkl') ,'rb') as f:
    class_dict = pickle.load(f)
    id2class = pickle.load(f)

# read keys from R1M to match 
keys_to_match = []
for p in ['train', 'test', 'val']:
    with open(os.path.join(ORIGINAL_DATA_PATH, p+'_keys.pkl'), 'rb') as f:
        keys_to_match += pickle.load(f)



########## PREPARE LMDB FILES ##########
env = {'train' : [], 'val':[], 'test':[]}
if SAVE:
    if os.path.isdir(os.path.abspath(os.path.join(SAVE_PATH, 'train_lmdb'))):
        shutil.rmtree(os.path.abspath(os.path.join(SAVE_PATH, 'train_lmdb')))
    if os.path.isdir(os.path.abspath(os.path.join(SAVE_PATH, 'val_lmdb'))):
        shutil.rmtree(os.path.abspath(os.path.join(SAVE_PATH, 'val_lmdb')))
    if os.path.isdir(os.path.abspath(os.path.join(SAVE_PATH, 'test_lmdb'))):
        shutil.rmtree(os.path.abspath(os.path.join(SAVE_PATH, 'test_lmdb')))

    env['train'] = lmdb.open(os.path.abspath(os.path.join(SAVE_PATH, 'train_lmdb')),map_size=int(1e11))
    env['val']   = lmdb.open(os.path.abspath(os.path.join(SAVE_PATH, 'val_lmdb')),map_size=int(1e11))
    env['test']  = lmdb.open(os.path.abspath(os.path.join(SAVE_PATH, 'test_lmdb')),map_size=int(1e11))



########## CREATE LMDB FILES ##########
print('Creating LMDBs...')
keys = {'train' : [], 'val':[], 'test':[]}

for i, entry in tqdm(enumerate(dataset), total=len(dataset)):
    if entry['id'] not in keys_to_match:
        continue
    partition = entry['partition']
    imgs = entry.get('images')

    recipe = []
    part_of_recipe = []
    rec_lenghts = []
    for language in LANGUAGES:
        language = '' if language=='en' else '_'+language
        tmp = entry['title'+language]+'. \n\n '
        title = tokenizer.encode(tmp.replace('..','.'), add_special_tokens=False, verbose=False)

        tmp = '. \n '.join([data['text'] for data in entry['ingredients'+language] if data['text']])+'. \n\n '
        ingredients = tokenizer.encode(tmp.replace('..','.'), add_special_tokens=False, verbose=False)

        tmp = '. \n '.join([data['text'] for data in entry['instructions'+language] if data['text']])
        instructions = tokenizer.encode(tmp.replace('..','.'), add_special_tokens=False, verbose=False)

        tmp = [101] + title + ingredients + instructions # combine CLS token with rest of recipe
        tmp = tmp[:512] + [0]*(512-len(tmp)) # Pad or truncate list of tokens to 512
        recipe.append(tmp)

        tmp = [-1] + [1]*len(title) + [2]*len(ingredients) + [3]*len(instructions) #-1: CLS, 1: title, 2: ingredients, 3: instructions, 0: padding
        tmp = tmp[:512] + [0]*(512-len(tmp)) # Pad or truncate list to 512
        part_of_recipe.append(tmp)

    serialized_sample = pickle.dumps( {'recipe':recipe,
                                       'part_of_recipe':part_of_recipe,
                                       'classes':class_dict[entry['id']]+1,
                                       'imgs':imgs[:MAX_NUM_IMG]} )

    if SAVE:
        with env[partition].begin(write=True) as txn:
            txn.put('{}'.format(entry['id']).encode(), serialized_sample)
    keys[partition].append(entry['id'])


if SAVE:
    for k in keys.keys():
        with open(os.path.join(SAVE_PATH, '{}_keys.pkl'.format(k)),'wb') as f:
            pickle.dump(keys[k],f)

print('Training samples: %d - Validation samples: %d - Testing samples: %d' % (len(keys['train']),len(keys['val']),len(keys['test'])))
if SAVE:
    with open( os.path.join(SAVE_PATH, 'files_in_DB.txt'), 'w') as f:
        f.write('Training samples: %d\nValidation samples: %d\nTesting samples: %d\nrecipe array row order: en, de-en, ru-en, de, ru, fr' % (len(keys['train']),len(keys['val']),len(keys['test'])) + '\n')
print('Done!')