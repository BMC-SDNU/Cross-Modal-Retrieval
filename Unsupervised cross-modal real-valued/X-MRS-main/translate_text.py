# !/usr/bin/env python
import sys
import os
import torch
import utils
import simplejson as json
import torch.nn.parallel
from tqdm import *

ORIGINAL_DATA_PATH = 'data/Recipe1M'
# language = sys.argv[1]
language = 'de'


######### DATALOADER #########
class Loader(torch.utils.data.Dataset):
    def __init__(self, lines):
        self.lines = lines

    def __getitem__(self, index):
        return self.lines[index]

    def __len__(self):
        return len(self.lines)


########## LOAD TRANSLATION MODEL ###########
if language=='de-en':
    model='transformer.wmt19.de-en'
    checkpoint_file='model1.pt'
elif language=='ru-en':
    model='transformer.wmt19.ru-en'
    checkpoint_file='model1.pt'
elif language=='de':
    model='transformer.wmt19.en-de'
    checkpoint_file='model1.pt'
elif language=='ru':
    model='transformer.wmt19.en-ru'
    checkpoint_file='model1.pt'
elif language=='fr':
    model='transformer.wmt14.en-fr'
    checkpoint_file='model.pt'
else:
        raise NotImplementedError
aug = torch.hub.load('pytorch/fairseq', model=model,
                    checkpoint_file=checkpoint_file, tokenzier_name='moses', bpe_name='fastbpe', force_reload=False)
aug.cuda()
aug.eval()



######### READ DATASET SENTENCES ##########
lang_sufix = ''
if '-en' in language: # Back-translation
    lang_sufix = '_'+language.split('-')[0]
dataset = utils.Layer.load('layer1'+lang_sufix, ORIGINAL_DATA_PATH)
lines = []
for i,entry in tqdm(enumerate(dataset), total=len(dataset)):
    lines.append(entry['title'+lang_sufix])
    [lines.append(data['text']) for data in entry['ingredients'+lang_sufix] if data['text']]
    [lines.append(data['text']) for data in entry['instructions'+lang_sufix] if data['text']]




############ TRANSLATE TEXT #############
print('Processing')
print('\tlanguage: {}'.format(language))
data = Loader(lines)
loader = torch.utils.data.DataLoader(data, 
                                    batch_size=500,
                                    shuffle=False,
                                    sampler=torch.utils.data.SequentialSampler(data),
                                    num_workers=0,
                                    drop_last=False,
                                    pin_memory=True)
translation = []
for i, (batch) in tqdm(enumerate(loader), total=len(loader)):
    batch = aug.translate(batch)
    translation += batch


########### REORGANIZE AND SAVE TRANSLATED TEXT ############
out = []
L1 = utils.Layer.load(utils.Layer.L1, ORIGINAL_DATA_PATH)
for i, entry in tqdm(enumerate(L1), total=len(L1)):
    entry['title_'+language] = translation.pop()
    entry['ingredients_'+language] = [{'text':translation.pop()} for t in entry['ingredients']]
    entry['instructions_'+language] = [{'text':translation.pop()} for t in entry['instructions']]
    entry.pop('title')
    entry.pop('ingredients')
    entry.pop('instructions')
    out.append(entry)
with open(os.path.join(ORIGINAL_DATA_PATH,'layer1_{}.json'.format(language)), 'w') as outfile:
    json.dump(out, outfile)
print('Done: {}'.format(language))
