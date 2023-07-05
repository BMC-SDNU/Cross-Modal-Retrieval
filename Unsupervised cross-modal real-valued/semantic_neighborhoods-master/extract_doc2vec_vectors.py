import os, pdb, sys, glob, time, re, pickle, random, gensim, json, unicodedata
from tqdm import tqdm
import numpy as np
import nltk
from multiprocessing import Pool
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_punctuation, strip_multiple_whitespaces, remove_stopwords, strip_short, strip_non_alphanum
from tqdm import tqdm

def get_db():
    db = pickle.load(open('complete_db.pickle', 'rb'))
    orig_paths_and_text = []
    for politics, issues in db.items():
        for issue, items in issues.items():
            for item in items:
                if len(item['content_text']) > 500:
                    orig_paths_and_text.append(
                        (item['local_path'], item['content_text']))
    random.shuffle(orig_paths_and_text)
    return orig_paths_and_text
def process_text(tup, sentence_limit=2):
    def normalize_text(text):
        text = text.encode('ascii', errors='replace').decode('ascii')
        return text
    pth, text = tup
    text = normalize_text(text)
    # Sentence limit
    sentences = []
    for sentence in nltk.tokenize.sent_tokenize(text)[:sentence_limit]:
        sentences.append(sentence)
    text = ' '.join(sentences)
    CUSTOM_FILTERS = [lambda x: x.lower(), strip_tags, strip_punctuation, strip_multiple_whitespaces, strip_non_alphanum, remove_stopwords, strip_short]
    tokenized_text = list(preprocess_string(text, CUSTOM_FILTERS))
    if tokenized_text:
        global d2v
        features = d2v.infer_vector(tokenized_text)
        return (pth, features)
    else:
        return (pth, None)

def main():
    dataset = pickle.load(open('db.pickle', 'rb'))
    train_pths = set(pickle.load(open('train_test_paths.pickle', 'rb'))['train'])
    pool = Pool(processes=48)
    dataset = [(pth, txt) for pth, txt in dataset if pth in train_pths and len(txt)>0]
    print(f'Len deduped dataset = {len(dataset)}')
    global d2v
    d2v = Doc2Vec.load('./doc2vec_model.gensim')
    pool = Pool(processes=48)
    pth_to_features = list(tqdm(pool.imap_unordered(process_text, dataset, chunksize=1), total=len(dataset)))
    pool.close()
    pth_to_features = dict([(pth, features.ravel()) for pth, features in pth_to_features if features is not None])
    pickle.dump(pth_to_features, open('doc2vec_features.pickle', 'wb'))
if __name__ == '__main__':
    main()
