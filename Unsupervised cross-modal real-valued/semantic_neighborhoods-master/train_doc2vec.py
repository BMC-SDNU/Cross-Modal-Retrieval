import os, pdb, sys, glob, time, re, pickle, random, gensim, json, unicodedata
from tqdm import tqdm
import numpy as np
import nltk
from multiprocessing import Pool
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_punctuation, strip_multiple_whitespaces, remove_stopwords, strip_short, strip_non_alphanum

# Set up log to terminal
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)
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
def convert_text(text, sentence_limit=2):
    def normalize_text(text):
        text = text.encode('ascii', errors='replace').decode('ascii')
        return text
    text = normalize_text(text)
    # Sentence limit
    sentences = []
    for sentence in nltk.tokenize.sent_tokenize(text)[:sentence_limit]:
        sentences.append(sentence)
    text = ' '.join(sentences)
    CUSTOM_FILTERS = [lambda x: x.lower(), strip_tags, strip_punctuation, strip_multiple_whitespaces, strip_non_alphanum, remove_stopwords, strip_short]
    tokenized_text = list(preprocess_string(text, CUSTOM_FILTERS))
    return tokenized_text

def main():
    dataset = pickle.load(open('db.pickle', 'rb'))
    train_pths = set(pickle.load(open('train_test_paths.pickle', 'rb'))['train'])
    pool = Pool(processes=48)
    all_train_text = [txt for pth, txt in dataset if pth in train_pths and len(txt)>0]
    documents = list(tqdm(pool.imap(convert_text, all_train_text, chunksize=1), total=len(all_train_text)))
    pool.close()
    documents = [d for d in documents if d]
    random.shuffle(documents)
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(documents)]
    print('Starting training on {} documents'.format(len(documents)))
    # WE REPORT USING 20 EPOCHS IN PAPER - BUT USE 50 ON SMALLER DATASETS - MAY NEED ADJUSTING DEPENDING ON DATASET SIZE
    d2v = Doc2Vec(documents=documents, vector_size=200, workers=48, epochs=50, window=20, min_count=20)
    d2v.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
    d2v.save('doc2vec_model.gensim')
    print('Finished training')
if __name__ == '__main__':
    main()
