from __future__ import print_function
import time
from nltk.tag.perceptron import PerceptronTagger

print('[Init] Loading NLTK tagger')
start_time = time.time()
tagger = PerceptronTagger()
print('[Init] Loaded NLTK in %4.4f' % (time.time() - start_time))


def is_text_relevant(query, retrieved, part_tag='NN'):
    if part_tag is not None:
        query = [w for w, p in tagger.tag(query) if p == part_tag]
        retrieved = [w for w, p in tagger.tag(retrieved) if p == part_tag]
    if type(query) is not set:
        query = set(query)
    if type(retrieved) is not set:
        retrieved = set(retrieved)
    return not query.isdisjoint(retrieved)
