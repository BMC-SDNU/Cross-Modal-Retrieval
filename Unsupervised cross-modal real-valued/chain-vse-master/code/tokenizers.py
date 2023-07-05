import torch
import pickle
import nltk
from preprocessing import Text2Char


class TokenizerBase(object):

    def __init__(self):     
        pass

    def tokenize_text(self, text):
        return text

    def tokenize_texts(self, texts):
        return texts


class WordTokenizer(TokenizerBase):

    def __init__(self, vocab_path):
        
        super(WordTokenizer, self).__init__()
        
        self.vocab = pickle.load(open(vocab_path, 'rb'))
        self.vocab_size = len(self.vocab)

    def tokenize_text(self, text):
        vocab = self.vocab

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(
            str(text).lower().decode('utf-8'))
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return target

    def tokenize_texts(self, texts):
        return texts


class CharacterTokenizer(TokenizerBase):

    def __init__(self):
        
        super(CharacterTokenizer, self).__init__()
        self.encoder = Text2Char()        

    def tokenize_text(self, text):
        caption = self.encoder.encode_text(text)
        target = torch.LongTensor(caption)
        return target

    def tokenize_texts(self, texts):
        return texts

