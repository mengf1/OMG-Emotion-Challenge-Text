import csv
import re
import numpy as np


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()

def process_transcripts(text_file, video_file):
    trans = {}
    with open(text_file) as f:
        f_csv = csv.reader(f)
        headers = next(f_csv)
        for row in f_csv:
            video = row[1]
            utterance = row[2]
            transcript = row[3]
            trans[(video, utterance)] = transcript
    data = []
    with open(video_file) as f:
        f_csv = csv.reader(f)
        headers = next(f_csv)
        for row in f_csv:
            video = row[3]
            utterance = row[4]
            arousal = row[5]
            valence = row[6]
            transcript = trans[(video, utterance)]
            data.append([video, utterance, clean_str(transcript), arousal, valence])
    return data

# corpus
class Corpus:
    def __init__(self, name):
        self.name = name
        self.word2index = {"<unk>": 0}
        self.word2count = {}
        self.index2word = {0: "<unk>"}
        self.n_words = 1  # Count unk

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

corpus = Corpus("omg")

def read_data_omg(data_set):
    print("Reading omg ...")
    X1 = []
    Y1 = []
    for line in data_set:
        l = line[2]
        corpus.add_sentence(l)
        X1.append(l)
        Y1.append([float(line[3]), float(line[4])])
    X2 = []
    Y2 = Y1
    for line in X1:
        x_i = []
        for item in line.split():
            x_i.append(corpus.word2index[item])
        X2.append(x_i)
    return X2, Y2, corpus

# file = "/path/to/glove.6B.100d.txt"
def load_embeddings_from_glove(emb_file, word2index, emb_size = 100):
    # Initialise embeddings to random values
    #emb_size = 100
    vocab_size = len(word2index)
    sd = 1/np.sqrt(emb_size)  # standard deviation to use
    w2v = np.random.normal(0, scale=sd, size=[vocab_size, emb_size])
    w2v = w2v.astype(np.float32)

    # Extract desired glove word vectors from a text file
    #with open(emb_file, encoding="utf-8", mode="r") as text_file:
    with open(emb_file, mode="r") as text_file:
        for line in text_file:
            # Separate the values from the word
            line = line.split()
            word = line[0]

            # If word is in our vocab, then update the corresponding values
            index = word2index.get(word, None)
            if index is not None:
                w2v[index] = np.array(line[1:], dtype=np.float32)

    return w2v

