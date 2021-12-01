import pandas as pd
from gensim.models import Word2Vec
import re

import os.path

global model
model = None

def train_extractive_from_dataset():
    if os.path.isfile('custom_word2vec.model'):
        model = Word2Vec.load("custom_word2vec.model")
    else:
        data = pd.read_csv('data.csv')
        tokenized_text_sents = [i.split() for i in data['text']]
        model = Word2Vec(sentences = tokenized_text_sents, vector_size=400, window=5, min_count=1, workers=4)
        model.save("custom_word2vec.model")

def document_vector(model,doc):
    words = model.wv.index_to_key
    doc = [word for word in doc if word in words]
    return np.mean(model.wv[doc], axis=0)
    
def text_strip(row):
    row = re.sub("(\\t)", " ", str(row)).lower()
    row = re.sub("(\\r)", " ", str(row)).lower()
    row = re.sub("(\\n)", " ", str(row)).lower()

    # Remove _ if it occurs more than one time consecutively
    row = re.sub("(__+)", " ", str(row)).lower()

    # Remove - if it occurs more than one time consecutively
    row = re.sub("(--+)", " ", str(row)).lower()

    # Remove ~ if it occurs more than one time consecutively
    row = re.sub("(~~+)", " ", str(row)).lower()

    # Remove + if it occurs more than one time consecutively
    row = re.sub("(\+\++)", " ", str(row)).lower()

    # Remove . if it occurs more than one time consecutively
    row = re.sub("(\.\.+)", " ", str(row)).lower()

    # Remove the characters - <>()|&©ø"',;?~*!
    row = re.sub(r"[<>()|&©ø\[\]\'\",;?~*!]", " ", str(row)).lower()

    # Remove mailto:
    row = re.sub("(mailto:)", " ", str(row)).lower()

    # Remove \x9* in text
    row = re.sub(r"(\\x9\d)", " ", str(row)).lower()

    # Replace INC nums to INC_NUM
    row = re.sub("([iI][nN][cC]\d+)", "INC_NUM", str(row)).lower()

    # Replace CM# and CHG# to CM_NUM
    row = re.sub("([cC][mM]\d+)|([cC][hH][gG]\d+)", "CM_NUM", str(row)).lower()

    # Remove punctuations at the end of a word
#     row = re.sub("(\.\s+)", " ", str(row)).lower()
    row = re.sub("(\-\s+)", " ", str(row)).lower()
    row = re.sub("(\:\s+)", " ", str(row)).lower()

    # Replace any url to only the domain name
    try:
        url = re.search(r"((https*:\/*)([^\/\s]+))(.[^\s]+)", str(row))
        repl_url = url.group(3)
        row = re.sub(r"((https*:\/*)([^\/\s]+))(.[^\s]+)", repl_url, str(row))
    except:
        pass

    # Remove multiple spaces
    row = re.sub("(\s+)", " ", str(row)).lower()

    # Remove the single character hanging between any two spaces
    row = re.sub("(\s+.\s+)", " ", str(row)).lower()
    return row


from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
 
def read_article(file_name):
    file = open(file_name, "r")
    filedata = file.read()
    filedata = text_strip(filedata)
    filedata.replace('\n\n','')
    filedata.replace('\n','')
    article = filedata.split(".")
    sentences = []
    for sentence in article:
        sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
    sentences.pop() 
    
    return sentences

def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
 
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
 
    all_words = list(set(sent1 + sent2))
 
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
 
    # build the vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
 
    # build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
 
    return 1 - cosine_distance(vector1, vector2)

def sentence_similarity_w2v(sent1, sent2, model, stopwords=None):
    vector1 = document_vector(model,sent1)
    vector2 = document_vector(model,sent2)
 
    return 1 - cosine_distance(vector1, vector2)
 
def build_similarity_matrix(sentences, model, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
 
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: #ignore if both are same sentences
                continue 
            similarity_matrix[idx1][idx2] = sentence_similarity_w2v(sentences[idx1], sentences[idx2], model, stop_words)

    return similarity_matrix


def generate_summary(file_name, top_n=5):
    if os.path.isfile('custom_word2vec.model'):
        model = Word2Vec.load("custom_word2vec.model")
    else:
        print('Model files are not found. Hence creating it. This might take some time.')
        train_extractive_from_dataset()
    stop_words = stopwords.words('english')
    summarize_text = []

    # Step 1 - Read text anc split it
    sentences =  read_article(file_name)
    # Step 2 - Generate Similary Martix across sentences
    sentence_similarity_martix = build_similarity_matrix(sentences,model, stop_words)
    # Step 3 - Rank sentences in similarity martix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph)
    # Step 4 - Sort the rank and pick top sentences
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)     

    for i in range(top_n):
        summarize_text.append(" ".join(ranked_sentence[i][1]))

#     # Step 5 - Offcourse, output the summarize texr
#     print("Summarize Text: \n", ". ".join(summarize_text))
    return summarize_text