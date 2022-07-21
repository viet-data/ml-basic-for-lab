import os
from nltk.stem import PorterStemmer
import regex as re
import numpy as np
from collections import defaultdict

cwd = os.getcwd()
cwd = cwd.replace("\\session1", "")
os.chdir(cwd)

def gather_20newsgroups_data():
    path = "datasets/20news_bydate/" 
    dirs = [path + dir_name + '/' for dir_name in os.listdir(path) if not os.path.isfile(path+dir_name)]
    if "train" in dirs[1]: dirs.reverse()
    train_dir, test_dir = dirs
    list_newgroups = [newsgroup for newsgroup in os.listdir(train_dir)]
    list_newgroups.sort()
    
    with open("datasets/20news_bydate/stop_words.txt", encoding="utf-8") as f:
        stop_words = f.read().splitlines()
    
    stemmer = PorterStemmer()
    train_data = collect_data_from(parent_dir=train_dir, newsgroup_list=list_newgroups, 
                                   stemmer=stemmer, stop_words=stop_words)
    test_data = collect_data_from(parent_dir=test_dir, newsgroup_list=list_newgroups,
                                  stemmer=stemmer, stop_words=stop_words)
    full_data = train_data + test_data
    with open("datasets/20news_bydate/20news_train_processes.txt", "w") as f:
        f.write("\n".join(train_data))
        
    with open("datasets/20news_bydate/20news_test_processes.txt", "w") as f:
        f.write("\n".join(test_data))
    
    with open("datasets/20news_bydate/20news_full_processes.txt", "w") as f:
        f.write("\n".join(full_data))

def collect_data_from(parent_dir, newsgroup_list, stemmer, stop_words):
    data = []
    for group_id, newsgroup in enumerate(newsgroup_list):
        label = group_id
        dir_path = parent_dir+"/"+newsgroup + "/"
        files = [(filename, dir_path+filename) for filename in os.listdir(dir_path) if os.path.isfile(dir_path+filename)]
        files.sort()
        for filename, filepath in files:
            with open(filepath) as f:
                text = f.read().lower()
                #remove stop words then stem remaing words
                words = [stemmer.stem(word) for word in re.split("\W+", text) if word not in stop_words]
                #combine remaing words
                content = " ".join(words)
                assert len(content.splitlines()) == 1
                data.append(str(label) + "<fff>" + filename + "<fff>" + content)
    return data


def compute_idf(df, corpus_size):
    assert df > 0
    return np.log10(corpus_size*1.0/df)

def generate_vocabulary(data_path):
    with open(data_path) as f:
        lines = f.read().splitlines()
    doc_count = defaultdict(int)
    corpus_size = len(lines)
    
    for line in lines:
        features = line.split("<fff>")
        text = features[-1]
        words = list(set(text.split()))
        for word in words:
            doc_count[word] += 1
            
    words_idfs = [(word, compute_idf(document_freq, corpus_size))
                  for word, document_freq in zip(doc_count.keys(), doc_count.values())
                  if document_freq > 10 and not word.isdigit()]
    words_idfs.sort(key=lambda token:-token[-1])
    print("Vocaulary size: {}".format(len(words_idfs)))
    with open("datasets/20news_bydate/words_idfs.txt", "w") as f:
        f.write("\n".join([word + "<fff>" + str(idf) for word, idf in words_idfs]))

def get_tf_idf(data_path):
    #get idf values
    with open("datasets/20news_bydate/words_idfs.txt") as f:
        words_idfs = list(map(lambda token: (token[0], float(token[1])), [line.split("<fff>") for line in f.read().splitlines()]))
        idfs = dict(words_idfs)
        word_IDs = dict([(word, index) for index, (word, idf) in enumerate(words_idfs)])
        
    with open(data_path) as f:
        documents = list(map(lambda line: (int(line[0]), int(line[1]), line[2]), [line.split("<fff>") for line in f.read().splitlines()]))
    
    data_tf_idf = []
    for document in documents:
        label, doc_id, text = document
        words = [word for word in text.split() if word in idfs]
        word_set = list(set(words))
        max_term_freq = max([words.count(word) for word in word_set])
        
        words_tfidfs = []
        sum_squares = 0.0
        for word in word_set:
            term_freq = words.count(word)
            tf_idf_value = (term_freq/max_term_freq)*idfs[word]
            words_tfidfs.append((word_IDs[word], tf_idf_value))
            sum_squares += tf_idf_value**2
        
        words_tfidfs_normalized = [str(index) + ":" + str(tf_idf_value/np.sqrt(sum_squares)) for index, tf_idf_value in words_tfidfs]
        
        sparse_rep = " ".join(words_tfidfs_normalized)
        data_tf_idf.append((label, doc_id, sparse_rep))

    with open("datasets/20news_bydate/data_tf_idf.txt", "a") as f:
        for label, doc_id, tf_idf in data_tf_idf:
            f.write(str(label) + "<fff>"+str(doc_id)+"<fff>" + str(tf_idf)+"\n")

get_tf_idf("datasets/20news_bydate/20news_full_processes.txt")