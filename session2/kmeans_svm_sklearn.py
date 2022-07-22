import os
from typing import Tuple
import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
from sklearn.svm import LinearSVC, SVC
from collections import defaultdict

#change working directory
cwd = os.getcwd()
cwd = cwd.replace("\\session2", "")
os.chdir(cwd)
print(os.getcwd())

def load_data(data_path: str) -> Tuple[list]:
    def sparse_to_dense(sparse_r_d: str, vocab_size: int) -> np.array:
        r_d = [0.0 for _ in range(vocab_size)]
        indices_tfidfs = sparse_r_d.split()
        for index_tfidf in indices_tfidfs:
            index, tfidf = map(float, index_tfidf.split(":"))
            index = int(index)
            r_d[index] = tfidf
        return np.array(r_d)
    
    with open(data_path) as f:
        d_lines = f.read().splitlines()
    with open("datasets/20news_bydate/words_idfs.txt") as f:
        vocab_size = len(f.read().splitlines())
    
    data = []
    label_count = defaultdict(int)
    labels = []
    for data_id, d in enumerate(d_lines):
        features = d.split("<fff>")
        label, doc_id = map(int, features[:2])
        label_count[label] += 1
        r_d = sparse_to_dense(sparse_r_d=features[2], vocab_size=vocab_size)
        data.append(r_d)
        labels.append(label)
    return data, labels

def compute_purity(labels, predicted_labels):
    majority_sum = 0
    clusters = [[] for i in range(20)]
    for index, predicted_label in enumerate(predicted_labels):
        clusters[predicted_label].append(labels[index])
        
    for cluster in clusters:
        member_labels = cluster
        max_count = max([member_labels.count(label) for label in range(20)])
        majority_sum += max_count
    return majority_sum / len(predicted_labels)

def clustering_with_Kmeans():
    data, labels = load_data("datasets/20news_bydate/data_tf_idf.txt")
    
    X = csr_matrix(data)
    print("=======")
    kmeans = KMeans(n_clusters=20, init="k-means++", n_init=5, tol=1e-3, random_state=2018).fit(X)
    print(compute_purity(labels=labels, predicted_labels=kmeans.labels_))
    labels = kmeans.labels_

def classifying_with_linear_SVMs():
    train_X, train_y = load_data(data_path = "datasets/20news_bydate/20news_train_tfidf.txt")
    classifier = LinearSVC(C=10.0, tol=0.001, verbose=True)
    classifier.fit(train_X, train_y)
    print(compute_accuracy(predicted_y=classifier.predict(train_X), expected_y=train_y))
    test_X, test_y = load_data(data_path = "datasets/20news_bydate/20news_test_tfidf.txt")
    predicted_y = classifier.predict(test_X)
    accuracy = compute_accuracy(predicted_y=predicted_y, expected_y=test_y)
    print("Accuracy:", accuracy)

def classifying_with_SVMs():
    train_X, train_y = load_data(data_path = "datasets/20news_bydate/20news_train_tfidf.txt")
    classifier = SVC(C=10.0, kernel="rbf", gamma=0.1, tol=0.001, verbose=True)
    classifier.fit(train_X, train_y)
    print(compute_accuracy(predicted_y=classifier.predict(train_X), expected_y=train_y))
    test_X, test_y = load_data(data_path = "datasets/20news_bydate/20news_test_tfidf.txt")
    predicted_y = classifier.predict(test_X)
    accuracy = compute_accuracy(predicted_y=predicted_y, expected_y=test_y)
    print("Accuracy:", accuracy)
    
def compute_accuracy(predicted_y, expected_y):
    return np.sum(predicted_y == expected_y)/len(predicted_y)

#classifying_with_linear_SVMs()
clustering_with_Kmeans()