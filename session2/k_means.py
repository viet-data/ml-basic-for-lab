from typing import List, Union
import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
import os
from collections import defaultdict

#change working directory
cwd = os.getcwd()
cwd = cwd.replace("\\session2", "")
os.chdir(cwd)
print(os.getcwd())

#Member class
class Member:
    def __init__(self, r_d, label=None, doc_id=None):
        self._r_d = r_d
        self._label = label
        self._doc_id = doc_id

#Cluster cluss
class Cluster:
    def __init__(self):
        self._centroid = None
        self._members = []
    
    def reset_members(self):
        self._members = []
    
    def add_member(self, member: Member):
        self._members.append(member)
    
    def set_centroid(self, new_centroid: np.array):
        self._centroid = new_centroid
        
#Kmeans model
class Kmeans:
    def __init__(self, num_clusters: int):
        self._num_clusters = num_clusters
        self._clusters = [Cluster() for _ in range(self._num_clusters)]
        
        self._E = []
        self._S = 0
    
    def load_data(self, data_path: str):
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
        
        self._data = []
        self._label_count = defaultdict(int)
        for data_id, d in enumerate(d_lines):
            features = d.split("<fff>")
            label, doc_id = map(int, features[:2])
            self._label_count[label] += 1
            r_d = sparse_to_dense(sparse_r_d=features[2], vocab_size=vocab_size)
            self._data.append(Member(r_d=r_d, label=label, doc_id=doc_id))
        
    def random_init(self, seed_value: int):
        if self._num_clusters <= 0: return
        np.random.seed = seed_value
        
        data = np.array([member._r_d for member in self._data])
        similarity_mat = data.dot(data.T)
        stds = np.std(similarity_mat, axis=0)
        sorted_indices = stds.argsort()[::-1]
        centroids = []
        
        quantile = np.quantile(similarity_mat, 0.4)
        
        for index in sorted_indices:
            if len(centroids) == self._num_clusters: break
            if len(centroids) == 0:
                centroids.append(index)
                self._clusters[0].set_centroid(data[index])
            check = True
            for centroid in centroids:
                if similarity_mat[index][centroid] > quantile:
                    print(similarity_mat[index][centroid] , quantile)
                    check = False
                    break
            if check == True:
                self._clusters[len(centroids)].set_centroid(data[index])
                centroids.append(index)
        
        
    def compute_similarity(self, member: Member, centroid: np.array) -> float:
        return member._r_d.dot(centroid)
    
    def select_cluster_for(self, member:Member) -> float:
        best_fit_cluster = None
        max_similarity = -1
        for cluster in self._clusters:
            similarity = self.compute_similarity(member=member, centroid=cluster._centroid)
            if similarity > max_similarity:
                best_fit_cluster = cluster
                max_similarity = similarity 
        best_fit_cluster.add_member(member)
        return max_similarity
    
    def update_centroid_of(self, cluster: Cluster):
        member_r_ds = [member._r_d for member in cluster._members]
        aver_r_d = np.mean(member_r_ds, axis=0)
        sqrt_sum_sqr = np.sqrt(np.sum(aver_r_d**2))
        new_centroid = aver_r_d / sqrt_sum_sqr
        cluster.set_centroid(new_centroid=new_centroid)
    
    def stopping_condition(self, criterion: str, threshold: Union[int, float]) -> bool:
        criteria = ["centroid", "similarity", "max_iters"]
        assert criterion in criteria
        if criterion == "max_iters":
            return True if threshold <= self._iteration else False
        elif criterion == "centroid":
            E_new = [cluster._centroid for cluster in self._clusters]
            E_new_minus_E = [centroid for centroid in E_new if centroid not in self._E]
            self._E = E_new_minus_E
            return True if len(E_new_minus_E <= threshold) else False
        else:
            new_S_minis_S = self._new_S - self._S 
            self._S = self._new_S
            return True if new_S_minis_S <= threshold else False
        
    def run(self, seed_value: int, criterion: str, threshold: Union[int, float]):
        self.random_init(seed_value)
        
        #continually update clusters until convergence
        self._iteration = 0
        while True:
            #reset clusters, retain only centroids
            for cluster in self._clusters:
                cluster.reset_members()
            self._new_S = 0
            for member in self._data:
                max_s = self.select_cluster_for(member)
                self._new_S += max_s
            for cluster in self._clusters:
                self.update_centroid_of(cluster)
                
            self._iteration += 1
            print("Purity and similarity: ", self.compute_purity(), self._new_S)
            if self.stopping_condition(criterion, threshold):
                break
    
    def compute_purity(self):
        majority_sum = 0
        for cluster in self._clusters:
            member_labels = [member._label for member in cluster._members]
            max_count = max([member_labels.count(label) for label in range(20)])
            majority_sum += max_count
        return majority_sum / len(self._data)
    
    def compute_NMI(self):
        I_value, H_omega, H_C, N = 0., 0., 0., len(self._data)
        for cluster in self._clusters:
            wk = len(cluster._members)
            H_omega += wk/N *np.log10(wk/N)
            member_labels = [member._label for member in cluster._members]
            
            for label in range(20):
                wk_cj = member_labels.count(label)
                cj = self._label_count[label]
                I_value += (wk_cj / N) * np.log10(N * wk_cj/(wk * cj) + 1e-12)
        
            for label in range(20):
                cj = self._label_count[label]
                H_C += -cj/N * np.log10(cj/N)
        return I_value * 2/(H_omega + H_C)
    
def clustering_with_KMeans():
    pass
    #data, labels = load
model = Kmeans(20)
print("Start loading")
model.load_data("datasets/20news_bydate/data_tf_idf.txt")
print("Finish loading data")
print("Model is running")
model.run(seed_value=1, criterion="max_iters", threshold=100)
print("Model has finished")
print(model.compute_purity())