import os
from collections import defaultdict
import regex as re

MAX_DOC_LENGTH = 500
NUM_CLASSES = 20

#change working directory
cwd = os.getcwd()
cwd = cwd.replace("\\session4", "")
os.chdir(cwd)
print(os.getcwd())

def gen_data_and_vocab():
    def collect_data_from(parent_path: str, newsgroup_list: list, word_count: int =None) -> list:
        data = []
        for group_id, newsgroup in enumerate(newsgroup_list):
            dir_path = parent_path + "/" + newsgroup + "/"
            
            files = [(filename, dir_path+filename) for filename in os.listdir(dir_path) if os.path.isfile(dir_path+filename)]
            files.sort()
            label = group_id
            print("Processing: {}-{}".format(group_id, newsgroup))
            for filename, filepath in files:
                with open(filepath) as f:
                    text = f.read().lower()
                    words = re.split("\W+", text)
                    if word_count != None:
                        for word in words:
                            word_count[word] += 1
                    content = " ".join(words)
                    assert len(content.splitlines()) == 1
                    data.append(str(label) + "<fff>" + filename + "<fff>" + content)
        return data
    word_count = defaultdict(int)
    path = "datasets/20news_bydate/"
    parts = [path + dir_name + "/" for dir_name in os.listdir(path) if not os.path.isfile(path + dir_name)]
    
    if "train" in parts[1]:
        parts.reverse()
    train_path, test_path = parts
    newsgroup_list = [newsgroup for newsgroup in os.listdir(train_path)]
    newsgroup_list.sort()
    
    train_data = collect_data_from(parent_path=train_path, newsgroup_list=newsgroup_list, word_count=word_count)
    vocab = [word for word, freq in word_count.items() if freq > 10]
    vocab.sort()
    with open("datasets/w2v/vocab_raw.txt", "w") as f:
        f.write("\n".join(vocab))
    
    test_data = collect_data_from(parent_path=test_path, newsgroup_list=newsgroup_list)
    
    with open("datasets/w2v/20news_train_raw.txt", "w") as f:
        f.write("\n".join(train_data))
    with open("datasets/w2v/20news_test_raw.txt", "w") as f:
        f.write("\n".join(test_data))