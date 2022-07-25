import os

MAX_DOC_LENGTH = 500
NUM_CLASSES = 20

#change working directory
cwd = os.getcwd()
cwd = cwd.replace("\\session4", "")
os.chdir(cwd)
print(os.getcwd())

def encode_data(data_path, vocab_path):
    with open(vocab_path) as f:
        vocab = dict([(word, word_ID+2) for word_ID, word in enumerate(f.read().splitlines())])
        
    with open(data_path) as f:
        documents = [tuple(line.split("<fff>")) for line in f.read().splitlines()]
    
    encoded_data = []
    for document in documents:
        label, doc_id, text = document
        words = text.split()[:MAX_DOC_LENGTH]
        sentence_length = len(words)
        
        encoded_text = []
        unknown_ID = 0
        padding_ID = 1
        for word in words:
            if word in vocab:
                encoded_text.append(str(vocab[word]))
            else:
                encoded_text.append(str(unknown_ID))
        if len(words) < MAX_DOC_LENGTH:
            num_padding = MAX_DOC_LENGTH - len(words)
            encoded_text += [str(padding_ID)]*num_padding
        
        encoded_data.append(str(label) + "<fff>" + str(doc_id) + "<fff>" + str(sentence_length) + "<fff>" + " ".join(encoded_text))
    dir_name = "/".join(data_path.split("/")[:-1])
    file_name = "_".join(data_path.split("/")[-1].split("_")[:-1]) + "_encoded.txt"
    with open(dir_name + "/" + file_name, "w") as f:
        f.write("\n".join(encoded_data))

if __name__ == "__main__":
    encode_data(data_path="datasets/w2v/20news_test_raw.txt", vocab_path="datasets/w2v/vocab_raw.txt")