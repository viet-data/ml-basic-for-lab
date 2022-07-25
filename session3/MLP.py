import numpy as np
import tensorflow.compat.v1 as tf
import os

#change working directory
cwd = os.getcwd()
cwd = cwd.replace("\\session3", "")
os.chdir(cwd)
print(os.getcwd())

class MLP:
    NUM_CLASSES = 20
    def __init__(self, vocab_size: int, hidden_size: int):
        self._vocab_size = vocab_size
        self._hidden_size = hidden_size
    
    def build_graph(self):
        tf.disable_eager_execution()
        self._X = tf.placeholder(tf.float32, shape=[None, self._vocab_size])
        self._real_y = tf.placeholder(tf.int32, shape=[None, ])
    
        weights_1 = tf.get_variable(
            name = "weights1_input_hidden",
            shape=(self._vocab_size, self._hidden_size),
            initializer=tf.random_normal_initializer(seed=2018)
        )
        biases_1 = tf.get_variable(
            name="biases1_input_hidden",
            shape=(self._hidden_size),
            initializer=tf.random_normal_initializer(seed=2018)
        )
        
        weights_2 = tf.get_variable(
            name="weights2_hidden_output",
            shape=(self._hidden_size, self.NUM_CLASSES),
            initializer=tf.random_normal_initializer(seed=2018)
        )
        
        biases_2 = tf.get_variable(
            name="biases2_input_hidden",
            shape=(self.NUM_CLASSES),
            initializer=tf.random_normal_initializer(seed=2018)
        )
        hidden = tf.matmul(self._X, weights_1) + biases_1
        hidden = tf.sigmoid(hidden)
        logits = tf.matmul(hidden, weights_2) + biases_2
        
        labels_one_hot = tf.one_hot(indices=self._real_y, depth=self.NUM_CLASSES, dtype=tf.float32)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels_one_hot, logits=logits)
        loss = tf.reduce_mean(loss)
        
        probs = tf.nn.softmax(logits)
        predicted_labels = tf.argmax(probs, axis=1)
        
        return predicted_labels, loss
    
    def trainer(self, loss: tf.Tensor, learning_rate: float):
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        return train_op

class DataReader:
    def __init__(self, data_path: str, batch_size: int, vocab_size:int):
        self._batch_size = batch_size
        with open(data_path) as f:
           d_lines = f.read().splitlines()
            
        self._data = []
        self._labels = []
        for data_id, line in enumerate(d_lines):
            vector = [0.0 for _ in range(vocab_size)]
            features = line.split("<fff>")
            label, doc_id = map(int, features[:2])
            tokens = features[2].split()
            for token in tokens:
                index, value = map(float, token.split(":"))
                index = int(index)
                vector[index] = value 
            self._data.append(vector)
            self._labels.append(label)
        self._data = np.array(self._data)
        self._labels = np.array(self._labels)
        
        self._num_epoch = 0
        self._batch_id = 0
    
    def next_batch(self) -> tuple[list]:
        start = self._batch_id*self._batch_size
        end = start + self._batch_size
        self._batch_id += 1
        
        if end + self._batch_size > len(self._data):
            end = len(self._data)
            self._num_epoch += 1
            self._batch_id = 0
            marks = list(range(len(self._data)))
            np.random.seed(2018)
            np.random.shuffle(marks)
            self._data, self._labels = self._data[marks], self._labels[marks]
        return self._data[start: end], self._labels[start: end]
    
def save_parameters(name: str, value: list, epoch: int):
    filename = name.replace(":","-colon-") + "-epoch-{}.txt".format(epoch)
    if len(value.shape) == 1:
        string_form = ",".join([str(number) for number in value])
    else:
        string_form = "\n".join([",".join([str(number) for number in value[row]]) for row in range(value.shape[0])])
    with open("session3/saved_paras/" + filename, "w") as f:
            f.write(string_form)

def restore_parameters(name: str, epoch: int) -> list[int]:
    filename = name.replace(":","-colon-") + "-epoch-{}.txt".format(epoch)
    with open("session3/saved_paras/" + filename) as f:
        lines = f.read().splitlines()
    if len(lines) == 1:
        value = list(map(float, lines[0].split(",")))
    else:
        value = [list(map(float, line.split(","))) for line in lines]
    return value

def load_dataset() -> tuple[DataReader]:
    train_data_reader = DataReader(data_path="datasets/20news_bydate/20news_train_tfidf.txt",
                                   batch_size=50,
                                   vocab_size=vocab_size)
    test_data_reader = DataReader(
        data_path="datasets/20news_bydate/20news_test_tfidf.txt",
        batch_size=50,
        vocab_size=vocab_size)
    return train_data_reader, test_data_reader
    
with open("datasets/20news_bydate/words_idfs.txt") as f:
    vocab_size = len(f.read().splitlines())
if __name__ == "__main__":
    mlp = MLP(vocab_size=vocab_size, hidden_size=50)
    predicted_labels, loss = mlp.build_graph()
    train_op = mlp.trainer(loss=loss, learning_rate=0.1)
    with tf.Session() as sess:
        train_data_reader, test_data_reader = load_dataset()
        step, MAX_STEP = 0, 1000
        
        sess.run(tf.global_variables_initializer())
        while step < MAX_STEP:
            train_data, train_labels = train_data_reader.next_batch()
            plabels_eval, loss_eval, _ = sess.run([predicted_labels, loss, train_op], feed_dict={mlp._X:train_data, mlp._real_y:train_labels})
            step += 1
            print("step: {}, loss: {}".format(step, loss_eval))
        trainable_variables = tf.trainable_variables()
        for variable in trainable_variables:
            save_parameters(name=variable.name, value=variable.eval(), epoch=train_data_reader._num_epoch)
    
    test_data_reader = DataReader(data_path="datasets/20news_bydate/20news_test_tfidf.txt", batch_size=50, vocab_size=vocab_size)
    with tf.Session() as sess:
        epoch = 4
        trainable_variables = tf.trainable_variables()
        for variable in trainable_variables:
            saved_value = restore_parameters(name=variable.name, epoch=train_data_reader._num_epoch)
            assign_op = variable.assign(saved_value)
            sess.run(assign_op)
        
        num_true_preds = 0
        while True:
            test_data, test_labels = test_data_reader.next_batch()
            test_plabes_eval = sess.run(predicted_labels, feed_dict={mlp._X:test_data, mlp._real_y: test_labels})
            matches = np.equal(test_plabes_eval, test_labels)
            num_true_preds += np.sum(matches)
            
            if test_data_reader._batch_id == 0: break
        
        print("Epoch:", epoch)
        print("Accuracy on test data:", num_true_preds/len(test_data_reader._data))