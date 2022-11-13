import random
import sys
from sampling_methods.graph_density import GraphDensitySampler
from sampling_methods.margin_AL import MarginAL
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import GridSearchCV
import tensorflow_datasets as tfds
import tensorflow as tf
from sklearn.svm import NuSVC
from utils.small_cnn import SmallCNN


def get_ag_news():
    def get_max_len(descriptions):
        return max([len(x) for x in descriptions])

    def get_sequences(tokenizer, descriptions, max_len):
        sequences = tokenizer.texts_to_sequences(descriptions)
        padded = pad_sequences(sequences, truncating='post',
                               padding='post', maxlen=max_len)
        return padded

    (x, y), (x_test, y_test) = tfds.as_numpy(tfds.load(
        'ag_news_subset',
        split=['train', 'test'],
        batch_size=-1,
        as_supervised=True,
    ))

    x = [i.decode('utf-8') for i in x]
    x_test = [i.decode('utf-8') for i in x_test]
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(x)
    max_len = get_max_len(x)
    x_seq = get_sequences(tokenizer, x, max_len)
    x_test_seq = get_sequences(tokenizer, x_test, max_len)
    return x_seq, y, x_test_seq, y_test


def get_random_indicies(num_indicies, all_indicies):
    assert num_indicies < all_indicies
    s = set()
    while len(s) < num_indicies:
        s.add(random.randint(0, all_indicies))
    return s


def main(argv):
    argv = ["margin", "svm", 3, 1000]
    x, y, x_test, y_test = get_ag_news()
    # active learning time!
    sampling_method = None
    if argv[0] == "margin":
        sampling_method = MarginAL(x, y, 13)
    if argv[0] == "graph":
        sampling_method = GraphDensitySampler(x, y, 13)
    # model time!
    model = None
    if argv[1] == "svm":
        print("svm time!")
        model = NuSVC(gamma="auto", probability=True)
    if argv[1] == "cnn":
        print("cable news time!")
        model = SmallCNN(random_state=13)
    if model is None:
        return

    batches = argv[2]
    initial_indicies = get_random_indicies(argv[3], len(x))
    x_part = [x[i] for i in initial_indicies]
    y_part = [y[i] for i in initial_indicies]
    model.fit(x_part, y_part)
    indicies = set()
    for b in range(batches):
        if argv[0] == "margin":
            indicies = indicies.union(
                sampling_method.select_batch(
                    model=model, already_selected=indicies, N=argv[3])
            )
        if argv[0] == "graph":
            indicies = indicies.union(
                sampling_method.select_batch(
                    already_selected=indicies, N=argv[3])
            )
        x_part = [x[i] for i in indicies]
        y_part = [y[i] for i in indicies]
        model.fit(x_part, y_part)
        accuracy = model.score(x_test, y_test)
        print(b, accuracy)


if __name__ == "__main__":
    main(sys.argv)
