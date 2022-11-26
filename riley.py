import numpy as np


def files_exist():
    from os.path import exists as file_exists

    return file_exists('x.json') and file_exists('x_test.json') and file_exists('y.json') and file_exists('y_test.json')


def get_ag_news():
    import json

    if files_exist():
        print("hit cache")
        with open('x.json', 'r') as file:
            x_seq = json.load(file)
        with open('x_test.json', 'r') as file:
            x_test_seq = json.load(file)
        with open('y.json', 'r') as file:
            y = json.load(file)
        with open('y_test.json', 'r') as file:
            y_test = json.load(file)
        print("loaded cache")
        return np.array(x_seq), np.array(y), np.array(x_test_seq), np.array(y_test)

    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    import tensorflow_datasets as tfds

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
    x = x[:12000]
    y = y[:12000]
    x_test = x_test[:760]
    y_test = y_test[:760]
    print("x = " + str(len(x)))
    print("y = " + str(len(y)))
    print("x_test = " + str(len(x_test)))
    print("y_test = " + str(len(y_test)))
    x = [i.decode('utf-8') for i in x]
    x_test = [i.decode('utf-8') for i in x_test]
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(x)
    x_seq = get_sequences(tokenizer, x, get_max_len(x + x_test))
    tokenizer.fit_on_texts(x_test)
    x_test_seq = get_sequences(tokenizer, x_test, get_max_len(x + x_test))
    with open('x.json', 'w') as file:
        json.dump(x_seq.tolist(), file)
    with open('x_test.json', 'w') as file:
        json.dump(x_test_seq.tolist(), file)
    with open('y.json', 'w') as file:
        json.dump(y.tolist(), file)
    with open('y_test.json', 'w') as file:
        json.dump(y_test.tolist(), file)
    return x_seq, y, x_test_seq, y_test


def get_random_indicies(num_indicies, all_indicies):
    import random

    assert num_indicies < all_indicies
    s = set()
    while len(s) < num_indicies:
        s.add(random.randint(0, all_indicies - 1))
    return s


def get_model(sm, m, x, y):
    # active learning time!
    from sklearn.svm import NuSVC

    sampling_method = None
    sampling_model = NuSVC(gamma="auto", probability=True)
    if sm == "margin":
        from sampling_methods.margin_AL import MarginAL

        print("margin time!")
        sampling_method = MarginAL(x, y, 13)
    if sm == "kcentre":
        from sampling_methods.kcenter_greedy import kCenterGreedy

        print("kcentre time!")
        sampling_method = kCenterGreedy(x, None, None)
    # model time!
    model = None
    if m == "svm":
        print("svm time!")
        model = NuSVC(gamma="auto", probability=True)
    if m == "cnn":
        from utils.small_cnn import SmallCNN

        print("cnn time!")
        model = SmallCNN(random_state=13)
    return model, sampling_method, sampling_model


def main(argv):
    argv = ["kcentre", "svm", 10, 200]
    x, y, x_test, y_test = get_ag_news()
    model, sampling_method, sampling_model = get_model(argv[0], argv[1], x, y)
    batches = argv[2]
    indicies = get_random_indicies(argv[3], len(x))
    for b in range(batches):
        print("starting round " + str(b))
        x_part = np.array([x[i] for i in indicies])
        y_part = np.array([y[i] for i in indicies])
        sampling_model.fit(x_part, y_part)
        model.fit(x_part, y_part)
        accuracy = model.score(x_test, y_test)
        print(b, accuracy)
        indicies = indicies.union(
            sampling_method.select_batch(
                model=sampling_model, already_selected=indicies, N=argv[3])
        )


if __name__ == "__main__":
    import sys

    main(sys.argv)
