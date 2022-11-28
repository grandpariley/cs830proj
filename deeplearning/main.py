import numpy as np


def files_exist():
    from os.path import exists as file_exists

    return file_exists('x.json') and file_exists('x_test.json') and file_exists('y.json') and file_exists('y_test.json')


def trim(b):
    return b.decode('utf-8').lower().replace("\\", " ")


def get_ag_news():
    import json

    if files_exist():
        print("hit cache")
        with open('x.json', 'r') as file:
            x = json.load(file)
        with open('x_test.json', 'r') as file:
            x_test = json.load(file)
        with open('y.json', 'r') as file:
            y = json.load(file)
        with open('y_test.json', 'r') as file:
            y_test = json.load(file)
        print("loaded cache")
        return np.array(x), np.array(y), np.array(x_test), np.array(y_test)

    import tensorflow_datasets as tfds
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import wordnet as wn
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk import pos_tag
    from sklearn.feature_extraction.text import TfidfVectorizer
    from collections import defaultdict
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('stopwords')

    def lemma(array):
        tag_map = defaultdict(lambda: wn.NOUN)
        tag_map['J'] = wn.ADJ
        tag_map['V'] = wn.VERB
        tag_map['R'] = wn.ADV
        rt = []
        word_net_lemmatizer = WordNetLemmatizer()
        for entry in array:
            final = []
            for word, tag in pos_tag(entry):
                if word not in stopwords.words('english') and word.isalpha():
                    word = word_net_lemmatizer.lemmatize(word, tag_map[tag[0]])
                    final.append(word)
            rt.append(str(final))
        return rt

    (x, y), (x_test, y_test) = tfds.as_numpy(tfds.load(
        'ag_news_subset',
        split=['train', 'test'],
        batch_size=-1,
        as_supervised=True,
    ))
    y = y[:3600].tolist()
    y_test = y_test[:228].tolist()
    x = lemma([word_tokenize(trim(i)) for i in x[:3600]])
    x_test = lemma([word_tokenize(trim(i)) for i in x_test[:228]])
    ## human readable until here ##
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    tfidf_vectorizer.fit(x + x_test)
    x = np.asarray(tfidf_vectorizer.transform(x).todense())
    x_test = np.asarray(tfidf_vectorizer.transform(x_test).todense())

    with open('x.json', 'w') as file:
        json.dump(x.tolist(), file)
    with open('x_test.json', 'w') as file:
        json.dump(x_test.tolist(), file)
    with open('y.json', 'w') as file:
        json.dump(y, file)
    with open('y_test.json', 'w') as file:
        json.dump(y_test, file)
    return np.array(x), np.array(y), np.array(x_test), np.array(y_test)


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
    sampling_model = NuSVC()
    if sm == "margin":
        from activelearning.margin import MarginAL

        print("margin time!")
        sampling_method = MarginAL(x, y, 13)
    if sm == "graph":
        from activelearning.graph import GraphDensitySampler

        print("graph time!")
        sampling_method = GraphDensitySampler(x, y, 13)
    # model time!
    model = None
    # if m == "svm":
    #     from sklearn.svm import NuSVC

    #     print("svm time!")
    #     model = make_pipeline(StandardScaler(with_mean=False), NuSVC())
    if m == "nn":
        from deeplearning.small_nn import SmallNN

        print("nn time!")
        model = SmallNN(random_state=13)
    return model, sampling_method, sampling_model


def main(argv):
    argv = ["margin", "nn", 5, 200, True, True]
    x, y, x_test, y_test = get_ag_news()
    if not argv[5]:
        print("that's all folks!")
        return
    model, sampling_method, sampling_model = get_model(argv[0], argv[1], x, y)
    if not argv[4]:
        print("passive learning time!")
        model.fit(x, y)
        print(model.score(x_test, y_test))
        return
    print("active learning time!")
    batches = argv[2]
    indicies = list(range(6 * len(np.unique(y))))
    for b in range(batches):
        print("starting round " + str(b) + " with " + str(len(indicies)) + " samples")
        x_part = np.array([x[i] for i in indicies])
        y_part = np.array([y[i] for i in indicies])
        sampling_model.fit(x_part, y_part)
        model.fit(x_part, y_part)
        accuracy = model.score(x_test, y_test)
        print(b, accuracy)
        indicies.extend(
            sampling_method.select_batch(
                model=sampling_model, already_selected=np.array(indicies), N=argv[3])
        )


if __name__ == "__main__":
    import sys

    main(sys.argv)
