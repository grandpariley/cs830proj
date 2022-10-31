import sys
from sampling_methods.graph_density import GraphDensitySampler
from sampling_methods.margin_AL import MarginAL
from sampling_methods.sampling_def import SamplingMethod
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import GridSearchCV
import tensorflow_datasets as tfds
from sklearn.svm import SVC
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

def main(argv):
    argv = ["margin", "svm", 3, 1000]
    x, y, x_test, y_test = get_ag_news()
    scaler = StandardScaler().fit(x)
    x = scaler.transform(x)
    x_test = scaler.transform(x_test)
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
        model = SVC(random_state=13, max_iter=1500, probability=True)
        params = {"C": [10.0**(i) for i in range(-4, 5)]}
        model = GridSearchCV(model, params, cv=3)
    if argv[1] == "cnn":
        print("cable news time!")
        model = SmallCNN(random_state=13)
    if model is None:
        return

    # use select batch to loop over x and y with smaller batches
    batches = argv[2]
    indicies = set()
    for b in range(batches):
        x_part = x[sorted(indicies)]
        y_part = y[sorted(indicies)]
        model.fit(x_part, y_part)
        accuracy = model.score(x_test, y_test)
        print(b, accuracy)
        if argv[0] == "margin":
            indicies += sampling_method.select_batch(model=model, already_selected=indicies, N=argv[3])
        if argv[0] == "graph":
            indicies += sampling_method.select_batch(already_selected=indicies, N=argv[3])

if __name__ == "__main__":
    main(sys.argv)