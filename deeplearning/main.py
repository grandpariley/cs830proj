import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import nlp
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.svm import NuSVC
import json
import time

def main(argv):
    (x, y), (_, _) = tfds.as_numpy(tfds.load(
        'ag_news_subset',
        split=['train', 'test'],
        batch_size=-1,
        as_supervised=True,
    ))

    def trim(b):
        return b.decode('utf-8').lower().replace("\\", " ")

    x = [trim(i) for i in x]

    new_dataset = nlp.dataset_dict.DatasetDict({
        'train': {
            'text': x[:10000],
            'label': y[:10000]
        },
        'validation': {
            'text': x[10000:11000],
            'label': y[10000:11000]
        },
        'test': {
            'text': x[11000:12000],
            'label': y[11000:12000]
        }
    })

    def get_sampling(sm, train_description, train_label):
        sampling_method = None
        sampling_model = NuSVC(probability=True)
        if sm == "margin":
            from deeplearning.activelearning.margin import MarginAL

            sampling_method = MarginAL(train_description, train_label, int(time.time()))
        if sm == "graph":
            from deeplearning.activelearning.graph import GraphDensitySampler

            sampling_method = GraphDensitySampler(
                train_description, train_label, int(time.time()))
        return sampling_method, sampling_model

    train = new_dataset['train']
    val = new_dataset['validation']
    test = new_dataset['test']

    def get_description(data):
        descriptions = data['text']
        labels = data['label']
        return descriptions, labels

    descriptions, labels = get_description(train)

    tokenizer = Tokenizer(num_words=10000, oov_token='<UNK>')
    tokenizer.fit_on_texts(descriptions)

    maxlen = 50

    def get_sequences(tokenizer, descriptions):
        sequences = tokenizer.texts_to_sequences(descriptions)
        padded = pad_sequences(sequences, truncating='post',
                               padding='post', maxlen=maxlen)
        return padded

    classes = set(labels)

    class_to_index = dict((c, i) for i, c in enumerate(classes))
    index_to_class = dict((v, k) for k, v in class_to_index.items())

    def names_to_ids(labels): return np.array(
        [class_to_index.get(l) for l in labels])

    model = tf.keras.models.Sequential([tf.keras.layers.Embedding(10000, 16, input_length=maxlen), tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
        20, return_sequences=True)), tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20)), tf.keras.layers.Dense(4, activation='softmax')])
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    val_descriptions, val_labels = get_description(val)
    val_seq = get_sequences(tokenizer, val_descriptions)
    val_labels = names_to_ids(val_labels)
    results = []
    indicies = list(range(25 * len(np.unique(y))))
    train_labels = names_to_ids(labels)
    padded_train_seq = get_sequences(tokenizer, descriptions)
    sampling_method, sampling_model = get_sampling(
        argv[0], padded_train_seq, train_labels)
    sampling_model.fit(padded_train_seq, train_labels)
    for b in range(10):
        model.fit(padded_train_seq[indicies], train_labels[indicies], validation_data=(val_seq, val_labels),
                      epochs=20, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=2)])

        test_descriptions, test_labels = get_description(test)
        test_seq = get_sequences(tokenizer, test_descriptions)
        test_labels = names_to_ids(test_labels)
        accuracy = model.evaluate(test_seq, test_labels)[1]
        results.append({'round': len(indicies), 'accuracy': accuracy})
        indicies.extend(
            sampling_method.select_batch(
                model=sampling_model, already_selected=np.array(indicies), N=100)
        )

    with open('deeplearning/results-' + argv[0] + '-' + argv[1] + '.json', 'w') as f:
        json.dump(results, f)
