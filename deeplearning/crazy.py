import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import nlp
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json

dataset = nlp.load_dataset('emotion')
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

# def filterfn(datum): return datum['label'] not in ['sadness', 'anger']


# train = list(filter(filterfn, dataset['train']))
# val = list(filter(filterfn, dataset['validation']))
# test = list(filter(filterfn, dataset['test']))

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
for b in range(5):
    train_labels = names_to_ids(labels)
    padded_train_seq = get_sequences(tokenizer, descriptions)
    h = model.fit(padded_train_seq, train_labels, validation_data=(val_seq, val_labels),
                epochs=20, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=2)])

    test_descriptions, test_labels = get_description(test)
    test_seq = get_sequences(tokenizer, test_descriptions)
    test_labels = names_to_ids(test_labels)
    accuracy = model.evaluate(test_seq, test_labels)[1]
    results.append({'round': b, 'accuracy': accuracy})

print(results)
with open('results.json', 'w') as f:
    json.dump(results, f)
