import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

def main(trials):
    results = {
        'classic': {
            'graph': [],
            'margin': []
        },
        'deep': {
            'graph': [],
            'margin': []
        }
    }
    for i in range(trials):
        t = str(i)
        with open('classiclearning/results-graph-' + t + '.json', 'r') as f:
            results['classic']['graph'] += json.load(f)
        with open('classiclearning/results-margin-' + t + '.json', 'r') as f:
            results['classic']['margin'] += json.load(f)
        with open('deeplearning/results-graph-' + t + '.json', 'r') as f:
            results['deep']['graph'] += json.load(f)
        with open('deeplearning/results-margin-' + t + '.json', 'r') as f:
            results['deep']['margin'] += json.load(f)

    def graph(index, x, y, title):
        plt.subplot(2, 2, index)
        plt.scatter(x, y)
        plt.title(title)
        model = LinearRegression()
        X = np.array(x).reshape(-1, 1)
        model.fit(X, y)
        y_line = model.predict(X)
        plt.plot(x, y_line, 'r')
        plt.ylim(0.00, 1.00)
        plt.yticks([i/10.0 for i in range(0, 10)],
                   [str(i/10.0) for i in range(0, 10)])
        plt.grid()
        plt.xlabel('Batch')
        plt.ylabel('Accuracy')

    x = [r['round'] for r in results['classic']['graph']]
    y = [r['accuracy'] for r in results['classic']['graph']]
    graph(1, x, y, 'SVM using \nGraph Density Active Learning')
    x = [r['round'] for r in results['classic']['margin']]
    y = [r['accuracy'] for r in results['classic']['margin']]
    graph(2, x, y, 'SVM using \nMargin Active Learning')
    x = [r['round'] for r in results['deep']['graph']]
    y = [r['accuracy'] for r in results['deep']['graph']]
    graph(3, x, y, 'Neural Network using \nGraph Density Active Learning')
    x = [r['round'] for r in results['deep']['margin']]
    y = [r['accuracy'] for r in results['deep']['margin']]
    graph(4, x, y, 'Neural Network using \nMargin Active Learning')
    plt.style.use('seaborn-poster')
    plt.tight_layout()
    plt.savefig('results.pdf', orientation='landscape')
    plt.show()

if __name__ == "__main__":
    main(10)