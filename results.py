import json
import matplotlib.pyplot as plt

def main():
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
    with open('classiclearning/results-graph.json', 'r') as f:
        results['classic']['graph'] = json.load(f)
    with open('classiclearning/results-margin.json', 'r') as f:
        results['classic']['margin'] = json.load(f)
    with open('deeplearning/results-graph.json', 'r') as f:
        results['deep']['graph'] = json.load(f)
    with open('deeplearning/results-margin.json', 'r') as f:
        results['deep']['margin'] = json.load(f)

    print(results)

    def graph(index, x, y, title): 
        plt.subplot(2, 2, index)
        plt.scatter(x, y)
        plt.title(title)
        plt.xlabel('Batch')
        plt.ylabel('Accuracy')

    x = [r['round'] for r in results['classic']['graph']]
    y = [r['accuracy'] for r in results['classic']['graph']]
    graph(1, x, y, 'SVM using Graph Density Active Learning')
    x = [r['round'] for r in results['classic']['margin']]
    y = [r['accuracy'] for r in results['classic']['margin']]
    graph(2, x, y, 'SVM using Margin Active Learning')
    x = [r['round'] for r in results['deep']['graph']]
    y = [r['accuracy'] for r in results['deep']['graph']]
    graph(3, x, y, 'Neural Network using Graph Density Active Learning')
    x = [r['round'] for r in results['deep']['margin']]
    y = [r['accuracy'] for r in results['deep']['margin']]
    graph(4, x, y, 'Neural Network using Margin Active Learning')

    plt.tight_layout()
    plt.savefig('results.pdf')
    plt.show()