import json
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
with open('classiclearning/results-graph.py', 'r') as f:
    results['classic']['graph'] = json.load(f)
with open('classiclearning/results-margin.py', 'r') as f:
    results['classic']['margin'] = json.load(f)
with open('deeplearning/results-graph.py', 'r') as f:
    results['deep']['graph'] = json.load(f)
with open('deeplearning/results-margin.py', 'r') as f:
    results['deep']['margin'] = json.load(f)

print(results)