from deeplearning.main import main as deep_learn
from classiclearning.main import main as classic_learn
from results import main as plot

def main(argv):
    classic_learn(["margin", "svm", 5, 200, True, True])
    classic_learn(["graph", "svm", 5, 200, True, True])
    deep_learn(['margin'])
    deep_learn(['graph'])
    plot()

if __name__ == "__main__":
    import sys

    main(sys.argv)