from deeplearning.main import main as deep_learn
from classiclearning.main import main as classic_learn
from results import main as plot


def main(argv):
    trials = 10
    for t in range(trials):
        classic_learn(["margin", "svm", 10, 100, True, True, str(t)])
        classic_learn(["graph", "svm", 10, 100, True, True, str(t)])
        deep_learn(['margin', str(t)])
        deep_learn(['graph', str(t)])
    plot(trials)


if __name__ == "__main__":
    import sys

    main(sys.argv)
