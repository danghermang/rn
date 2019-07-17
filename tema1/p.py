import _pickle as cPickle
import gzip
import numpy

f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f, encoding='latin1')


def activation(input):
    if input > 0:
        return 1
    else:
        return 0

def set_perceptroni():
    perceptroni = []
    for cifra in range(0, 10):
        perceptron_cifra = learn(train_set, cifra)
        perceptroni += [perceptron_cifra]
    return perceptroni

def learn(train_set, cifra):
    learning_rate = 0.1
    iterations = 1
    classified = False
    b = 0
    w = numpy.random.uniform(0, 1, 784)
    while not classified and iterations >= 0:
        classified = True
        for i in range(0, 50000):
            x = train_set[0][i]
            target = train_set[1][i]
            if target == cifra:
                target = 1
            else:
                target = 0
            z = numpy.add(numpy.dot(w, x), b)
            output = activation(z)
            actual_x = numpy.dot(x, (target - output))
            learn_x = numpy.dot(actual_x, learning_rate)
            w = numpy.add(w, learn_x)
            b = b + (target - output) * learning_rate
            if output != target:
                classified = False
        iterations -= 1
    return (w, b)


def test_values(test_set):
    corecte = numpy.zeros((10,), dtype=numpy.int)
    total = numpy.zeros((10,), dtype=numpy.int)
    total_all = 0
    perceptroni = set_perceptroni()
    for i in range(0, len(test_set[0])):
        x = test_set[0][i]
        t = test_set[1][i]
        perceptron = perceptroni[t]
        w = perceptron[0]
        b = perceptron[1]
        z = numpy.add(numpy.dot(w, x), b)
        output = activation(z)
        if output == 1:
            corecte[t] += 1
            total_all += 1
        total[t] += 1
    for i in range(0, len(corecte)):
        print(str(i) + ': ' + str(corecte[i] / float(total[i]) * 100))
    print('\nTotal corecte: ' + str(total_all / float(len(test_set[0])) * 100))

if __name__ == "__main__":
    test_values(test_set)
