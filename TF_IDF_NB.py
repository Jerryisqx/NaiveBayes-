import numpy as np
import scipy.sparse
import time

class NaiveBayes(object):
    def __init__(self, alpha=1e-2):
        # (define parameters)
        self.feature_num = 0
        self.label_num = 0
        self.log_prob = None
        self.label_log_prob = None
        self.alpha = alpha

    def train(self, train_X, train_y):
        # (estimate parameters)
        self.feature_num = train_X.shape[1]
        self.label_num = np.unique(train_y).size
        prob = np.zeros([self.label_num, self.feature_num])
        label_prob = np.zeros(self.label_num)
        for i in range(train_X.shape[0]):
            prob[train_y[i] - 1] += train_X[i]
            label_prob[train_y[i] - 1] += 1
        label_prob /= label_prob.sum()
        for i in range(self.label_num):
            prob[i] = (prob[i] + self.alpha) / (prob[i] + self.alpha).sum()
        self.label_log_prob = np.log(label_prob)
        self.log_prob = np.log(prob)

    def test(self, test_X, test_y):
        # (predict, evaluation, etc.)
        pred_y = np.zeros(test_y.shape[0])
        for i in range(test_X.shape[0]):
            prob = np.dot(test_X[i], self.log_prob.T)
            pred_y[i] = np.argmax(self.label_log_prob + prob) + 1
        accuracy = (pred_y == test_y).sum() / test_y.shape[0] # calculate score
        print('The accuracy in test set: {:.2f}%.'.format(accuracy*100))


def main():
    train_X, test_X = scipy.sparse.load_npz('D:/onedrive/School work/ERG2050/Assignment3/training_feats.npz').toarray(), scipy.sparse.load_npz('D:/onedrive/School work/ERG2050/Assignment3/test_feats.npz').toarray() # DO NOT modify the PATH
    train_y, test_y = np.load('D:/onedrive/School work/ERG2050/Assignment3/training_labels.npy', allow_pickle=True), np.load('D:/onedrive/School work/ERG2050/Assignment3/test_labels.npy', allow_pickle=True) # DO NOT modify the PATH
    nb = NaiveBayes(alpha=0.12)
    nb.train(train_X, train_y)
    nb.test(test_X, test_y)


if __name__ == '__main__':
    t1 = time.time()
    main()
    t2 = time.time()
    print("Running time is: ",t2-t1)