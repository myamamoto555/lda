# coding:utf-8
import numpy as np
import preprocessing

class LDA:
    def __init__(self, alpha, beta, K, data):
        self.alpha = alpha
        self.beta = beta
        self.D = len(data.docs)
        self.K = K
        self.V = len(data.vocs)
        self.docs = data.docs

        self.n_k = np.zeros(self.K)
        self.n_dk = np.zeros((self.D, self.K))
        self.n_kv = np.zeros((self.K, self.V))

        self.z_dn = []

        self.w_n = []  # 単語のindex
        for d in self.docs:
            self.z_dn.append(np.zeros(len(d)))
            for n in d:
                if not n in self.w_n:
                    self.w_n.append(n)
            

    def learning(self, epoc):
        for i, d in enumerate(self.docs):
            z_d=[]
            for j, n in enumerate(d):
                if epoc > 0:
                    self.n_dk[i][self.z_dn[i][j]] -= 1
                    self.n_kv[self.z_dn[i][j]][self.w_n.index(n)] -= 1
                    self.n_k[self.z_dn[i][j]] -= 1
                p = []
                for k in range(self.K):
                    prob = float(self.n_dk[i][k]+self.alpha) \
                    * (self.n_kv[k][self.w_n.index(n)]+self.beta) \
                    / (self.n_k[k] + self.beta * self.V)
                    p.append(prob)
                p = np.array(p)
                p /= p.sum()
                
                self.z_dn[i][j] = int(np.random.multinomial(1, p).argmax())
                print self.z_dn[i][j]
                
                self.n_dk[i][self.z_dn[i][j]] += 1
                self.n_kv[self.z_dn[i][j]][self.w_n.index(n)] += 1
                self.n_k[self.z_dn[i][j]] += 1


if __name__ == '__main__':
    K = 10
    data = preprocessing.DATA("./docs/")
    
    lda = LDA(0.5, 0.5, K, data)
    for i in range(10):
        lda.learning(i)

    print lda.n_dk
    print lda.n_kv
    print lda.n_k
    print lda.z_dn
