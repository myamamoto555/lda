# coding:utf-8
import numpy as np

class LDA:
    def __init__(self, alpha, beta, D, K, V, docs):
        self.alpha = alpha
        self.beta = beta
        self.D = D
        self.K = K
        self.V = V
        self.docs = docs

        self.n_k = np.zeros(K)
        self.n_dk = np.zeros((D, K))
        self.n_kv = np.zeros((K, V))

        self.z_dn = []

        self.w_n = []  # 単語のindex
        for d in docs:
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
                
                self.z_dn[i][j] = np.random.multinomial(1, p).argmax()
                
                self.n_dk[i][self.z_dn[i][j]] += 1
                self.n_kv[self.z_dn[i][j]][self.w_n.index(n)] += 1
                self.n_k[self.z_dn[i][j]] += 1




if __name__ == '__main__':
    documents = [["a", "a", "b"], ["c", "c", "b"]]
    lda = LDA(0.5, 0.5, len(documents), 2, 3, documents)
    for i in range(10):
        lda.learning(i)

    print lda.n_dk
    print lda.n_kv
    print lda.n_k
    print lda.z_dn
