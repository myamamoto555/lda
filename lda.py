# coding:utf-8
import numpy as np

class LDA("object"):
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

        self.w_dn = []

        for d in docs:
            

    def learning(self):
        for i, d in enumerate(self.docs):
            for j, n in enumerate(d):
                p = []
                for k in range(K):
                    prob = float(self.n[i][k]+self.alpha) * (+self.beta) / (self.n[k] + self.beta * self.V)





if __name__ == '__main__':
    documents = [["a", "a", "b"], ["c", "c", "b"]]
    
