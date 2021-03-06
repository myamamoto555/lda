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

        self.n_k = np.zeros(self.K) + self.V * self.beta  # word count of topic k
        self.n_dk = np.zeros((self.D, self.K)) + self.alpha  # word count of document d and topic k
        self.n_kv = np.zeros((self.K, self.V)) + self.beta  # word count of topic k and vocabulary v

        self.z_dn = []  # topic of document d and word n
	
	self.init_topic_assign()

    def init_topic_assign(self):
        for i, d in enumerate(self.docs):
            z_n = []
            for n in d:
                p_z = self.n_kv[:, n] * self.n_dk[i] / self.n_k
		z = np.random.multinomial(1, p_z / p_z.sum()).argmax()
		
		z_n.append(z)
		self.n_k[z] += 1
            	self.n_dk[i, z] += 1
		self.n_kv[z, n] += 1
	    self.z_dn.append(np.array(z_n))

    def learning(self):
        for i, d in enumerate(self.docs):
            z_n = self.z_dn[i]
	    n_dk = self.n_dk[i]
            for j, n in enumerate(d):
		# discount
		z = z_n[j]
		n_dk[z] -= 1
		self.n_kv[z, n] -= 1
		self.n_k[z] -= 1
                
		# sampling
		p_z = self.n_kv[:, n] * n_dk / self.n_k
		new_z = np.random.multinomial(1, p_z / p_z.sum()).argmax()
                
		# update
                z_n[j] = new_z
		n_dk[new_z] += 1
		self.n_kv[new_z, n] += 1
		self.n_k[new_z] += 1
    
    def topic_word_dist(self):
	return self.n_kv / self.n_k[:, np.newaxis]
    
    def perplexity(self, docs=None):
	if docs == None:
	    docs = self.docs
	phi = self.topic_word_dist()
	log_per = 0
	N = 0
	Kalpha = self.K * self.alpha
	for i, doc in enumerate(docs):
	    theta = self.n_dk[i] / (len(self.docs[i]) + Kalpha)
	    for n in doc:
		log_per -= np.log(np.inner(phi[:, n], theta))
	    N += len(doc)
        return np.exp(log_per / N)

if __name__ == '__main__':
    K = 100
    print "preprocessing"
    data = preprocessing.DATA("./docs/")
    print "preprocessing done"

    lda = LDA(0.5, 0.5, K, data)
    for i in range(1000):
        lda.learning()
        print lda.perplexity()
