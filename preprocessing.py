# coding:utf-8
import os

class DATA:
    def __init__(self, documents_directory_path):
        fis = os.listdir(documents_directory_path)
        self.docs = []
        self.vocs = []
        for fi in fis:
            with open(documents_directory_path+fi) as f:
                doc_ws = []
                for l in f:
                    l = l.split("\n")[0]
                    l = l.split("\r")[0]
                    ws = l.split()
                    for w in ws:
                        doc_ws.append(w)
                        if not w in self.vocs:
                            self.vocs.append(w)
                self.docs.append(doc_ws)

