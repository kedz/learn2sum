import os
import numpy as np
import re
np.set_printoptions(precision=2,)

model_path = os.path.join(
    os.getenv("DUC_DATA"), "detagged.duc2003.abstracts", "models")
global docset2paths
docset2paths = None

def get_model_paths(docset):
    #global docset2paths 
    paths = []
    did = docset[:-1].upper()
   # if docset2paths is None:
    for fname in os.listdir(model_path):
        if re.search(did + r"\.\w\.100\.\w\.\w\.html", fname):
            paths.append(os.path.join(model_path, fname))
    return sorted(paths)



class FeatureIndexer(object):
    def __init__(self, features, interactions=False):
        self.use_interactions = interactions
        self.f2i = dict()
        self.i2f = list()
        self.orig_features = features
        all_features = ["CONST"]
        for feature in sorted(features):
            all_features.append(feature)
            all_features.append(feature + "==0")
        for i, f in enumerate(all_features):
            self.f2i[f] = i
            self.i2f.append(f)
            assert self.i2f[self.f2i[f]] == f
        i += 1
        self.dim = i 
        self.sim_start = i
        self.f2i["MAX_TF_SIM"] = i
        self.i2f.append("MAX_TF_SIM")
        i += 1
        self.f2i["MAX_TF_SIM==0"] = i
        self.i2f.append("MAX_TF_SIM==0")

        i += 1
        self.f2i["MEAN_TF_SIM"] = i
        self.i2f.append("MEAN_TF_SIM")
        i += 1
        self.f2i["MEAN_TF_SIM==0"] = i
        self.i2f.append("MEAN_TF_SIM==0")
        i += 1
        self.int_start = i
        if interactions:
            for i, feat in enumerate(all_features[1:], i):
                ifeat = feat +"^MAX_TF_SIM"    
                self.f2i[ifeat] = i
                self.i2f.append(ifeat)
                assert self.i2f[self.f2i[ifeat]] == ifeat
            for i, feat in enumerate(all_features[1:], i+1):
                ifeat = feat +"^MEAN_TF_SIM"    
                self.f2i[ifeat] = i
                self.i2f.append(ifeat)
                assert self.i2f[self.f2i[ifeat]] == ifeat

    def make_static_data(self, vw, inputs):
        Xinp = inputs[self.orig_features].values 
        X = np.zeros((Xinp.shape[0], self.dim))
        X[:,0] = 1
        for i in range(Xinp.shape[0]):
            for j, f in enumerate(self.orig_features):
                k = self.f2i[f]
                if Xinp[i,j] != 0:
                    X[i,k] = Xinp[i,j]
                else:
                    X[i,k+1] = 1.
        
        examples = []
        for i in range(X.shape[0]):
            d = {"a": []}
            for j in range(self.sim_start):
                d["a"].append((j, X[i,j]))
            ex = vw.example(d, labelType=vw.lCostSensitive)
            ex.push_namespace("b")
            examples.append(ex)
        return X, examples
