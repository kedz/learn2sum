#cython: boundscheck=False
#cython: wraparound=False
from cpyvw cimport SearchTask
import numpy as np
cimport numpy as np
import pandas as pd
from nltk.util import ngrams
from collections import defaultdict
import os
import pyvw
import pylibvw
from itertools import izip
from scipy.sparse import vstack

duc03_model_path = os.path.join(
    os.getenv("DUC_DATA"),
    "detagged.duc2003.abstracts/models/")

INT_DTYPE = np.int
DBL_DTYPE = np.double



cdef class L2SSum(SearchTask):
    def __cinit__(self, vw, sch, num_actions):
        SearchTask.__init__(self, vw , sch, num_actions)
        sch.set_options(sch.IS_LDF)


    def compute_oracle_scores(self, model_ngrams, model_Z, 
            summary_ngrams, summary_Z, input_ngrams, input_Z):
        def score_ngrams(item):
            (input_ng, input_z) = item
            score = 0
            for ng in set(input_ng.keys() + summary_ngrams.keys()):
                score += min(
                    model_ngrams[ng], input_ng[ng] + summary_ngrams[ng])
            if score > 0:
                rec = score / float(model_Z)
                prec = score / float(summary_Z + input_z)
                return 2 * (rec * prec) / (rec + prec)
            else:
                return 0
        scores = map(score_ngrams, zip(input_ngrams, input_Z))
        return scores

    cdef void update_examples(L2SSum self, object examples, int sim_start, 
            np.ndarray[DBL_DTYPE_t, ndim=2] Kinp_tf, 
            object summary_i, object index, 
            np.ndarray[DBL_DTYPE_t, ndim=2] Xinp_sf):

        cdef np.ndarray[DBL_DTYPE_t, ndim=2] K_inp_x_sum
        cdef np.ndarray[DBL_DTYPE_t, ndim=2] X
        cdef np.ndarray[DBL_DTYPE_t, ndim=1] max_tf_sims
        cdef np.ndarray[DBL_DTYPE_t, ndim=1] mean_tf_sims
        cdef np.ndarray[DBL_DTYPE_t, ndim=2] X_int_max_tf_sims
        cdef np.ndarray[DBL_DTYPE_t, ndim=2] X_int_mean_tf_sims
       
        cdef int i, j

        if len(summary_i) > 0:
            K_inp_x_sum = Kinp_tf[:,summary_i][index,:]
            max_tf_sims = K_inp_x_sum.max(axis=1)
            mean_tf_sims = K_inp_x_sum.mean(axis=1)
        else:
            max_tf_sims =  np.zeros(len(index), dtype=DBL_DTYPE)
            mean_tf_sims = max_tf_sims
        X = Xinp_sf[index,:]
        X_int_max_tf_sims = X * max_tf_sims[:, np.newaxis]
        X_int_mean_tf_sims = X * mean_tf_sims[:, np.newaxis]
        
        for i in range(len(index)):
            examples[i].pop_namespace()     
            examples[i].push_namespace('b')     
            if max_tf_sims[i] > 0.:
                examples[i].push_features(
                    'b', [(sim_start, max_tf_sims[i]), (sim_start + 1, 0)])
            else:
                examples[i].push_features(
                    'b', [(sim_start, 0.), (sim_start + 1, 1.)])
            if mean_tf_sims[i] > 0.:
                examples[i].push_features(
                    'b', [(sim_start + 2, mean_tf_sims[i]), (sim_start+3, 0)])
            else:
                examples[i].push_features(
                    'b', [(sim_start+2, 0.), (sim_start + 3, 1.)])
            offset = sim_start + 4 - 1
            interactions = [(offset + j, X_int_max_tf_sims[i,j])
                            for j in range(1, sim_start)]
            offset = sim_start + 4 -1 + X.shape[1] - 1
            interactions.extend(
                [(offset + j, X_int_mean_tf_sims[i,j]) 
                 for j in range(1, sim_start)])
            examples[i].push_features('b', interactions)

    cdef object _run(self, object instance):

        (model_ngrams, Z_recall, input_ngrams, input_Z, 
         inputs, Kinp_tf, Xinp_sf, examples, fi) = instance

        oracle_I = []

        cdef int summary_length = 0
        summary_ngrams = defaultdict(int)
        cdef double summary_Z = 0.0
        input_ngrams = list(input_ngrams)
        input_Z = list(input_Z)
        examples = list(examples)
        index = inputs.index.tolist()
         
        word_lengths = inputs["word length"]

        cdef list summary_i = []

        cdef int n = 0
        cdef int action
        cdef int action_i
        cdef double action_score, score, loss, count
        while summary_length < 100:
            n+=1
            scores = self.compute_oracle_scores(model_ngrams, Z_recall, 
                    summary_ngrams, summary_Z, input_ngrams, input_Z)
            oracle_score = max(scores)
            oracle = scores.index(oracle_score)
            oracle_i = index[oracle]
            
            if self.sch.predict_needs_example():
                self.update_examples(
                    examples, fi.sim_start, Kinp_tf, summary_i, index,
                    Xinp_sf)
                
                #examples = self.make_examples(
                #    scores, inputs.iloc[index], cols, Kinp_tf, summary_i, index)
            #else:
            #    examples = [None] * len(index)

            action = self.sch.predict(
                examples=examples, my_tag=n, oracle=oracle, condition=[])
            if self.sch.predict_needs_example():
                oracle_I.append(oracle_i)
            
            action_score = scores[action]
            action_i = index[action]

            summary_length += word_lengths[action_i]
            summary_i.append(action_i)
            summary_Z ++ input_Z[action]

            for ng, count in input_ngrams[action].items():
                summary_ngrams[ng] += count
            del index[action]
            del input_ngrams[action]
            del input_Z[action]
            del examples[action]

        score = 0 
        Z_prec = 0.
        for ng, count in summary_ngrams.items():
            score += min(model_ngrams[ng], count)
            Z_prec += count
        prec = score / Z_prec if Z_prec > 0 else 0
        recall = score / Z_recall    
        cdef double f1 
        if prec == 0 or recall == 0:
            f1 = 0.
        else:
            f1 = 2 * (prec * recall) / (prec + recall)
        #score = score / Z 
        #loss = 1 - score + len(summary_i)


        #run 2
        #f1 loss

        # run 3 .9 f1 - .1 len

        # run 4
        loss = .5 * (1. - f1) + .5 * (Kinp_tf[:, summary_i][summary_i,:]).mean()

        self.sch.loss(loss)

        if self.sch.predict_needs_example():
            print inputs.iloc[summary_i][["doc id", "sent id", "text"]]
            print "oracle"
            print inputs.iloc[oracle_I][["doc id", "sent id", "text"]]
            print loss, "f1",f1, "prec", prec, "recall", recall

        return summary_i

def main():
    converters = {"lemmas": eval}
    with open("duc2003.models.tsv", "rb") as f:
        all_models = pd.read_csv(f, sep="\t", converters=converters)
    docset_ids = set(all_models["docset id"].tolist())

    with open("duc2003.input.feats.tsv", "rb") as f:
        all_inputs = pd.read_csv(f, sep="\t", converters=converters)

    output_dir = "system-runs4"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    instances = make_instances(docset_ids, all_models, all_inputs)
    vw = pyvw.vw("--search 0 --csoaa_ldf m --search_task hook " \
                 "--ring_size 1024 " \
                 "--search_no_caching --quiet" )

    task = vw.init_search_task(L2SSum)

    for num_iter in range(1,11):
        
        task.learn(instances)

        for instance in instances:
            
            model_ngrams, Z, input_ngrams, inputs = instance
            summary_i = task.predict(instance)
            docset_id = inputs["docset id"].tolist()[0]
            system_path = os.path.join(output_dir, "iter{}_{}".format(
                num_iter, docset_id))
            with open(system_path, "w") as f:
                f.write("\n".join(inputs.iloc[summary_i]["text"].tolist()))


def make_instances(docset_ids, all_models, all_inputs):
    instances = []
    for docset_id in docset_ids:
        print docset_id
        models = all_models.loc[all_models["docset id"] == docset_id]
        sents = models["lemmas"].tolist()
        model_ngrams = defaultdict(int)
        for lemmas in sents:
            for ng in ngrams(lemmas, 1):
                model_ngrams[ng] = 1
        Z = float(sum(count for count in model_ngrams.values()))

        inputs = all_inputs.loc[all_inputs["docset id"] == docset_id]
        inputs = inputs.reset_index(drop=True)

        input_ngrams = []
        for lemmas in inputs["lemmas"].tolist():
            ingrams = defaultdict(int)
            for ng in ngrams(lemmas, 1):
                if ng in model_ngrams:
                    ingrams[ng] += 1
            input_ngrams.append(ingrams)
        instances.append((model_ngrams, Z, input_ngrams, inputs))
    return instances
        

