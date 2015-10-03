from cpyvw cimport SearchTask
import pandas as pd
from nltk.util import ngrams
from collections import defaultdict
import os
import pyvw
import pylibvw
from itertools import izip
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import vstack

duc03_model_path = os.path.join(
    os.getenv("DUC_DATA"),
    "detagged.duc2003.abstracts/models/")
def make_counts(lemmas):
    counts = defaultdict(int)
    for lemma in lemmas:
        counts[lemma] += 1
    return counts



cdef class L2SSum(SearchTask):
    def __cinit__(self, vw, sch, num_actions):
        SearchTask.__init__(self, vw , sch, num_actions)
        sch.set_options(sch.IS_LDF)


    def compute_oracle_scores(self, model_ngrams, Z, 
            summary_ngrams, input_ngrams):
        def score_ngrams(input_ng):
            score = 0
            for ng in set(input_ng.keys() + summary_ngrams.keys()):
                score += min(
                    model_ngrams[ng], input_ng[ng] + summary_ngrams[ng])
            return score / float(Z)
        scores = map(score_ngrams, input_ngrams)
        return scores

    cdef object make_examples(L2SSum self, object oscores, object inputs, 
            object cols, object Xsum_tf, object Xinp_tf):

        Ktf = cosine_similarity(Xinp_tf, Xsum_tf)
        max_tf_sims = Ktf.max(axis=1)
        mean_tf_sims = Ktf.mean(axis=1)
        
        X = inputs[cols].values

        cdef list examples = []
        cdef double tf_max, tf_mean;

        for x, tf_max, tf_mean in izip(X, max_tf_sims, mean_tf_sims):
            feats = [(f_i, x_i) if x_i != 0 else (f_i +"==0", 1.) 
                     for f_i, x_i in izip(cols, x)]
            tf_max_feat = \
                ("MAX_TF_SIM", tf_max) if tf_max != 0 else ("MAX_TF_SIM==0",1.)
            tf_mean_feat = \
              ("MEAN_TF_SIM",tf_mean) if tf_mean!=0 else ("MEAN_TF_SIM==0", 1.)

            interactions = [(f_i+"^MAX_TF_SIM", tf_max *x_i)
                            for f_i, x_i in feats]  
            feats += interactions + [tf_max_feat, tf_mean_feat]

            fd = {"a": feats}
            #if self.sch.predict_needs_example():
            #    print fd
            ex = self.vw.example(fd,
                                 labelType=self.vw.lCostSensitive)
            examples.append(ex)

        return examples

    cdef object _run(self, object instance):
        model_ngrams, Z_recall, input_ngrams, input_Z, inputs = instance
        cdef int summary_length = 0
        summary_ngrams = defaultdict(int)
        input_ngrams = list(input_ngrams)
        index = inputs.index.tolist()
         
        vec = DictVectorizer()
        Xinp_tf = inputs["lemmas"].apply(make_counts)
        Xinp_tf = vec.fit_transform(Xinp_tf)
        
        Xsum_tf = None
        word_lengths = inputs["word length"]

        cdef list cols = [c for c in inputs.columns
                if "INPUT" in c or "DOC" in c or "SENT" in c]
        cdef list summary_i = []

        cdef int n = 0
        cdef int action
        cdef int action_i
        cdef double action_score, score, loss, count
        while summary_length < 100:
            n+=1
            scores = self.compute_oracle_scores(model_ngrams, Z_recall, 
                    summary_ngrams, input_ngrams)
            oracle_score = max(scores)
            oracle = scores.index(oracle_score)
            oracle_i = index[oracle]
            
            if self.sch.predict_needs_example():
                examples = self.make_examples(
                    scores, inputs.iloc[index], cols, 
                    Xsum_tf, Xinp_tf[index,:])
            else:
                examples = [None] * len(index)

            action = self.sch.predict(
                examples=examples, my_tag=n, oracle=oracle, condition=[])
            action_score = scores[action]
            action_i = index[action]

            summary_length += word_lengths[action_i]
            summary_i.append(action_i)
            Xsum_tf = Xinp_tf[summary_i, :]
                

            for ng, count in input_ngrams[action].items():
                summary_ngrams[ng] += count
            del index[action]
            del input_ngrams[action]
            

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
        loss = (1. - f1) 
        self.sch.loss(loss)

        if self.sch.predict_needs_example():
            print inputs.iloc[summary_i][["doc id", "sent id", "text"]]
            print loss, "f1",f1, "prec", prec, "recall", recall
            print Xsum_tf.shape
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
        

