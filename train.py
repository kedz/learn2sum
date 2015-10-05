import os
from cpyvw import SearchTask
import l2sum
from l2sum import FeatureIndexer
from l2sum._learn import L2SSum
import pyvw
import pylibvw
import pandas as pd
from collections import defaultdict
from nltk.util import ngrams
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics.pairwise import cosine_similarity
pd.options.display.float_format = '{:.3f}'.format

def lemma_filter(inputs, lemma_length_cutoff):
    print("Removing input sentences with number of stopword-filtered" \
          " lemmas < {}".format(lemma_length_cutoff))
    old_len = len(inputs)
    inp_select = inputs["lemmas"].apply(len) >= lemma_length_cutoff
    inputs = inputs.loc[inp_select]
    new_len = len(inputs)
    print("Filtered {} sentences. New input size is {}".format(
        old_len - new_len, new_len))
    inputs = inputs.reset_index(drop=True)
    return inputs

def make_model_ngrams(models, ngram_size):
    sents = models["lemmas"].tolist()
    model_ngrams = defaultdict()
    for lemmas in sents:
        for ng in ngrams(lemmas, ngram_size):
            model_ngrams[ng] = 1. + model_ngrams.get(ng, 0.)
    model_Z = float(sum(count for count in model_ngrams.values()))
    return model_ngrams, model_Z

def make_input_ngrams(inputs, ngram_size, model_ngrams):
    input_ngrams = []
    input_Z = []
    for lemmas in inputs["lemmas"].tolist():
        ingrams = defaultdict(int)
        Z = 0.
        for ng in ngrams(lemmas, ngram_size):
            if ng in model_ngrams:
                ingrams[ng] += 1.
            Z += 1.
                
        Z = float(Z)
        input_ngrams.append(ingrams)
        input_Z.append(Z)
    return input_ngrams, input_Z

def make_counts(lemmas):
    counts = defaultdict(int)
    for lemma in lemmas:
        counts[lemma] += 1
    return counts

def make_instances(vw, all_models, all_inputs, ngram_size=1):
    docset_ids = all_models["docset id"].unique()    
    instances = []
    for docset_id in docset_ids:
        print docset_id
        models = all_models.loc[all_models["docset id"] == docset_id]
        model_ngrams, model_Z = make_model_ngrams(models, ngram_size)

        inputs = all_inputs.loc[all_inputs["docset id"] == docset_id]
        inputs = inputs.reset_index(drop=True)
        input_ngrams, input_Z = make_input_ngrams(
                inputs, ngram_size, model_ngrams)

        input_centroid_f = [u'INPUT_TF_CENTROID_SIM',]
        doc_centroid_f = ['DOC_TF_CENTROID_SIM',]

        input_freqsum_f = ['INPUT_FREQSUM_UNI_AMEAN', 
                           'INPUT_FREQSUM_UNI_MAX',
                           'INPUT_FREQSUM_UNI_GMEAN']
        input_freqsum_ne_f = ['INPUT_FREQSUM_NE_PERSON_AMEAN', 
                              'INPUT_FREQSUM_NE_LOCATION_AMEAN',
                              'INPUT_FREQSUM_NE_ORGANIZATION_AMEAN',] 
        doc_freqsum_f = ['DOC_FREQSUM_UNI_AMEAN', 
                         'DOC_FREQSUM_UNI_MAX',
                         'DOC_FREQSUM_UNI_GMEAN',]
        doc_basic_f = ['DOC_POSITION', 'DOC_IS_LEAD',] 
        sent_basic_f = ['SENT_LENGTH', 'SENT_NUM_PRON', 'SENT_NUM_QUOT'] 

        features = input_centroid_f + doc_centroid_f + \
                input_freqsum_f + input_freqsum_ne_f + doc_freqsum_f + \
                doc_basic_f + sent_basic_f

        fi = FeatureIndexer(features)
        Xinp_sf, examples = fi.make_static_data(vw, inputs)
        tf_vec = DictVectorizer()
        Xinp_tf = inputs["lemmas"].apply(make_counts)
        Xinp_tf = tf_vec.fit_transform(Xinp_tf)
        Kinp_tf = cosine_similarity(Xinp_tf)
        instances.append(
                (model_ngrams, model_Z, input_ngrams, input_Z,
                 inputs, Kinp_tf, Xinp_sf, examples, fi))

    fi = instances[0][-1]
    print "FEATURES"
    print "========"
    feats = sorted(fi.f2i.items(), key=lambda x: x[1])
    for f, i in feats:
        if i == fi.sim_start:
            print "SIM FEATS"
            print "========="
        elif i == fi.int_start:
            print "INT FEATS"
            print "========="
        print i, f

    return instances
 

def write_weights(output_dir, num_iter, vw, fi):
    wp = os.path.join(output_dir, "weights.iter{}.tsv".format(num_iter))
    with open(wp, "w") as f:
        for i, feat in enumerate(fi.i2f):
            f.write("{}\t{}\t{}\n".format(i, feat, vw.get_weight(i)))


def write_output(summary_indices, num_iter, instance, output_dir):
    (model_ngrams, Z_recall, input_ngrams, input_Z, 
     inputs, Kinp_tf, Xinp_sf, examples, fi) = instance
    docset_id = inputs["docset id"].unique()[0]
    path = os.path.join(output_dir, 
        "{}.iter{}.spl".format(docset_id, num_iter))
    with open(path, "w") as f:
        f.write("\n".join(inputs.iloc[summary_indices]["text"].tolist()))
    return docset_id, path    

def write_eval(output_paths, num_iter, output_dir):
    eval_path = os.path.join(
        output_dir, "system.eval.iter{}".format(num_iter))
    with open(eval_path, "w") as f:
        for docset, output_path in output_paths:
            model_paths = l2sum.get_model_paths(docset)
            f.write(" ".join(model_paths + [output_path]) + "\n")
#    model_dir = "models"
#    all_model_paths = [os.path.join(model_dir, path)
#                       for path in os.listdir(model_dir)]


def main(input_path, model_path, lemma_length_cutoff, max_iters,
        output_dir):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    converters = {"lemmas": eval}
    with open(model_path, "rb") as f:
        all_models = pd.read_csv(f, sep="\t", converters=converters)
    docset_ids = set(all_models["docset id"].tolist())

    with open(input_path, "rb") as f:
        all_inputs = pd.read_csv(f, sep="\t", converters=converters)
        print("Read {} input sentences from {}".format(
            len(all_inputs), input_path))

    if lemma_length_cutoff > 0:
        all_inputs = lemma_filter(all_inputs, lemma_length_cutoff)

    vw_str = "--search 0 --csoaa_ldf m --search_task hook --ring_size 1024 " \
             "--search_no_caching --quiet --noconstant"
    vw = pyvw.vw(vw_str)
    instances = make_instances(vw, all_models, all_inputs)
    #instances = instances[0:1]
    task = vw.init_search_task(L2SSum)

    from datetime import datetime, timedelta
    now = datetime.now()
    total_train_time = timedelta(0)
    for num_iter in range(1, max_iters + 1):    
        print("iter {}/{}".format(num_iter, max_iters))
        task.learn(instances)
        dur = datetime.now() - now
        total_train_time += dur
        print("took {}".format(dur))
        now = datetime.now()
        write_weights(output_dir, num_iter, vw, instances[0][-1])
        output_paths = []
        for instance in instances:
            docset, opath = write_output(
                task.predict(instance), num_iter, instance, output_dir)
            output_paths.append((docset, opath))
        write_eval(output_paths, num_iter, output_dir)

    print total_train_time, 
    print timedelta(seconds=total_train_time.total_seconds() / 10.)
#        for instance in instances:
#            
#            model_ngrams, model_Z, input_ngrams, input_Z, inputs = instance
#            summary_i = task.predict(instance)
#            docset_id = inputs["docset id"].tolist()[0]
#            system_path = os.path.join(output_dir, "iter{}_{}".format(
#                num_iter, docset_id))
#            with open(system_path, "w") as f:
#                f.write("\n".join(inputs.iloc[summary_i]["text"].tolist()))
#
#



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--duc-inputs", 
            help="preprocessed input tsv", required=True)
    parser.add_argument("--duc-models", 
            help="preprocessed models tsv", required=True)
    parser.add_argument("--stopped-lemma-length-filter",
        type=int, required=True,
        help="remove sentences with stopped lemmas less than argval")
    parser.add_argument("--iters", required=True, type=int,
            help="Number of training iterations.")
    parser.add_argument("-o", required=True, help="Output directory")
    args = parser.parse_args()

    main(args.duc_inputs, 
         args.duc_models, 
         args.stopped_lemma_length_filter,
         args.iters, args.o)


