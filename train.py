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
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
pd.options.display.float_format = '{:.3f}'.format
from sumpy.util import DUCHelper
dh = DUCHelper()
import corenlp as cnlp

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

def make_model_ngrams(ds, ngram_size, macro=True):
    
    if macro is True:
        all_counts = []
        for model in ds.model_iter():
            model_Z = 0.
            model_counts = {}
            model_text = unicode(model)
            sents = model_text.split(u"\n")
            for sent in sents:
                for ng in sent.lower().split(u" "):
                    model_counts[ng] = model_counts.get(ng, 0.) + 1.
                    model_Z += 1.
            all_counts.append((model_counts, model_Z))
        return all_counts
    else:
        raise Exception("Not implemented")

#    sents = models["lemmas"].tolist()
#    model_ngrams = defaultdict()
#    for lemmas in sents:
#        for ng in ngrams(lemmas, ngram_size):
#            model_ngrams[ng] = 1. + model_ngrams.get(ng, 0.)
#    model_Z = float(sum(count for count in model_ngrams.values()))
#    return model_ngrams, model_Z

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

def make_instances(vw, all_inputs, features, use_interactions, ngram_size=1):

    duc2003 = dh.docsets(2003, 2)
    duc2004 = dh.docsets(2004, 2)

    instances = []

    for year, inputs_year in all_inputs.groupby("year"):
        for docset_id, inputs in inputs_year.groupby("docset id"):
            if year == 2003:
                ds = duc2003[docset_id]
            else:
                ds = duc2004[docset_id]
            print docset_id, year
            
            model_ngrams = make_model_ngrams(ds, ngram_size)

            inputs = all_inputs.loc[all_inputs["docset id"] == docset_id]
            inputs = inputs.reset_index(drop=True)

            input_centroid_f = [u'INPUT_TFIDF_CENTROID_SIM',]
            input_lexrank_f = [u'INPUT_TFIDF_CONT_LEXRANK',]


            #doc_centroid_f = ['DOC_TF_CENTROID_SIM',]

            input_freqsum_f = ['INPUT_FREQSUM_UNI_AMEAN',] 
                            #   'INPUT_FREQSUM_UNI_MAX',
                            #   'INPUT_FREQSUM_UNI_GMEAN']
          #  input_freqsum_ne_f = ['INPUT_FREQSUM_NE_PERSON_AMEAN', 
          #                        'INPUT_FREQSUM_NE_LOCATION_AMEAN',
          #                        'INPUT_FREQSUM_NE_ORGANIZATION_AMEAN',] 
          #  doc_freqsum_f = ['DOC_FREQSUM_UNI_AMEAN', 
          #                   'DOC_FREQSUM_UNI_MAX',
          #                   'DOC_FREQSUM_UNI_GMEAN',]
            doc_basic_f = ['DOC_POSITION', 'DOC_IS_LEAD',] 
            sent_basic_f = ['SENT_LENGTH', 'SENT_NUM_PRON', 'SENT_NUM_QUOT'] 

            #features = input_centroid_f + input_lexrank_f + \
            #    input_freqsum_f + doc_basic_f + sent_basic_f
               # + input_freqsum_ne_f + doc_freqsum_f + \

            

            fi = FeatureIndexer(features, use_interactions)
            Xinp_sf, examples = fi.make_static_data(vw, inputs)
            tf_vec = DictVectorizer()
            tfidf_vec = TfidfTransformer()
            Xinp_tf = inputs["lemmas"].apply(make_counts)
            Xinp_tf = tf_vec.fit_transform(Xinp_tf)
            #Xinp_tfidf = tfidf_vec.fit_transform(Xinp_tf)
            Kinp_tf = cosine_similarity(Xinp_tf)
            instances.append(
                    (model_ngrams,
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


def write_output(summary_indices, num_iter, instance, output_dir, dev=False):
    (model_ngrams, 
     inputs, Kinp_tf, Xinp_sf, examples, fi) = instance
    docset_id = inputs["docset id"].unique()[0]
    year = inputs["year"].unique()[0]
    path = os.path.join(output_dir, 
        "{}.{}{}.iter{}.spl".format(
            year, docset_id, ".dev" if dev else "", num_iter))
    with open(path, "w") as f:
        f.write("\n".join(inputs.iloc[summary_indices]["text"].tolist()))
    return docset_id, year, path    

def write_eval(output_paths, num_iter, output_dir):
    eval_path = os.path.join(
        output_dir, "system.eval.iter{}".format(num_iter))
    duc2003 = dh.docsets(2003, 2)
    duc2004 = dh.docsets(2004, 2)
    with open(eval_path, "w") as f:
        for docset, year, output_path in output_paths:
            if year == 2003:
                model_paths = [m.path for m in duc2003[docset].model_iter()]
            else:
                model_paths = [m.path for m in duc2004[docset].model_iter()]
            f.write(" ".join(model_paths + [output_path]) + "\n")

def write_eval_dev(output_paths, num_iter, output_dir):
    eval_path = os.path.join(
        output_dir, "system.dev.eval.iter{}".format(num_iter))
    
    duc2003 = dh.docsets(2003, 2)
    duc2004 = dh.docsets(2004, 2)

    with open(eval_path, "w") as f:
        for docset, year, output_path in output_paths:
            if year == 2003:
                model_paths = [m.path for m in duc2003[docset].model_iter()]
            else:
                model_paths = [m.path for m in duc2004[docset].model_iter()]
            f.write(" ".join(model_paths + [output_path]) + "\n")

def main(input_path, features, loss_metric, fold,
        lemma_length_cutoff, use_interactions, max_iters,
        output_dir):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    converters = {"lemmas": eval, "tokens": eval}

    with open(input_path, "rb") as f:
        all_inputs = pd.read_csv(f, sep="\t", converters=converters)
        print("Read {} input sentences from {}".format(
            len(all_inputs), input_path))

    if lemma_length_cutoff > 0:
        all_inputs = lemma_filter(all_inputs, lemma_length_cutoff)

    vw_str = "--search 0 --csoaa_ldf m --search_task hook --ring_size 1024 " \
             "--search_no_caching --quiet --noconstant"
    vw = pyvw.vw(vw_str)
    instances = make_instances(vw, all_inputs, features, use_interactions)

    chunk_size = 20
    chunks = [instances[i:i+chunk_size] for i in range(0, 80, 20)]
    instances_train = []
    for i, chunk in enumerate(chunks):
        if i != fold:
            instances_train.extend(chunk)
        else:
            instances_dev = chunks[i]
    print "Fold {}".format(fold)

    task = vw.init_search_task(L2SSum)
    task.set_loss_func(loss_metric)
    print task.get_loss_func()

    from datetime import datetime, timedelta
    now = datetime.now()
    total_train_time = timedelta(0)
    for num_iter in range(1, max_iters + 1):    
        print("iter {}/{}".format(num_iter, max_iters))
        task.learn(instances_train)
        dur = datetime.now() - now
        total_train_time += dur
        print("took {}".format(dur))
        now = datetime.now()
        write_weights(output_dir, num_iter, vw, instances[0][-1])
        
        output_paths = []
        for instance in instances_train:
            docset, year, opath = write_output(
                task.predict(instance), num_iter, instance, output_dir)
            output_paths.append((docset, year, opath))
        write_eval(output_paths, num_iter, output_dir)

        output_paths_dev = []
        for instance in instances_dev:
            docset, year, opath = write_output(
                task.predict(instance), num_iter, instance, output_dir,
                dev=True)
            output_paths_dev.append((docset, year, opath))
        write_eval_dev(output_paths_dev, num_iter, output_dir)



    print total_train_time, 
    print timedelta(seconds=total_train_time.total_seconds() / 10.)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--duc-inputs", 
            help="preprocessed input tsv", required=True)

    parser.add_argument("--stopped-lemma-length-filter",
        type=int, required=True,
        help="remove sentences with stopped lemmas less than argval")
    parser.add_argument("--iters", required=True, type=int,
            help="Number of training iterations.")
    parser.add_argument("--features", nargs="+",
            required=True)
    parser.add_argument("--loss", required=True, 
        choices=["p", "r", "f"],)
    parser.add_argument("--fold", required=True, type=int,
        choices=[0,1,2,3])
    parser.add_argument("--inter", required=True, type=int,
            choices=[0,1])
#    parser.add_argument("--loss-alpha", required=True, type=float,
#        help="loss = alp * (1 -f1) + (1 - alp) * E[sim]")
    parser.add_argument("-o", required=True, help="Output directory")
    args = parser.parse_args()

    main(args.duc_inputs, args.features, args.loss, args.fold,
         args.stopped_lemma_length_filter, args.inter,
         args.iters, args.o)


