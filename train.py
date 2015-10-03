from cpyvw import SearchTask
import l2sum
from l2sum._learn import L2SSum
import pyvw
import pylibvw
import pandas as pd
from collections import defaultdict
from nltk.util import ngrams


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

def make_instances(all_models, all_inputs, ngram_size=1):
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
        
        instances.append(
                (model_ngrams, model_Z, input_ngrams, input_Z, inputs))
    return instances
 

def main(input_path, model_path, lemma_length_cutoff, max_iters):

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

    instances = make_instances(all_models, all_inputs)

    vw_str = "--search 0 --csoaa_ldf m --search_task hook --ring_size 1024 " \
             "--search_no_caching --quiet"
    vw = pyvw.vw(vw_str)
    task = vw.init_search_task(L2SSum)

    from datetime import datetime, timedelta
    now = datetime.now()
    total_train_time = timedelta(0)
    for num_iter in range(1, max_iters + 1):    
        print("iter {}/{}".format(num_iter, max_iters + 1))
        task.learn(instances[0:1])
        dur = datetime.now() - now
        total_train_time += dur
        print("took {}".format(dur))
        now = datetime.now()
    print total_train_time
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
    args = parser.parse_args()

    main(args.duc_inputs, 
         args.duc_models, 
         args.stopped_lemma_length_filter,
         args.iters)


