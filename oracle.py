import pandas as pd
from nltk.util import ngrams
from collections import defaultdict
import os

duc03_model_path = os.path.join(
    os.getenv("DUC_DATA"),
    "detagged.duc2003.abstracts/models/")

def oracle_run(model_bigrams, index, inputs):
    Z = sum(count for count in model_bigrams.values())
    summary_bigrams = defaultdict(int)
    summary_length = 0
    input_bigrams = []
    for lemmas in inputs["lemmas"].tolist():
        bigrams = defaultdict(int)
        for bg in ngrams(lemmas, 2):
            if bg in model_bigrams:
                bigrams[bg] += 1
        input_bigrams.append(bigrams)

    summary_i = []

    
    while summary_length < 100:
        max_score = 0
        max_i = None
        scores = []
        
        for i, v in enumerate(index):
          #  print i, v
            score = 0
            for bg in set(input_bigrams[i].keys() + summary_bigrams.keys()):
                bg_score = min(model_bigrams[bg], 
                               input_bigrams[i][bg] + summary_bigrams[bg])
          #      print bg_score, model_bigrams[bg], 
         #       print input_bigrams[i][bg] + summary_bigrams[bg]

                score += bg_score
        #    print "SCORE:", score
            scores.append(score)
        max_score = max(scores)
        max_i = scores.index(max_score)
       #
        print max_i, index[max_i], inputs.iloc[index[max_i]]["text"]
        summary_length += inputs.iloc[index[max_i]]["word length"]
        #print summary_length
        summary_i.append(index[max_i])
        for bg, count in input_bigrams[max_i].items():
            summary_bigrams[bg] += count

        del index[max_i]
        del input_bigrams[max_i]
    texts = inputs.iloc[summary_i]["text"].tolist()
    return texts

converters = {"lemmas": eval}
with open("duc2003.models.tsv", "rb") as f:
    all_models = pd.read_csv(f, sep="\t", converters=converters)
docset_ids = set(all_models["docset id"].tolist())

with open("duc2003.input.tsv", "rb") as f:
    all_inputs = pd.read_csv(f, sep="\t", converters=converters)

output_dir = "oracle-runs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

model_output_dir = "models"
if not os.path.exists(model_output_dir):
    os.makedirs(model_output_dir)



with open("oracle.eval", "w") as evalf:
    docset_ids = sorted(list(docset_ids))
    for docset_id in docset_ids:
        print docset_id
        models = all_models.loc[all_models["docset id"] == docset_id]
        sents = models["lemmas"].tolist()
        model_bigrams = defaultdict(int)
        for lemmas in sents:
            for bg in ngrams(lemmas, 2):
                model_bigrams[bg] = 1

        inputs = all_inputs.loc[all_inputs["docset id"] == docset_id]
        inputs = inputs.reset_index(drop=True)

        
        index = inputs.index.tolist()
        oracle_summary = oracle_run(model_bigrams, index, inputs)
        output_path = os.path.join(
            output_dir, "{}.oracle.spl".format(docset_id))
        with open(output_path, "w") as f:
            f.write("\n".join(oracle_summary))
                
        model_paths = [os.path.join(duc03_model_path, model_id)
                       for model_id in set(models["doc id"].tolist())]

        model_paths = []
        for model_id, model in models.groupby("doc id"):
            model_path = os.path.join(
                model_output_dir, "{}.spl".format(model_id))
            print model_id
            with open(model_path, "w") as mf:
                mf.write("\n".join(model["text"].tolist()))
            model_paths.append(model_path) 
        model_paths.sort() 
        evalf.write(" ".join(model_paths + [output_path]) + "\n")
