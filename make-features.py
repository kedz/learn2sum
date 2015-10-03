from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy.stats.mstats import gmean
from collections import defaultdict
import pandas as pd
import os


def make_counts(lemmas):
    counts = defaultdict(int)
    for lemma in lemmas:
        counts[lemma] += 1
    return counts




def main():
    converters = {"lemmas": eval, "ne": eval, "pos": eval}
    with open("duc2003.models.tsv", "rb") as f:
        all_models = pd.read_csv(f, sep="\t", converters=converters)
    docset_ids = set(all_models["docset id"].tolist())

    with open("duc2003.input.tsv", "rb") as f:
        all_inputs = pd.read_csv(f, sep="\t", converters=converters)

    output_dir = "system-runs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for docset_id in docset_ids:
        models = all_models.loc[all_models["docset id"] == docset_id]
        docset_select = all_inputs["docset id"] == docset_id
        inputs = all_inputs.loc[docset_select]
        doc_ids = set(inputs["doc id"].tolist())

        # Compute centroid similarity
        vec = DictVectorizer()
        X_tf = inputs["lemmas"].apply(make_counts)

        X_tf = vec.fit_transform(X_tf)
        print X_tf.shape
        centroid = X_tf.mean(axis=0)
        fid = "INPUT_TF_CENTROID_SIM"
        all_inputs.loc[docset_select, fid] = cosine_similarity(X_tf, centroid)
        
        # FreqSum (SumBasic) Features
        input_uni = defaultdict(int)
        Z_inp = 0.0
        for lemmas in inputs["lemmas"].tolist():
            for tok in lemmas:
                input_uni[tok] += 1
            Z_inp += len(lemmas)
        for tok, count in input_uni.items():
            input_uni[tok] = count / Z_inp
        def freqsum_uni_amean(lemmas):
            Z = len(lemmas)
            if Z == 0: Z = 1
            return sum(input_uni[lem] for lem in lemmas) / Z
        fid = "INPUT_FREQSUM_UNI_AMEAN"
        fs = inputs["lemmas"].apply(freqsum_uni_amean)
        all_inputs.loc[docset_select, fid] = fs

        def freqsum_uni_max(lemmas):
            if len(lemmas) == 0: return 0.0
            return max(input_uni[lem] for lem in lemmas)
        fid = "INPUT_FREQSUM_UNI_MAX"
        fs = inputs["lemmas"].apply(freqsum_uni_max)
        all_inputs.loc[docset_select, fid] = fs

        def freqsum_uni_gmean(lemmas):
            if len(lemmas) == 0: return 0
            return gmean([input_uni[lem] for lem in lemmas])
        fid = "INPUT_FREQSUM_UNI_GMEAN"
        fs = inputs["lemmas"].apply(freqsum_uni_gmean)
        all_inputs.loc[docset_select, fid] = fs


        input_per = defaultdict(int)
        Z_inp_per = 0.0
        for ners in inputs["ne"].tolist():
            for tok, ne in ners:
                if ne == "PERSON":
                    input_per[tok] += 1
                    Z_inp_per += 1.0
        for tok, count in input_per.items():
            input_per[tok] = count / Z_inp_per
        def freqsum_per_amean(nes):
            probs = [input_per[tok] for tok, ne in nes if ne == "PERSON"]
            if len(probs) == 0: return 0
            return np.mean(probs)
        fid = "INPUT_FREQSUM_NE_PERSON_AMEAN"
        fs = inputs["ne"].apply(freqsum_per_amean)
        all_inputs.loc[docset_select, fid] = fs

        input_loc = defaultdict(int)
        Z_inp_loc = 0.0
        for ners in inputs["ne"].tolist():
            for tok, ne in ners:
                if ne == "LOCATION":
                    input_loc[tok] += 1
                    Z_inp_loc += 1.0
        for tok, count in input_loc.items():
            input_loc[tok] = count / Z_inp_loc
        def freqsum_loc_amean(nes):
            probs = [input_loc[tok] for tok, ne in nes if ne == "LOCATION"]
            if len(probs) == 0: return 0
            return np.mean(probs)
        fid = "INPUT_FREQSUM_NE_LOCATION_AMEAN"
        fs = inputs["ne"].apply(freqsum_loc_amean)
        all_inputs.loc[docset_select, fid] = fs

        input_org = defaultdict(int)
        Z_inp_org = 0.0
        for ners in inputs["ne"].tolist():
            for tok, ne in ners:
                if ne == "ORGANIZATION":
                    input_org[tok] += 1
                    Z_inp_org += 1.0
        for tok, count in input_org.items():
            input_org[tok] = count / Z_inp_org
        def freqsum_org_amean(nes):
            probs = [input_org[tok] for tok, ne in nes if ne == "ORGANIZATION"]
            if len(probs) == 0: return 0
            return np.mean(probs)
        fid = "INPUT_FREQSUM_NE_ORGANIZATION_AMEAN"
        fs = inputs["ne"].apply(freqsum_org_amean)
        all_inputs.loc[docset_select, fid] = fs


        for doc_id in doc_ids:
            
            doc_select = (all_inputs["doc id"] == doc_id) & docset_select
            doc = all_inputs.loc[doc_select]
            
            # Doc Centroid Feature
            Xd_tf = doc["lemmas"].apply(make_counts)
            Xd_tf = vec.fit_transform(Xd_tf)
            doc_centroid = Xd_tf.mean(axis=0)
            fid = "DOC_TF_CENTROID_SIM"
            all_inputs.loc[doc_select, fid] = cosine_similarity(
                    Xd_tf, doc_centroid)

            # FreqSum (SumBasic) Features
            doc_uni = defaultdict(int)
            Z_doc = 0.0
            for lemmas in doc["lemmas"].tolist():
                for tok in lemmas:
                    doc_uni[tok] += 1
                Z_doc += len(lemmas)
            for tok, count in doc_uni.items():
                doc_uni[tok] = count / Z_doc
            def freqsum_uni_amean(lemmas):
                if len(lemmas) == 0: return 0
                probs = [doc_uni[lem] for lem in lemmas]
                return np.mean(probs)
            fid = "DOC_FREQSUM_UNI_AMEAN"
            fs = inputs["lemmas"].apply(freqsum_uni_amean)
            all_inputs.loc[doc_select, fid] = fs

            def freqsum_uni_max(lemmas):
                if len(lemmas) == 0: return 0.0
                return max(doc_uni[lem] for lem in lemmas)
            fid = "DOC_FREQSUM_UNI_MAX"
            fs = doc["lemmas"].apply(freqsum_uni_max)
            all_inputs.loc[doc_select, fid] = fs

            def freqsum_uni_gmean(lemmas):
                if len(lemmas) == 0: return 0
                return gmean([doc_uni[lem] for lem in lemmas])
            fid = "DOC_FREQSUM_UNI_GMEAN"
            fs = doc["lemmas"].apply(freqsum_uni_gmean)
            all_inputs.loc[doc_select, fid] = fs
    from sklearn.preprocessing import scale
    all_inputs["SENT_LENGTH"] = scale(all_inputs["tokens"].apply(len).values.astype("float64").ravel())
    all_inputs["DOC_POSITION"] = all_inputs["sent id"]
    all_inputs["DOC_IS_LEAD"] = all_inputs["sent id"].apply(lambda x: x in set([1, 2])).astype("int64")
    all_inputs["SENT_NUM_PRON"] = all_inputs["pos"].apply(lambda X: sum([1 if pos == "PRP" else 0 for tok, pos in X]))
    all_inputs["SENT_NUM_QUOT"] = all_inputs["pos"].apply(lambda X: sum([1 if pos in set(["''", "``"]) else 0 for tok, pos in X]))
    


    for c in all_inputs.columns:
        print "Checking {} for NANs".format(c)
        assert pd.notnull(all_inputs[c]).all()
    with open("duc2003.input.feats.tsv", "w") as f:
        all_inputs.to_csv(f, sep="\t", index=False)
    print all_inputs[["SENT_LENGTH", "SENT_NUM_PRON", "SENT_NUM_QUOT"]]

if __name__ == "__main__":
    main()
