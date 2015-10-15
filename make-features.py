from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
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

def lexrank(X, d, iters, tol):
    K = cosine_similarity(X)
    M_hat = (d * K) + \
            (float(1 - d) / X.shape[0]) * np.ones(
                (X.shape[0], X.shape[0]))
    M_hat /=  np.sum(M_hat, axis=0)
    r = np.ones((X.shape[0]), dtype=np.float64) / X.shape[0]

    converged = False
    for n_iter in xrange(iters):
        last_r = r
        r = np.dot(M_hat, r)

        if (np.abs(r - last_r) < tol).any():
            converged = True
            break

    if not converged:
        print "warning:", 
        print "lexrank failed to converged after {} iters".format(
            iters)
    return r


def make_features(all_inputs, output_tsv):
    docset_ids = sorted(all_inputs["docset id"].unique())
    for docset_id in docset_ids:
        print("Docset ID: {}".format(docset_id))
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

        tfidf_vec = TfidfTransformer()
        X_tfidf = tfidf_vec.fit_transform(X_tf)

        centroid_tfidf = X_tfidf.mean(axis=0)
        assert (centroid_tfidf >= 0).all()
        fid = "INPUT_TFIDF_CENTROID_SIM"
        all_inputs.loc[docset_select, fid] = cosine_similarity(
                X_tfidf, centroid_tfidf)

        
        
        # Compute lexrank
        fid = "INPUT_TF_CONT_LEXRANK"
        all_inputs.loc[docset_select, fid] = lexrank(X_tf, .85, 200, .0001)
        fid = "INPUT_TFIDF_CONT_LEXRANK"
        all_inputs.loc[docset_select, fid] = lexrank(X_tfidf, .85, 200, .0001)
        



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
            return sum(input_uni[lem] for lem in lemmas)
        fid = "INPUT_FREQSUM_UNI_AMEAN"
        fs = inputs["lemmas"].apply(freqsum_uni_amean) / inputs["word length"]
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
            return np.sum(probs)
        fid = "INPUT_FREQSUM_NE_PERSON_AMEAN"
        fs = inputs["ne"].apply(freqsum_per_amean) / inputs["word length"]
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
            return np.sum(probs)
        fid = "INPUT_FREQSUM_NE_LOCATION_AMEAN"
        fs = inputs["ne"].apply(freqsum_loc_amean) / inputs["word length"]
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
            return np.sum(probs)
        fid = "INPUT_FREQSUM_NE_ORGANIZATION_AMEAN"
        fs = inputs["ne"].apply(freqsum_org_amean) / inputs["word length"]
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

            tfidf_vec = TfidfTransformer()
            Xd_tfidf = tfidf_vec.fit_transform(Xd_tf)

            doc_centroid_tfidf = Xd_tfidf.mean(axis=0)
            assert (doc_centroid_tfidf >= 0).all()
            fid = "DOC_TFIDF_CENTROID_SIM"
            all_inputs.loc[doc_select, fid] = cosine_similarity(
                Xd_tfidf, doc_centroid_tfidf)

            # Compute lexrank
            fid = "DOC_TF_CONT_LEXRANK"
            all_inputs.loc[doc_select, fid] = lexrank(Xd_tf, .85, 200, .0001)
            fid = "DOC_TFIDF_CONT_LEXRANK"
            all_inputs.loc[doc_select, fid] = lexrank(
                Xd_tfidf, .85, 200, .0001)
        
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
                return np.sum(probs)
            fid = "DOC_FREQSUM_UNI_AMEAN"
            fs = doc["lemmas"].apply(freqsum_uni_amean) / doc["word length"]
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

    all_inputs["SENT_LENGTH"] = all_inputs["tokens"].apply(len).values.astype("float64").ravel()
    all_inputs["DOC_POSITION"] = all_inputs["sent id"]
    all_inputs["DOC_IS_LEAD"] = all_inputs["sent id"].apply(lambda x: x in set([1, 2])).astype("int64")
    all_inputs["SENT_NUM_PRON"] = all_inputs["pos"].apply(lambda X: sum([1 if pos == "PRP" else 0 for tok, pos in X]))
    all_inputs["SENT_NUM_QUOT"] = all_inputs["pos"].apply(lambda X: sum([1 if pos in set(["''", "``"]) else 0 for tok, pos in X]))
    


    for c in all_inputs.columns:
        print "Checking {} for NANs".format(c)
        assert pd.notnull(all_inputs[c]).all()
    with open(output_tsv, "w") as f:
        all_inputs.to_csv(f, sep="\t", index=False)
    return all_inputs


def main():
    converters = {"lemmas": eval, "ne": eval, "pos": eval}
    with open("duc2003.inputs.tsv", "rb") as f:
        all_inputs = pd.read_csv(f, sep="\t", converters=converters)
    inputs2003 = make_features(all_inputs, "duc2003.inputs.feats.tsv")
    inputs2003["year"] = 2003
    with open("duc2004.inputs.tsv", "rb") as f:
        all_inputs = pd.read_csv(f, sep="\t", converters=converters)
    inputs2004 = make_features(all_inputs, "duc2004.inputs.feats.tsv")
    inputs2004["year"] = 2004
    df = pd.concat([inputs2003, inputs2004])
    with open("duc2003-2004.feats.tsv", "w") as f:
        df.to_csv(f, sep="\t", index=False)

if __name__ == "__main__":
    main()
