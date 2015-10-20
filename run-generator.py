
import os
from itertools import permutations, combinations

f2s = {"CENTROID_SIM": "csim", "CONT_LEXRANK": "clr", 
        "FREQSUM_UNI_AMEAN": "sb.a", "FREQSUM_UNI_MAX": "sb.m",
        "FREQSUM_UNI_GMEAN": "sb.g", 
        "FREQSUM_NE_PERSON_AMEAN": "sb.per.a",
        "FREQSUM_NE_LOCATION_AMEAN": "sb.loc.a",
        "FREQSUM_NE_ORGANIZATION_AMEAN": "sb.org.a",
        "LENGTH": "len", "IS_LEAD": "lede", "NUM_PRON": "nprn",
        "NUM_QUOT": "nqte", "POSITION": "pos"}

features = ["INPUT_TF_CENTROID_SIM", "INPUT_TFIDF_CENTROID_SIM", 
            "INPUT_TF_CONT_LEXRANK", "INPUT_TFIDF_CONT_LEXRANK", 
            "INPUT_FREQSUM_UNI_AMEAN", "INPUT_FREQSUM_UNI_MAX", 
            "INPUT_FREQSUM_UNI_GMEAN", "INPUT_FREQSUM_NE_PERSON_AMEAN", 
            "INPUT_FREQSUM_NE_LOCATION_AMEAN", 
            "INPUT_FREQSUM_NE_ORGANIZATION_AMEAN", "DOC_TF_CENTROID_SIM", 
            "DOC_TFIDF_CENTROID_SIM", "DOC_TF_CONT_LEXRANK", 
            "DOC_TFIDF_CONT_LEXRANK", "DOC_FREQSUM_UNI_AMEAN", 
            "DOC_FREQSUM_UNI_MAX", "DOC_FREQSUM_UNI_GMEAN", "SENT_LENGTH", 
            "DOC_POSITION", "DOC_IS_LEAD", "SENT_NUM_PRON", "SENT_NUM_QUOT"]

def main(duc_input_path, num_feats, train_path, bash_file_dir, output_path):

    if not os.path.exists(bash_file_dir):
        os.makedirs(bash_file_dir)

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    for i in num_feats:
        results_path = os.path.join(output_path, "results-{}".format(i))
        if not os.path.exists(results_path):
            os.makedirs(results_path)

        for perm in combinations(features, i):
            input_tfidf_feats = []
            input_tf_feats = []
            input_feats = []
            doc_tfidf_feats = []
            doc_tf_feats = []
            doc_feats = []
            sent_feats = []

            feats = [f for f in perm] 

            for feature in feats:
                items = feature.split("_")
                if items[0] == "INPUT" and items[1] == "TFIDF":
                    input_tfidf_feats.append(f2s["_".join(items[2:])])
                elif items[0] == "DOC" and items[1] == "TFIDF":
                    doc_tfidf_feats.append(f2s["_".join(items[2:])])
                elif items[0] == "INPUT" and items[1] == "TF":
                    input_tf_feats.append(f2s["_".join(items[2:])])
                elif items[0] == "DOC" and items[1] == "TF":
                    doc_tf_feats.append(f2s["_".join(items[2:])])
                elif items[0] == "INPUT":
                    input_feats.append(f2s["_".join(items[1:])])
                elif items[0] == "DOC":
                    doc_feats.append(f2s["_".join(items[1:])])
                elif items[0] == "SENT":
                    sent_feats.append(f2s["_".join(items[1:])])
                else:
                    raise Exception("Found this: " + feature)
            
            fstring = ""

            if len(doc_feats) + len(doc_tf_feats) + len(doc_tfidf_feats) > 0:
                fstring += "doc."
            
            if len(doc_feats) > 0:
                for doc_feat in sorted(doc_feats):
                    fstring += doc_feat + "."
            
            if len(doc_tf_feats) > 0:
                fstring += "tf."
                for doc_tf_feat in sorted(doc_tf_feats):
                    fstring += doc_tf_feat + "."    

            if len(doc_tfidf_feats) > 0:
                fstring += "tfidf."
                for doc_tfidf_feat in sorted(doc_tfidf_feats):
                    fstring += doc_tfidf_feat + "."

            if len(input_feats) + len(input_tf_feats) + len(input_tfidf_feats) > 0:
                fstring += "input."

            if len(input_feats) > 0:
                for input_feat in sorted(input_feats):
                    fstring += input_feat + "." 

            if len(input_tf_feats) > 0:
                fstring += "tf."
                for input_tf_feat in sorted(input_tf_feats):
                    fstring += input_tf_feat + "."    

            if len(input_tfidf_feats) > 0:
                fstring += "tfidf."
                for input_tfidf_feat in sorted(input_tfidf_feats):
                    fstring += input_tfidf_feat + "."    
           
            if len(sent_feats) > 0:
                fstring += "sent."
                for sent_feat in sorted(sent_feats):
                    fstring += sent_feat + "."    
            for inter in [0, 1]:
                if inter == 1:
                    fstring += "tfint."

                for fold in range(4):
                    name = fstring + "optr.{}".format(fold)
                    bash_file = os.path.join(bash_file_dir, name + ".sh")
                    with open(bash_file, "w") as bf:
                        bf.write("if [ ! -d \"{}/{}\" ]; then\n".format(
                            results_path, name))
                        bf.write("mkdir {}\n".format(
                            os.path.join(results_path, name)))
                        bf.write("    time python -u {}/train.py ".format(
                            train_path)) 
                        bf.write("--duc-inputs {} \\\n".format(duc_input_path))

                        bf.write("                ")
                        bf.write("--stopped-lemma-length-filter 10 \\\n")

                        bf.write("                --iters 20 \\\n")
                        bf.write("                --loss r \\\n")
                        bf.write("                --features {} \\\n".format(
                            feats[0]))
                        for f in feats[1:]:
                            bf.write("                    {} \\\n".format(f))
                        bf.write("                --fold {} \\\n".format(fold))
                        bf.write("                --inter {} \\\n".format(
                            inter))
                        bf.write("                -o {}/{} \\\n".format(
                            results_path, name))
                        bf.write("                &> {}/{}/log\n".format(
                            results_path, name))
                        bf.write("fi\n")
        


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--duc-inputs", 
            help="preprocessed input tsv", required=True)
    parser.add_argument("--num-feats", nargs="+", required=True, type=int)
    parser.add_argument("-d", required=True)
    parser.add_argument("-t", required=True)
    parser.add_argument("-o", required=True, help="Output directory")
    args = parser.parse_args()
    main(args.duc_inputs, args.num_feats, args.t, args.d, args.o)


