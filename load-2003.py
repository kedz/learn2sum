import os
import re
import corenlp as cnlp
from datetime import datetime
import pandas as pd

stopwords = set()
with open(u"stopwords.en.txt", "rb") as f:
    for line in f:
        stopwords.add(line.strip())

def load_docset(path, pipeline, data):
    docset_id = os.path.split(path)[1]
    files = [file for file in os.listdir(path)]
    files.sort()

    for file in files:
        print file
        docid = file
        timestamp = int(file[3:7]), int(file[7:9]), int(file[9:11])
        timestamp = datetime(*timestamp)

        with open(os.path.join(path, file), "rb") as f:
            text = f.read()
            #print text
            m = re.search(r"<TEXT>(.*)</TEXT>", text, re.DOTALL)
            text = m.group(1) 
            text = re.sub(r"</?P>", "", text)
            text = re.sub(r"<ANNOTATION>.*?</ANNOTATION>", "", text, flags=re.DOTALL)
            doc = pipeline.annotate(text)
            for i, sent in enumerate(doc, 1):
                datum = {}
                datum["docset id"] = docset_id
                datum["doc id"] = docid
                datum["timestamp"] = timestamp
                datum["sent id"] = i
                datum["text"] = u" ".join(
                        unicode(tok) for tok in sent).encode("utf-8")
                datum["word length"] = len([tok for tok in sent])
                datum["byte length"] = len(datum["text"])
                datum["tokens"] = [unicode(tok).lower().encode("utf-8") 
                                   for tok in sent 
                                   if unicode(tok).lower() not in stopwords]
                datum["lemmas"] = [tok.lem.encode("utf-8") for tok in sent
                                   if tok.lem.encode("utf-8") not in stopwords]
                        
                datum["ne"] = [(tok.lem.lower().encode("utf-8"), 
                                tok.ne.encode("utf-8")) 
                               for tok in sent
                               if tok.ne != u"O"]
                datum["pos"] = [(unicode(tok).lower().encode("utf-8"), tok.pos)
                                for tok in sent]

                data.append(datum)
    print 


def load_models(model_dir, docset_id, pipeline, data):
    ds_id = docset_id[:-1].upper()
    files = []
    for file in os.listdir(model_dir):
        if re.search(ds_id + r"\.\w\.100\.\w\.\w\.html", file):
            files.append(file)
    for file in files:
        with open(os.path.join(model_dir, file), "rb") as f:
            print file
            text = f.read()
            doc = pipeline.annotate(text)
            for i, sent in enumerate(doc, 1):
                datum = {}
                datum["docset id"] = docset_id
                datum["doc id"] = file
                datum["sent id"] = i
                datum["text"] = u" ".join(
                        unicode(tok) for tok in sent).encode("utf-8")
                datum["word length"] = len([tok for tok in sent])
                datum["byte length"] = len(datum["text"])
                datum["tokens"] = [unicode(tok).lower().encode("utf-8") 
                                   for tok in sent 
                                   if unicode(tok).lower() not in stopwords]
                datum["lemmas"] = [tok.lem.encode("utf-8") for tok in sent
                                   if tok.lem.encode("utf-8") not in stopwords]
                datum["ne"] = [(tok.lem.lower().encode("utf-8"), 
                                tok.ne.encode("utf-8")) 
                               for tok in sent
                               if tok.ne != u"O"]
                data.append(datum)
    print

duc03_path = os.path.join(
    os.getenv("DUC_DATA"),
    "DUC2003_Summarization_Documents/duc2003_testdata/task2/docs")
duc03_model_path = os.path.join(
    os.getenv("DUC_DATA"),
    "detagged.duc2003.abstracts/models/")

annotators=["tokenize", "ssplit", "pos", "lemma", "ner"]
with cnlp.Server(annotators=annotators) as pipeline:
    data = []
    for docset_dir in os.listdir(duc03_path):
        load_models(duc03_model_path, docset_dir, pipeline, data)
    df = pd.DataFrame(
        data, 
        columns=[
            "docset id", "doc id", "sent id", "text", 
            "word length", "byte length", "tokens", "lemmas", "ne"])
    with open("duc2003.models.tsv", "w") as f:
        df.to_csv(f, sep="\t", index=False) 

    data = []
    for docset_dir in os.listdir(duc03_path):
        load_docset(os.path.join(duc03_path, docset_dir), pipeline, data)
    df = pd.DataFrame(
        data, 
        columns=[
            "docset id", "doc id", "timestamp", "sent id", "text", 
            "word length", "byte length", "tokens", "lemmas", "ne", "pos"])
    with open("duc2003.input.tsv", "w") as f:
        df.to_csv(f, sep="\t", index=False) 
 
