import os
import re
import corenlp as cnlp
from datetime import datetime
import pandas as pd

stopwords = set()
with open(u"stopwords.en.txt", "rb") as f:
    for line in f:
        stopwords.add(line.strip())

def load_inputs(docset_iter, pipeline, data):

    for docset in docset_iter:
        print("Docset ID: {}".format(docset.docset_id))
        for doc in docset.input_iter():
            print("\tInput ID: {}".format(doc.doc_id))
            text = bytes(doc)
            m = re.search(r"<TEXT>(.*)</TEXT>", text, re.DOTALL)
            text = m.group(1) 
            text = re.sub(r"</?P>", "", text)
            text = re.sub(
                r"<ANNOTATION>.*?</ANNOTATION>", "", text, flags=re.DOTALL)
            text_ann = pipeline.annotate(text)
            for i, sent in enumerate(text_ann, 1):
                datum = {}
                datum["docset id"] = docset.docset_id
                datum["doc id"] = doc.doc_id
                datum["timestamp"] = doc.timestamp
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

def load_model(docset_iter, pipeline, data):

    for docset in docset_iter:
        print("Docset ID: {}".format(docset.docset_id))
        for doc in docset.model_iter():
            print("\tModel ID: {}".format(doc.doc_id))
            text_ann = pipeline.annotate(bytes(doc))

            for i, sent in enumerate(text_ann, 1):
                datum = {}
                datum["docset id"] = docset.docset_id
                datum["doc id"] = doc.doc_id
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
 
from sumpy.util import DUCHelper
dh = DUCHelper()

annotators=["tokenize", "ssplit", "pos", "lemma", "ner"]
with cnlp.Server(annotators=annotators) as pipeline:

    data_model_2003 = []
    load_model(dh.docset_iter(2003, 2), pipeline, data_model_2003)
    df = pd.DataFrame(
        data_model_2003,
        columns=[
            "docset id", "doc id", "sent id", "text", 
            "word length", "byte length", "tokens", "lemmas", "ne"])
    with open("duc2003.models.tsv", "w") as f:
        df.to_csv(f, sep="\t", index=False) 

    data_input_2003 = []
    load_inputs(dh.docset_iter(2003, 2), pipeline, data_input_2003)
    df = pd.DataFrame(
        data_input_2003, 
        columns=[
            "docset id", "doc id", "timestamp", "sent id", "text", 
            "word length", "byte length", "tokens", "lemmas", "ne", "pos"])
    with open("duc2003.inputs.tsv", "w") as f:
        df.to_csv(f, sep="\t", index=False) 
    
    data_model_2004 = []
    load_model(dh.docset_iter(2004, 2), pipeline, data_model_2004)
    df = pd.DataFrame(
        data_model_2004, 
        columns=[
            "docset id", "doc id", "sent id", "text", 
            "word length", "byte length", "tokens", "lemmas", "ne"])
    with open("duc2004.models.tsv", "w") as f:
        df.to_csv(f, sep="\t", index=False) 

    data_input_2004 = []
    load_inputs(dh.docset_iter(2004, 2), pipeline, data_input_2004)
    df = pd.DataFrame(
        data_input_2004, 
        columns=[
            "docset id", "doc id", "timestamp", "sent id", "text", 
            "word length", "byte length", "tokens", "lemmas", "ne", "pos"])
    with open("duc2004.inputs.tsv", "w") as f:
        df.to_csv(f, sep="\t", index=False) 
 
