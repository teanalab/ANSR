# coding=utf-8
import HTMLParser
import warnings

import nltk

__author__ = 'Saeid Balaneshin-kordan'
__email__ = "saeid@wayne.edu"
__date__ = 7 / 19 / 18

import pandas as pd
import numpy as np
import os
from bs4 import BeautifulSoup


class Prunner(object):
    """
    reads collection of fielded documents and prune those that are not judged in the qrels.
    """

    def __init__(self, qrels_filename):
        self.qrels = pd.read_csv(qrels_filename, delimiter=' ', names=["qno", "nl", "docname", "rel"])

    @staticmethod
    def parse_trec(filename, docnames):
        """
        parse a fielded file with trec (or xml-like) format that contains three fields of title, body and meta
        :param filename: the file that may contain multiple documents with trec format trec
        :param docnames: list of documents to be kept
        :return: a dictionary of extracted documents with value as the fields
        """
        docs = dict()
        with open(filename, 'r') as f:
            s = f.read()
            try:
                soup = BeautifulSoup(s, 'html.parser')
                for d in soup.findAll("doc"):
                    docno = ""
                    try:
                        docno = d.docno.get_text()
                        if docno not in docnames:
                            continue

                        title, body, meta = [""] * 3
                        try:
                            title = d.findAll("title")
                            title = ' '.join(map(lambda m_: m_.get_text(), title))
                        except KeyError:
                            pass
                        try:
                            body = d.findAll("body")
                            body = ' '.join(map(lambda m_: m_.get_text(), body))
                        except KeyError:
                            pass
                        try:
                            meta = d.findAll("meta")
                            meta = ' '.join(map(lambda m_: m_['content'], meta))
                        except KeyError:
                            pass

                        txt = [title, body, meta]

                        for i, t in enumerate(txt):
                            try:
                                t = map(lambda x: x.lower().encode('utf-8'), nltk.word_tokenize(t))
                                txt[i] = filter(lambda v: v.replace('-', '').isalpha(), t)
                            except TypeError:
                                txt[i] = ""
                        docs[docno] = txt
                    except AttributeError:
                        warnings.warn("The following document has parsing issues:" + docno)
            except HTMLParser.HTMLParseError:
                warnings.warn("The following document has parsing issues:" + filename)
        return docs

    def parse_qrels_documents(self):
        """
        parse documents that are in the qrels either judged as relevant or non-relevant
        :return: parsed documents
        """
        docs = dict()
        files_to_read = set()
        doc_names = np.unique(self.qrels['docname'])
        for d in doc_names:
            d = d.split('-')
            p = os.path.join("/backup/data/gov/data/", d[0], d[1])
            files_to_read.add(p)
        for i, p in enumerate(files_to_read):
            docs.update(self.parse_trec(p, doc_names))
            print(len(doc_names), len(files_to_read), i, len(docs))
            # if i > 10:
            #     break
        return docs

    def run(self):
        """
            main function
        """
        docs = self.parse_qrels_documents()
        with open('../../../mfnn-data/govs_mod.csv', 'w') as f:
            for k, v in docs.iteritems():
                f.write(','.join([k] + [' '.join(v_) for v_ in v]) + '\n')


if __name__ == "__main__":
    prunner = Prunner("../../configs/qrels/gov.txt")
    prunner.run()
