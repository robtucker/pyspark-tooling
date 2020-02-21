import spacy as _spacy
import pyspark.sql.functions as F
from pyspark.sql import SQLContext, DataFrame
from pyspark.ml.linalg import Vectors
from spacy.tokens import Doc
from typing import List

from pyspark_tooling.dataframe import to_tuples
from pyspark_tooling.logger import log


DOCUMENT_ID = "document_id"


def get_spacy_docs(
    document_id_col: str,
    document_text_col: str,
    df: DataFrame,
    spacy_model_version="en_core_web_lg",
):
    """Retrieve the spacy docs as a dataframe. Note that this is done in the driver"""
    log.info("initate spacy pipeline")

    # select both the document id (can be a row number for instance)
    # as well as the raw document text
    raw = to_tuples(df.select(F.col(document_id_col), F.col(document_text_col)))
    # load spacy
    nlp = _spacy.load(spacy_model_version)
    # each entry is a tuple of (text, context) where the context is a dictionary
    raw_texts = [(a if isinstance(a, str) else "", {DOCUMENT_ID: b}) for a, b in raw]

    # use the spacy pipe method to process all the docs at once
    docs = nlp.pipe(raw_texts)

    for doc, context in docs:
        # set the id as an "extension attribute" on each doc object
        doc._.document_id = context[DOCUMENT_ID]

    return docs


def extract_document_vectors(docs: List[Doc]):
    return [
        (
            doc._.get(DOCUMENT_ID),
            (Vectors.dense(doc.vector.tolist()) if len(doc) > 0 else None),
        )
        for doc in docs
    ]


def with_spacy_document_vectors(
    spark: SQLContext,
    document_id_col: str,
    document_text_col: str,
    output_col: str,
    df: DataFrame,
):
    """A high level method to turn a column containing text into spacy document vectors"""
    # convert the documents to spacy objects
    docs = get_spacy_docs(document_id_col, document_text_col, df)
    # extract the document vectors as python lists
    document_vectors = extract_document_vectors(docs)
    # create a new dataframe from the python lists
    vectors_df = spark.createDataFrame(document_vectors, [document_id_col, output_col],)
    # join the new dataframe back onto the original dataframe
    return df.join(vectors_df, how="inner", on=[document_id_col])
