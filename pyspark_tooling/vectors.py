import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.types import ArrayType, IntegerType, BooleanType
from pyspark.ml.feature import HashingTF, IDF, Normalizer
from pyspark.ml.linalg import SparseVector


def term_frequency_vectors(
    input_col: str, output_col: str, df: DataFrame, num_features=262144, binary=False
):
    """Add the term frequency vectors as a new column"""
    hashing_tf = HashingTF(
        numFeatures=num_features,
        binary=binary,
        inputCol=input_col,
        outputCol=output_col,
    )

    # the resulting term frequency vectors will have one
    # entry for every token in the corpus where only a few
    # of those tokens will be active for any given document
    # for this reason they are representes as sparse vectors
    res = hashing_tf.transform(df)
    res.cache()
    return res


def tfidf_vectors(input_col: str, output_col: str, df: DataFrame, num_features=262144):
    """Calculate the tfidf vectors for the given input tokens"""
    # the invese document frequency can be calculated using
    # only the term frequency, essentially it is a column wise
    # operation over every term in the corpus
    idf = IDF(minDocFreq=0, inputCol=input_col, outputCol=output_col).fit(df)
    return idf.transform(df)


def normalize_vectors(input_col: str, output_col: str, df: DataFrame):
    """Normalize a column of vectors so they all have a magnitude of 1"""
    normalizer = Normalizer(inputCol=input_col, outputCol=output_col)
    return normalizer.transform(df)


def sparse_vector_indices(input_col: str, output_col: str, df: DataFrame):
    """Get the indices of the active elements in a sparse vector"""

    def _indices(a: SparseVector):
        if not isinstance(a, SparseVector):
            raise Exception("Expected ml sparse vector")
        return a.indices.tolist()

    func = F.udf(_indices, returnType=ArrayType(IntegerType()))
    return df.withColumn(output_col, func(F.col(input_col)))


def remove_zero_vectors(input_col: str, df: DataFrame):
    func = F.udf(lambda v: v.numNonzeros > 0, returnType=BooleanType())
    return df.filter(func(F.col(input_col)))
