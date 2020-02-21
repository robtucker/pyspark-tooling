from nltk import stem

import pyspark.sql.functions as F
from pyspark.ml.feature import NGram, StopWordsRemover, Tokenizer
from pyspark.sql import DataFrame
from pyspark.sql.types import ArrayType, StringType

from pyspark_tooling.arrays import remove_empty_strings


def tokenize_words(input_col: str, output_col: str, df: DataFrame):
    """Tokenize the input col and save as a new column"""
    tokenizer = Tokenizer(inputCol=input_col, outputCol=output_col)
    return tokenizer.transform(df)


def string_to_character_array(input_col: str, output_col: str, df: DataFrame):
    """Convert a string into an array of characters"""
    func = F.udf(lambda x: [char for char in x], returnType=ArrayType(StringType()))
    return df.withColumn(output_col, func(F.col(input_col)))


def ngrams(input_col: str, output_col: str, df: DataFrame, n: int = 3) -> DataFrame:
    """Transform an array of tokens to ngrams and save as a new column"""
    transformer = NGram(inputCol=input_col, outputCol=output_col, n=n)
    return transformer.transform(df)


def character_ngrams(input_col: str, output_col: str, df: DataFrame, n: int = 3):
    func = F.udf(
        lambda x: [x[i : i + n] for i in range(len(x) - n + 1)]
        if isinstance(x, str)
        else None,
        returnType=ArrayType(StringType()),
    )
    return df.withColumn(output_col, func(F.col(input_col)))


def remove_stop_words(input_col: str, output_col: str, stop_words: list, df: DataFrame):
    """Remove all stop words from an array of tokens and save as a new column"""
    transformer = StopWordsRemover(
        inputCol=input_col, outputCol=output_col, stopWords=stop_words
    )
    return transformer.transform(df)


def sorted_tokens(input_col: str, output_col: str, df: DataFrame):
    """Sort the tokens alphabetically"""
    return df.withColumn(output_col, F.sort_array(F.col(input_col)))


def porter_tokens(input_col: str, output_col: str, df: DataFrame):
    """Stem the tokens using the porter stemmer"""
    stemmer = stem.porter.PorterStemmer()

    func = F.udf(
        lambda tokens: [stemmer.stem(word) for word in tokens],
        returnType=ArrayType(StringType()),
    )

    return df.withColumn(output_col, func(F.col(input_col)))


def lancaster_tokens(input_col: str, output_col: str, df: DataFrame):
    """Stem the tokens using the Lancaster stemmer"""
    stemmer = stem.lancaster.LancasterStemmer()

    func = F.udf(
        lambda tokens: [stemmer.stem(word) for word in tokens],
        returnType=ArrayType(StringType()),
    )

    return df.withColumn(output_col, func(F.col(input_col)))


def snowball_tokens(input_col: str, output_col: str, df: DataFrame):
    """Stem the tokens using the snowball stemmer"""
    stemmer = stem.snowball.EnglishStemmer()

    func = F.udf(
        lambda tokens: [stemmer.stem(word) for word in tokens],
        returnType=ArrayType(StringType()),
    )

    return df.withColumn(output_col, func(F.col(input_col)))


def fill_nulls_with_empty_string(input_col: str, output_col: str, df: DataFrame):
    """Fill nulls with empty string and save to a new column"""
    return df.withColumn(output_col, F.col(input_col)).fillna({output_col: ""})


def rm_empty_strings_from_tokens(input_col: str, output_col: str, df: DataFrame):
    """Remove empty strings from tokens"""
    return df.withColumn(output_col, remove_empty_strings(F.col(input_col)))
