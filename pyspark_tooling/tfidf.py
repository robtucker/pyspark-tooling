from cytoolz import pipe, partial
from pyspark.sql import DataFrame

from pyspark_tooling.tokens import (
    fill_nulls_with_empty_string,
    rm_empty_strings_from_tokens,
    tokenize_words,
    character_ngrams,
)
from pyspark_tooling.vectors import (
    term_frequency_vectors,
    tfidf_vectors,
    normalize_vectors,
    sparse_vector_indices,
)
from pyspark_tooling.dataframe import drop_cols


def token_vectors_pipeline(
    input_col: str, output_col: str, df: DataFrame, stemmer_func=None
):
    """Convert a string into an array of integer token ids"""
    filled_col = input_col + "_filled"
    tokenised_col = input_col + "_tokenised"
    tf_vectors = input_col + "_tf_vectors"

    transforms = [
        # note that the tokenizer completely breaks given null input values
        partial(fill_nulls_with_empty_string, input_col, filled_col),
        partial(tokenize_words, filled_col, tokenised_col),
    ]

    # optionally stem the tokens
    if stemmer_func:
        transforms += [partial(stemmer_func, tokenised_col, tokenised_col)]

    transforms += [
        partial(rm_empty_strings_from_tokens, tokenised_col, tokenised_col),
        partial(term_frequency_vectors, tokenised_col, tf_vectors),
        partial(sparse_vector_indices, tf_vectors, output_col),
        partial(drop_cols, [filled_col, tokenised_col, tf_vectors]),
    ]
    return pipe(df, *transforms)


def tf_ngrams_pipeline(input_col: str, output_col: str, df: DataFrame, n=3):
    """Calculate the term frequency vectors for the character-wise trigrams of an input string"""
    filled_col = input_col + "_filled"
    trigrams_col = input_col + "_character_ngrams"

    return pipe(
        df,
        partial(fill_nulls_with_empty_string, input_col, filled_col),
        partial(character_ngrams, filled_col, trigrams_col, n=n),
        partial(rm_empty_strings_from_tokens, trigrams_col, trigrams_col),
        partial(term_frequency_vectors, trigrams_col, output_col),
        partial(drop_cols, [filled_col, trigrams_col]),
    )


def tfidf_vectors_pipeline(input_col: str, output_col: str, df: DataFrame, n=3):
    """Calculate the tfidf vectors for the character-wise trigrams of an input string"""
    tf_vectors = input_col + "_tf_vectors"
    tfidf_col = input_col + "_tfifd_vectors"

    return pipe(
        tf_ngrams_pipeline(input_col, tf_vectors, df),
        partial(tfidf_vectors, tf_vectors, tfidf_col),
        partial(normalize_vectors, tfidf_col, output_col),
        partial(drop_cols, [tf_vectors, tfidf_col]),
    )
