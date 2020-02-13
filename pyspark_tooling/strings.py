from unidecode import unidecode

from pyspark.sql import DataFrame, Column, WindowSpec
import pyspark.sql.functions as F


def count_str_length(original_col: str, updated_col: str, df: DataFrame):
    """Count the length of a string column and save to a new column"""
    return df.withColumn(updated_col, F.length(original_col))


def select_longest_string(col_name: str, w: WindowSpec, df: DataFrame):
    """Create a new dataframe containing only"""
    # create temporary columns fro storing the ranking and length data
    length_col = "length_count_temp"
    rank_col = "length_rank_temp"

    # add a column with the length of each string
    with_lengths = count_str_length(col_name, length_col, df)

    # note that the ordering is primarily by the length of the string but in the
    # case of a tie then the alphabetical ordering will be used to settle the tie.
    # This is important because we need one winner per partition and therefore if
    # there are strings that are different but have the same length we must rank them alphabetically
    window = w.orderBy(F.col(length_col).desc(), col_name)

    # filter out any rows that do not have the max length
    # remove the intermediary columns for counting and ranking
    # drop duplicates to ensure that only one winner remains
    return (
        with_lengths.withColumn(rank_col, F.dense_rank().over(window))
        .filter(F.col(rank_col) == 1)
        .drop(length_col, rank_col)
        .dropDuplicates()
    )


def alphanumeric(col: Column):
    """Return the column containing only lowercase alphanumeric characters (no spaces or punctuation)"""
    return F.regexp_replace(col, r"[^\w]", "")


def replace_non_alphabet_with_spaces(col: Column):
    """Remove any char that is not either alphanumeric or a space with a space"""
    return F.regexp_replace(col, r"[^a-zA-Z\s]", " ")


def alphanumeric_lowercase(col: Column):
    """Return the column containing only lowercase alphanumeric characters (no spaces or punctuation)"""
    return F.lower(F.regexp_replace(col, r"[^\w]", ""))


def remove_multiple_spaces(col: Column):
    """Replace multiple spaces with single spaces"""
    return F.trim(F.regexp_replace(col, " +", " "))


def trim_leading_zeros(col: Column):
    """Trim the leading zeros from a string column"""
    return F.trim(F.regexp_replace(col, "^0*", " "))


def remove_punctuation(col: Column):
    """Remove all punctuation except . % and / leaving only alphanumerics and spaces"""
    return F.regexp_replace(col, r"[^a-zA-Z\d\s/%\.]", "")


def remove_text_in_brackets(col: Column) -> Column:
    """Remove any text inside () or [] brackets"""
    r = col
    # handle round brackets and square brackets separately just to be sure
    for expr in [r"\([^\(\)]*\)", r"\[[^\[\]]*\]"]:
        r = F.regexp_replace(r, expr, "")
    return r


def remove_stop_words(stop_words: list, col: Column):
    """Remove all stop words from list"""
    expr = [f"\\b{word}\\b" for word in stop_words]
    regex = "|".join(expr)
    return F.regexp_replace(col, regex, "")


def remove_full_stops_not_decimal_places(col: Column):
    """Remove full stops without removing decimal places"""
    exp = r"(?<!\d)\.(?!\d)"
    # exp = '(\\d+\\.\\d+)|[.]'
    return F.regexp_replace(col, exp, "")


def remove_n_a(col: Column) -> Column:
    """Remove any instance of n/a"""
    return F.regexp_replace(col, r"[n]\s*[/]\s*[a]", "")


def replace_foward_slashes_with_spaces(col: Column):
    """Replace forward slashes with spaces"""
    return F.regexp_replace(col, "/", " ")


def remove_special_chars(col: Column):
    """Remove all special unicode characters in a string"""

    def cleaner(input_str: str):
        if input_str:
            return unidecode(input_str)
        return input_str

    cleaner_udf = F.udf(cleaner)
    return cleaner_udf(col)


def remove_external_brackets(col: Column):
    """Remove any instances of outer brackets wrapping the text"""

    def cleaner(text: str):
        # If ingredient column contains a value, otherwise not need to do anything
        if text is not None:
            if len(text) > 0:
                if ((text[0] == "[") and (text[-1] == "]")) or (
                    (text[0] == "(") and (text[-1] == ")")
                ):
                    return text[1:-1]
            return text

    cleaner_udf = F.udf(cleaner)
    return cleaner_udf(F.trim(col))


def remove_leading_trailing_double_quotes_except_inches(col: Column) -> Column:
    """Remove any leading and trailing double quote(") with null string except for any double quote used for inch(es)"""
    return F.regexp_replace(col, '^"|(?<![0-9])"$', "")


def replace_consecutive_double_quotes(col: Column) -> Column:
    """replace any two consecutive double quotes("") with single double quote(")"""
    return F.regexp_replace(col, '""', '"')


def remove_double_quote_both_sides_of_number(col: Column) -> Column:
    """replace single double quote(") with null string except for any double quote used for inch(es)"""
    return F.regexp_replace(col, '"([0-9]+)"', "$1")


def remove_double_quote_except_inches(col: Column) -> Column:
    """replace single double quote(") with null string except for any double quote used for inch(es)"""
    return F.regexp_replace(col, '(?<![0-9])"', "")


def trim(col: Column):
    """Trim a string column"""
    return F.trim(col)


def lowercase(col: Column):
    return F.lower(col)
