import pytest
from functools import partial

from pyspark.sql import Window

from pyspark_tooling import strings
from pyspark_tooling.dataframe import to_tuples, to_dicts

from tests import base


# @pytest.mark.focus
class TestStringUtils(base.BaseTest):
    @pytest.mark.usefixtures("spark")
    def test_string_length(self, spark):
        original_col = "column_a"
        updated_col = "length_column_a"

        data = [("a",), ("ab",), ("abc",), ("abcd",)]

        raw = spark.createDataFrame(data, [original_col])
        df = strings.count_str_length(original_col, updated_col, raw)
        assert df.count() == 4

        res = to_dicts(df.orderBy(updated_col))

        names = [i[original_col] for i in res]
        lengths = [i[updated_col] for i in res]

        assert lengths == [1, 2, 3, 4]
        assert names == ["a", "ab", "abc", "abcd"]

    @pytest.mark.usefixtures("spark")
    def test_select_longest_string(self, spark):

        # group the rows by the partiton col
        partition_col = "partition_col"

        # rank the rows by the aggregation col
        agg_col = "agg_col"

        # more than 1 string might have the same length
        # alphabetical ordering should be used to break the tie
        data = [
            ("A", "longest_a"),
            ("A", "longest_a"),  # duplicate
            ("A", "longest_b"),
            ("A", "short"),
            ("A", None),
            ("B", "longest_x"),
            ("B", "longest_x"),  # duplicate
            ("B", "longest_y"),
            ("B", "longest_y"),
            ("B", "short"),
            ("B", None),
            ("B", None),
            ("C", None),
            ("C", None),
        ]

        raw = spark.createDataFrame(data, [partition_col, agg_col])

        window = Window.partitionBy("partition_col")
        df = strings.select_longest_string("agg_col", window, raw).orderBy(
            "partition_col"
        )
        assert df.count() == 3
        res = to_tuples(df)
        expected = [["A", "longest_a"], ["B", "longest_x"], ["C", None]]

        for i, expected in enumerate(expected):
            actual = res[i]
            assert actual[0] == expected[0]
            assert actual[1] == expected[1]

    @pytest.mark.usefixtures("spark")
    def test_trim(self, spark):
        """Trim strings in the column"""
        data = [("abc ",), ("  defg",), ("  hij-k  ",)]
        expected = ["abc", "defg", "hij-k"]

        self._run_transform_test(spark, strings.trim, data, expected)

    @pytest.mark.usefixtures("spark")
    def test_lowercase(self, spark):
        """Convert all characters to lowercase"""
        data = [("ABC",), ("dEf",), ("xyZ",)]

        expected = ["abc", "def", "xyz"]

        self._run_transform_test(spark, strings.lowercase, data, expected)

    @pytest.mark.usefixtures("spark")
    def test_alphanumeric_only(self, spark):
        """Remove all non alphanumeric characters"""
        data = [
            ("abc, DEF",),
            ("a!@#b$%^c&*() \"def\" 'g'",),
            ("phrase with spaces",),
            ("pHrAsE  WiTh   mAnY    SpAcEs",),
        ]

        expected = ["abcDEF", "abcdefg", "phrasewithspaces", "pHrAsEWiThmAnYSpAcEs"]

        self._run_transform_test(spark, strings.alphanumeric, data, expected)

    @pytest.mark.usefixtures("spark")
    def test_replace_non_alphabet_with_spaces(self, spark):
        """Remove all non alphanumeric characters"""
        data = [
            ("a!@#b$%^c&*() \"def\" 'ghi' 123",),
            ("ph#r%as&e w!i@t$h 1*7 c^h:a;r's",),
        ]

        expected = ["a   b   c      def   ghi     ", "ph r as e w i t h     c h a r s"]

        self._run_transform_test(
            spark, strings.replace_non_alphabet_with_spaces, data, expected
        )

    @pytest.mark.usefixtures("spark")
    def test_alphanumeric_lowercase(self, spark):
        """Remove non alphanumerica characters and lowercase"""
        data = [("a!@#B$%^c&*() \"dEf\" 'G'",), ("pHraSe wiTh   MaNy    sP{aCes",)]

        expected = ["abcdefg", "phrasewithmanyspaces"]

        self._run_transform_test(spark, strings.alphanumeric_lowercase, data, expected)

    @pytest.mark.usefixtures("spark")
    def test_remove_multiple_spaces(self, spark):
        """ Convert multiple spaces to a single space"""
        data = [("  test phrase  with   multiple     spaces      ",)]

        expected = ["test phrase with multiple spaces"]

        self._run_transform_test(spark, strings.remove_multiple_spaces, data, expected)

    @pytest.mark.usefixtures("spark")
    def test_trim_leading_zeros(self, spark):
        """Remove any leading zeros from a string"""
        data = [("00001",), ("023",), ("00000456",), ("00abc",)]

        expected = ["1", "23", "456", "abc"]

        self._run_transform_test(spark, strings.trim_leading_zeros, data, expected)

    @pytest.mark.usefixtures("spark")
    def test_remove_punctuation(self, spark):
        """Remove all punctuation except % . and /"""

        data = [("f!u@l#l$ f^a&t* 5% y(o)g[u]r't; / 4:.5|g",)]
        expected = ["full fat 5% yogurt / 4.5g"]

        self._run_transform_test(spark, strings.remove_punctuation, data, expected)

    @pytest.mark.usefixtures("spark")
    def test_remove_text_in_brackets(self, spark):
        """Remove any text that is inside () or [] brackets - can create multiple spaces"""

        data = [
            ("hello (ignore me) world",),
            ("test [abc]123",),
            ("outer[random chars]text",),
            ("()",),
            ("[]",),
        ]
        expected = ["hello  world", "test 123", "outertext", "", ""]

        self._run_transform_test(spark, strings.remove_text_in_brackets, data, expected)

    @pytest.mark.usefixtures("spark")
    def test_remove_stop_words(self, spark):
        """Remove a list of predefined stop words - can create multiple spaces"""

        stop_words = ["a", "the", "with", "on", "of", "but"]
        data = [
            ("We are the knights who say ni",),
            ("Always look on the bright side of life",),
            ("Tis but a scratch",),
        ]

        expected = [
            "We are  knights who say ni",
            "Always look   bright side  life",
            "Tis   scratch",
        ]

        func = partial(strings.remove_stop_words, stop_words)
        self._run_transform_test(spark, func, data, expected)

    @pytest.mark.usefixtures("spark")
    def test_remove_stop_phrases(self, spark):
        """Remove a list of predefined stop phrases - can create multiple spaces"""

        phrases = [
            "for up to date ingredients",
            "active ingredients",
            "for allergens see capitalised ingredients",
            "allergy advice",
        ]

        data = [
            ("fat (1%) for up to date ingredients see reverse",),
            ("active ingredients salt (1g)",),
            ("sugar (2g) allergy advice nuts (3g)",),
        ]

        expected = ["fat (1%)  see reverse", " salt (1g)", "sugar (2g)  nuts (3g)"]

        func = partial(strings.remove_stop_words, phrases)
        self._run_transform_test(spark, func, data, expected)

    @pytest.mark.usefixtures("spark")
    def test_remove_full_stops_not_decimal_places(self, spark):
        """Remove full stops without affecting decimal places - can create multiple spaces"""
        data = [("Avocado. 41.2g",), ("Yoghurt. Full fat. 21.6cl",)]
        expected = ["Avocado 41.2g", "Yoghurt Full fat 21.6cl"]

        self._run_transform_test(
            spark, strings.remove_full_stops_not_decimal_places, data, expected
        )

    @pytest.mark.usefixtures("spark")
    def test_remove_n_a(self, spark):
        """Remove any instances of n/a - can create multiple spaces"""
        data = [
            ("Hello n/a world",),
            ("Hello world n / a",),
            ("Hello n /a world",),
            ("n/ a Hello world",),
        ]
        expected = ["Hello  world", "Hello world ", "Hello  world", " Hello world"]

        self._run_transform_test(spark, strings.remove_n_a, data, expected)

    @pytest.mark.usefixtures("spark")
    def test_replace_foward_slashes_with_spaces(self, spark):
        """Replace forward slashes - can create multiple spaces"""
        data = [("This/that",), ("Hello / world",)]
        expected = ["This that", "Hello   world"]

        self._run_transform_test(
            spark, strings.replace_foward_slashes_with_spaces, data, expected
        )

    @pytest.mark.usefixtures("spark")
    def test_remove_special_chars(self, spark):
        """Replace all non standard ascii characters"""
        data = [
            ("Ă",),
            ("Ā",),
            ("â",),
            ("à",),
            ("ä",),
            ("Ĕ",),
            ("Ę",),
            ("é",),
            ("è",),
            ("ê",),
            ("ë",),
            ("Ĩ",),
            ("ï",),
            ("ï",),
            ("î",),
            ("Ő",),
            ("ô",),
            ("ö",),
            ("Ŭ",),
            ("ù",),
            ("û",),
            ("ü",),
            ("ç",),
            ("ß",),
            ("",),
            (None,),
        ]

        expected = [
            "A",
            "A",
            "a",
            "a",
            "a",
            "E",
            "E",
            "e",
            "e",
            "e",
            "e",
            "I",
            "i",
            "i",
            "i",
            "O",
            "o",
            "o",
            "U",
            "u",
            "u",
            "u",
            "c",
            "ss",
            "",
            None,
        ]

        self._run_transform_test(spark, strings.remove_special_chars, data, expected)

    @pytest.mark.usefixtures("spark")
    def test_remove_external_brackets(self, spark):
        """Replace all non standard ascii characters"""
        data = [
            # should remove matching brackets
            (" (We are the knights who say ni) ",),
            (" [Always look on the bright side of life] ",),
            # should leave mis-matching brackets
            ("(Tis but a scratch]",),
            ("[What is the air-speed velocity of an unladen swallow?)",),
            # should leave inner brackets
            ("We used to live in one room (all hundred and twenty-six of us)",),
        ]
        expected = [
            "We are the knights who say ni",
            "Always look on the bright side of life",
            "(Tis but a scratch]",
            "[What is the air-speed velocity of an unladen swallow?)",
            "We used to live in one room (all hundred and twenty-six of us)",
        ]

        self._run_transform_test(
            spark, strings.remove_external_brackets, data, expected
        )

    @pytest.mark.usefixtures("spark")
    def test_remove_leading_trailing_double_quotes_except_inches(self, spark):
        """Remove any leading and trailing double quote(") with the empty string
        except for any double quote used for inch(es)"""
        data = [
            ('"Armand de Brignac Gold Brut Multi-Vintage Champagne 75cl 69"-70"',),
            ('"B&M Flexi Ruler 12""Colour Blue 1Each"',),
            ('"A Big Thank You" Elephant Notecards 10 per pack',),
        ]
        expected = [
            'Armand de Brignac Gold Brut Multi-Vintage Champagne 75cl 69"-70"',
            'B&M Flexi Ruler 12""Colour Blue 1Each',
            'A Big Thank You" Elephant Notecards 10 per pack',
        ]

        self._run_transform_test(
            spark,
            strings.remove_leading_trailing_double_quotes_except_inches,
            data,
            expected,
        )

    @pytest.mark.usefixtures("spark")
    def test_replace_consecutive_double_quotes(self, spark):
        """replace any two consecutive double quotes("") with single double quote(")"""
        data = [
            ('"Armand de Brignac Gold Brut Multi-Vintage Champagne 75cl 60"-70"',),
            ('"B&M Flexi Ruler 12""Colour Blue 1Each"',),
            ('"A Big Thank You" Elephant Notecards 10 per pack',),
        ]
        expected = [
            '"Armand de Brignac Gold Brut Multi-Vintage Champagne 75cl 60"-70"',
            '"B&M Flexi Ruler 12"Colour Blue 1Each"',
            '"A Big Thank You" Elephant Notecards 10 per pack',
        ]

        self._run_transform_test(
            spark, strings.replace_consecutive_double_quotes, data, expected
        )

    @pytest.mark.usefixtures("spark")
    def test_remove_double_quote_both_sides_of_number(self, spark):
        """replace single double quote(") with null string except for any double quote used for inch(es)"""
        data = [
            ('"Armand de Brignac Gold Brut Multi-Vintage Champagne 75cl 60"-70"',),
            ('"B&M Flexi Ruler "12" Colour Blue "1"Each"',),
            ('"A Big Thank You" Elephant Notecards 10 per pack',),
        ]
        expected = [
            '"Armand de Brignac Gold Brut Multi-Vintage Champagne 75cl 60"-70"',
            '"B&M Flexi Ruler 12 Colour Blue 1Each"',
            '"A Big Thank You" Elephant Notecards 10 per pack',
        ]

        self._run_transform_test(
            spark, strings.remove_double_quote_both_sides_of_number, data, expected
        )

    @pytest.mark.usefixtures("spark")
    def test_remove_double_quote_except_inches(self, spark):
        """replace single double quote(") with null string except for any double quote used for inch(es)"""
        data = [
            ('"Armand de Brignac Gold Brut Multi-Vintage Champagne 75cl 60"-70"',),
            ('"B&M Flexi Ruler "2" Colour Blue 1Each"',),
            ('"A Big Thank You" Elephant Notecards 10 per pack',),
        ]
        expected = [
            'Armand de Brignac Gold Brut Multi-Vintage Champagne 75cl 60"-70"',
            'B&M Flexi Ruler 2" Colour Blue 1Each',
            "A Big Thank You Elephant Notecards 10 per pack",
        ]

        self._run_transform_test(
            spark, strings.remove_double_quote_except_inches, data, expected
        )
