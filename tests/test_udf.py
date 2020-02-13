import pytest

from app.nlp import udf_funcs
from tests import base


# @pytest.mark.focus
@pytest.mark.nlp
class TestUDFs(base.BaseTest):
    def test_set_intersection(self):

        example1 = udf_funcs.set_intersection(["a", "b", "c"], ["b", "c", "d"])
        assert set(example1) == {"c", "b"}

        example2 = udf_funcs.set_intersection(["x", "y", "z"], ["b", "", None])
        assert set(example2) == set()

        example3 = udf_funcs.set_intersection([None, "", ""], ["a", "b", ""])
        assert example3 is None

        example4 = udf_funcs.set_intersection(None, ["a", "b", ""])
        assert example4 is None
