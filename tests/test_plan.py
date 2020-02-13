# import pytest

from grada_pyspark_utils import plan
from tests import base


# @pytest.mark.focus
class TestPlanUtils(base.BaseTest):
    def test_parse_plan(self):

        with open("./tests/sample_plan.txt") as f:
            txt = f.read()
        res = plan.parse_plan(txt)
        assert (len(res)) == 4
        keys = [i[0] for i in res.items()]
        assert keys[0] == "physical"
        assert keys[1] == "optimized"
        assert keys[2] == "analyzed"
        assert keys[3] == "logical"

        for _, v in res.items():
            assert len(v) > 0
