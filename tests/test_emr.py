import pytest

from tests import base
from grada_pyspark_utils.emr import InfrastructureConfig


# @pytest.mark.focus
class TestEmrDeployment(base.BaseTest):
    @pytest.mark.focus
    def test_calc_required_resources(self):

        conf = InfrastructureConfig(minimum_memory_in_gb=100)

        # the actual minimum memory should include a 10% bonus for yarn
        assert conf.total_minimum_memory == 110
        # this can be satisfied by
        assert conf.instance_type == "r5.4xlarge"
