import pytest
import logging
import json
import uuid
from time import time
from datetime import datetime
from dateutil import parser


class TestLogger:
    @pytest.mark.usefixtures("caplog", "logger")
    def test_debug_log(self, caplog, logger):

        with caplog.at_level(logging.DEBUG):
            msg = f"test log msg: {str(uuid.uuid4())}"

            t = datetime.utcnow()
            logger.debug(msg)

            for record in caplog.records:
                raw = json.loads(record.msg)
                assert raw["env"] == "prod"
                assert raw["level"] == "debug"
                assert raw["service"] == "tooling_test"
                assert raw["event"] == msg
                self.validate_timestamp(raw["timestamp"], t)

    @pytest.mark.usefixtures("caplog", "logger")
    def test_error_log(self, caplog, logger):

        with caplog.at_level(logging.WARN):
            msg = f"test err msg: {str(uuid.uuid4())}"
            t = datetime.utcnow()

            logger.error(msg)

            for record in caplog.records:
                raw = json.loads(record.msg)
                assert raw["env"] == "prod"
                assert raw["level"] == "error"
                assert raw["service"] == "tooling_test"
                assert raw["event"] == msg
                self.validate_timestamp(raw["timestamp"], t)

    def validate_timestamp(self, actual: str, expected: time):
        """Validate times are equal to the nearest millisecond"""
        actual_time = parser.parse(actual).replace(tzinfo=None)
        a = actual_time.isoformat(sep=" ", timespec="milliseconds")
        e = expected.isoformat(sep=" ", timespec="milliseconds")
        assert a == e
