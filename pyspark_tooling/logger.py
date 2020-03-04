import logging
import sys
import structlog
from structlog.stdlib import add_log_level, LoggerFactory, BoundLogger
from datetime import datetime


DEFAULT_LOGGER_NAME = "root"
DEFAULT_TIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"


# export the logger as a global
log = structlog.get_logger(DEFAULT_LOGGER_NAME)


def configure(
    env: str,
    service_name: str,
    level,
    processors=[],
    logger_name: str = DEFAULT_LOGGER_NAME,
):
    """Configure the structlogger
    Run this function with the desired params at application startup
    """
    _level = get_log_level(level)
    _processors = get_processors(env, service_name, _level, processors)

    # configure python's native logger
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=_level)

    structlog.configure(
        processors=_processors,
        logger_factory=LoggerFactory(),
        wrapper_class=BoundLogger,
        cache_logger_on_first_use=True,
    )

    global log
    log = structlog.get_logger(logger_name)
    return log


def add_service_name(service_name: str):
    """ Configure the service name processor"""

    def processor(_, __, event_dict):
        event_dict["service"] = service_name
        return event_dict

    return processor


def add_env(env: str):
    """Configure the env processor"""

    def processor(_, __, event_dict):
        event_dict["env"] = env
        return event_dict

    return processor


def add_duration_in_seconds():
    """Create a processor to log the total time elapsed of the process"""

    start = datetime.utcnow()

    def processor(_, __, event_dict):
        event_dict["duration"] = datetime.utcnow() - start
        return event_dict

    return processor


def get_processors(
    env: str,
    service_name: str,
    level: int,
    custom_processors=[],
    time_format=DEFAULT_TIME_FORMAT,
):
    """Create a list of processors including the required processors and the renderer"""

    # in prod-like environments use the json renderer
    renderer = structlog.processors.JSONRenderer()

    with_timestamp = structlog.processors.TimeStamper(fmt=time_format)

    _processors = [with_timestamp]

    # locally we can print out key values
    if env in ["local", "loc"]:
        renderer = structlog.processors.KeyValueRenderer()
    else:
        with_env = add_env(env)
        with_service_name = add_service_name(service_name)

        additional_processors = [add_log_level, with_service_name, with_env]

        _processors = _processors + additional_processors + custom_processors

    _processors.append(renderer)

    return _processors


def get_log_level(level):
    """Convert a string log level into a logging level"""

    # the user may already have inputted an integer log level
    if isinstance(level, int):
        # the level must not be negative
        if level < 0:
            return logging.NOTSET
        return level

    # otherwise it must be a string
    if not isinstance(level, str):
        raise ValueError("log level must be a string or an integer")

    _level = level.lower()

    if _level in ["debug", "dbg"]:
        return logging.DEBUG

    if _level in ["info", "inf"]:
        return logging.INFO

    if _level == "warn" or _level == "warning":
        return logging.WARN

    if _level == "error" or _level == "err":
        return logging.ERROR

    # if no level is set raise an exception
    raise ValueError("Received unkown log level: {0!s}".format(level))
