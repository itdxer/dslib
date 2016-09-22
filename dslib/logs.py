import os
import sys
import time
import logging
from functools import wraps
from contextlib import contextmanager


def memoization(function):
    memo = {}

    @wraps(function)
    def wrapper(*args, **kwargs):
        hashed_kwargs = tuple(kwargs.items())
        arguments_hash = (args, hashed_kwargs)

        if arguments_hash in memo:
            return memo[arguments_hash]

        result = function(*args, **kwargs)
        memo[args] = result

        return result
    return wrapper


@memoization
def get_logger(level=logging.INFO):
    logging.basicConfig(
        format="[%(levelname)-8s:%(asctime)s] %(message)s",
        level=level,
        datefmt="%d/%m/%Y %H:%M:%S",
    )
    logger = logging.getLogger('root')
    return logger


logger = get_logger()


@contextmanager
def logtime(title):
    if not hasattr(logtime, 'last_id'):
        logtime.last_id = 0

    logtime.last_id += 1
    last_id = logtime.last_id

    logger = get_logger()
    logger.info("[start:{:0>3}] Start {}".format(last_id, title))
    t0 = time.time()

    yield

    total_time = time.time() - t0
    logger.info("[finish:{:0>3}] Finish {} (took {:.3f} sec)".format(
        last_id, title, total_time
    ))


def is_color_supported():
    """
    Returns ``True`` if the running system's terminal supports
    color, and ``False`` otherwise.

    Notes
    -----
    Code from Djano: https://github.com/django/django/blob/\
    master/django/core/management/color.py

    Returns
    -------
    bool
    """
    supported_platform = (
        sys.platform != 'Pocket PC' and
        (sys.platform != 'win32' or 'ANSICON' in os.environ)
    )

    # isatty is not always implemented
    is_a_tty = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
    is_support = (supported_platform and is_a_tty)
    return is_support


def create_style(ansi_code, use_bright_mode=False):
    """
    Create style based on ANSI code number.

    Parameters
    ----------
    ansi_code : int
        ANSI style code.

    Returns
    -------
    function
        Function that takes string argument and add ANDI styles
        if its possible.
    """
    def style(text):
        if is_color_supported():
            mode = int(use_bright_mode)
            return "\033[{};{}m{}\033[0m".format(mode, ansi_code, text)
        return text
    return style


red = create_style(ansi_code=91)
green = create_style(ansi_code=92)
yellow = create_style(ansi_code=93)
