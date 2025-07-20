# -*- encoding: utf-8 -*-
"""Python 3.11+ utilities"""

# Standard library imports
from urllib.parse import urlparse, urlencode
from urllib.request import urlopen, build_opener
from urllib.request import HTTPPasswordMgrWithDefaultRealm
from urllib.request import HTTPBasicAuthHandler
from configparser import ConfigParser
from io import StringIO
from queue import Queue
from functools import reduce

# Type aliases for consistency
string_type = str
binary_type = bytes
text_type = str
int_types = (int,)

def to_str(b):
    """Convert bytes to string"""
    if isinstance(b, bytes):
        return b.decode("utf-8")
    return str(b)
