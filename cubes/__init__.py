"""OLAP Cubes"""

__version__ = "1.1"

# Fix grako compatibility with Python 3.11+
import sys
import collections
if sys.version_info >= (3, 3):
    import collections.abc
    # Monkey-patch collections.Mapping and MutableMapping for grako compatibility
    if not hasattr(collections, 'Mapping'):
        collections.Mapping = collections.abc.Mapping
    if not hasattr(collections, 'MutableMapping'):
        collections.MutableMapping = collections.abc.MutableMapping

from .common import *
from .query import *
from .metadata import *
from .workspace import *
from .errors import *
from .formatters import *
from .mapper import *
from .calendar import *
from .auth import *
from .logging import *
from .namespace import *
