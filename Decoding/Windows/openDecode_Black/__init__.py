__verion__ = '2.0'

from .config import Para
from . import preprocess as pp
from . import segmentation as seg
from . import registration as reg
from . import decoding as dc
from . import matrixization as mtx
from .spotdetection import runSpotiflow

from .main import OpenDecoder

from . import explorer


import logging
from ._logging import configure_logger

log = logging.getLogger("openDecode_Black")
configure_logger(log)