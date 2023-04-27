import json
from enum import Enum
from collections import OrderedDict
from pathlib import Path
import pandas as pd

class FCDEF(Enum):
    ENCODER = 0x0
    DECODER = 0x1