from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

import numpy as np
import pandas as pd

ArrayLike1D: TypeAlias = Sequence[float] | np.ndarray | pd.Series
DataFrameLike: TypeAlias = pd.DataFrame
MetricsDict: TypeAlias = dict[str, float]
