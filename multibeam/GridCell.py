from pathlib import Path

import numpy as np
from pandas.core.nanops import nanargmax

from tool.Tool import read_grid

BASE_DIR = Path(__file__).resolve().parent
data_path = BASE_DIR.parent / "data" / "data.xlsx"

x, y, z = read_grid(data_path)
print(x, y, z)

z_max = np.max(z)
z_max_idx = nanargmax(z)
index_x, index_y = np.unravel_index(z_max_idx, z.shape)
print(f"z_max: {z_max}")
print(f"z_max_idx: {z_max_idx}")
print(f"index_x: {index_x}, index_y: {index_y}")

