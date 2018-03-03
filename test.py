import numpy as np
import pandas as pd

d = {'x-col': [0, 1, 3, 4], 'y-col': [42, 35, 23, 23]}
l = pd.DataFrame(d)

print(l.pow(2))