from typing import List, Tuple
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as pyplot

def plot_heatmap(pvalues: pd.DataFrame) -> None:
    sns.heatmap(pvalues,  cmap='RdYlGn_r', mask = (pvalues >= 0.98))
    pyplot.show()