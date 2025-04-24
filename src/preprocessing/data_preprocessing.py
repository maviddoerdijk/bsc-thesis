from typing import List, Tuple, Union
import numpy as np

def filter_pairs_data(pairs_data: List[Tuple[Tuple[str, str], Union[float, np.float64]]]):
  # 1. Remove all overly perfectly cointegrated pairs, based on a threshold.
  # In theory, the pairs with a 0.0 cointegration score should already be filtered out by preprocessing.cointegration.find_cointegrated_pairs
  step_1_filtered_pairs_data = [pair for pair in pairs_data if pair[1] != 0.0]

  # 2. Filter all times the two strings are the same
  step_2_filtered_pairs_data = [pair for pair in step_1_filtered_pairs_data if pair[0][0] != pair[0][1]]

  
  return step_2_filtered_pairs_data