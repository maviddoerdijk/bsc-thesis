from typing import List, Tuple
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import os
from backtesting.utils import calculate_return_uncertainty

def plot_heatmap(pvalues: pd.DataFrame) -> None:
    sns.heatmap(pvalues,  cmap='RdYlGn_r', mask = (pvalues >= 0.98))
    plt.show()
    
def plot_return_uncertainty(S1, S2, spread_pred_series, test_index, look_back, # Note: look_back is not actually used in the function
                            position_thresholds=None, clearing_thresholds=None,
                            long_windows=None, short_windows=None, verbose=False, result_dir="", filename_base="data_begindate_enddate_hash.pkl"):
    returns_array, param_type = calculate_return_uncertainty(
        S1, S2, spread_pred_series,
        position_thresholds, clearing_thresholds,
        long_windows, short_windows, return_for_plotting=True
    )
    mean_returns = returns_array.mean(axis=0)
    std_returns = returns_array.std(axis=0)
    # due to mean_returns having prepended a single day at the beginning, the shapes are mismatched. Therefore, also prepend a day to time_axis_series
    extra_date = test_index[0] - pd.Timedelta(days=1)
    time_axis_series_prepended = pd.Index([extra_date]).append(test_index)

    plt.figure(figsize=(10, 6))
    plt.plot(time_axis_series_prepended, mean_returns, label='Mean Strategy Returns')
    plt.fill_between(time_axis_series_prepended, mean_returns - std_returns, mean_returns + std_returns,
                     alpha=0.3, label='±1 Std Dev')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Return')
    if param_type == "thresholds":
        plt.title(f"Trading Strategy Returns - position threshold ({min(position_thresholds):.2f}-{max(position_thresholds):.2f}), "
                  f"clearing threshold ({min(clearing_thresholds):.2f}-{max(clearing_thresholds):.2f})")
    else:
        plt.title(f"Trading Strategy Returns - short window ({min(short_windows)}-{max(short_windows)}), "
                  f"long window ({min(long_windows)}-{max(long_windows)})")
    plt.legend()
    filename_base_empty = filename_base.replace(".pkl", "")
    filename = f"{filename_base_empty}_plot_{param_type}.png"
    filepath = os.path.join(result_dir, filename)
    plt.savefig(filepath)
    plt.close()  # prevent automatic display in notebooks
    if verbose:
        print(f"Saved plot to {filepath}")
    return filename

def plot_comparison(y_true, y_hat, index, workflow_type="Unknown Workflow Type", pair_tup_str="(?,?)", result_dir="", filename_base="data_begindate_enddate_hash.pkl", verbose=False):
    y_hat = y_hat[:len(y_true)]
    index = index[:len(y_true)]
    y_hat = pd.Series(y_hat, index=index)
    y_true = pd.Series(y_true, index=index)

    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label='Actual Spread')
    plt.plot(y_hat, label='Predicted Spread')
    plt.xlabel('Time')
    plt.ylabel('Spread')
    plt.title(f'Actual vs Predicted Spread for {workflow_type} with pair {pair_tup_str}')
    plt.legend(["Actual Spread", "Predicted Spread"])

    filename_base_empty = filename_base.replace(".pkl", "")
    filename = f"{filename_base_empty}_groundtruth_comparison.png"
    filepath = os.path.join(result_dir, filename)
    plt.savefig(filepath)
    plt.close()  # prevent automatic display in notebooks
    if verbose:
        print(f"Saved plot to {filepath}")
    return filename

def plot_train_val_loss(train_losses, val_losses, workflow_type="Unknown Workflow Type", pair_tup_str="(?,?)", result_dir="", verbose=False, filename_base="data_begindate_enddate_hash.pkl"):
  plt.figure(figsize=(10, 6))
  plt.plot(train_losses, label="Train Loss")
  plt.plot(val_losses, label="Val Loss")
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.title(f"Train and Validation Loss for {workflow_type} with pair {pair_tup_str}")
  plt.legend()

  filename_base_empty = filename_base.replace(".pkl", "")
  filename = f"{filename_base_empty}_train_val_loss.png"
  filepath = os.path.join(result_dir, filename)
  plt.savefig(filepath)
  plt.close()  # prevent automatic display in notebooks
  if verbose:
      print(f"Saved plot to {filepath}")
  return filename

def results_to_latex(results):
    headers = [
        "Pair",
        "Cointegration Score",
        "val MSE",
        "test MSE",
        "YoY Returns (std)",
        "\makecell{Theoretical Return\\\\Under Perfect\\\\Information}",
        "Return Score"
    ]
    # Latex column alignment: l for first col, c for others
    align_str = "l" + "c" * (len(headers)-1)
    # Begin building latex table string
    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\resizebox{\\textwidth}{!}{")
    lines.append("\\begin{tabular}{" + align_str + "}")
    lines.append("\\toprule")
    lines.append(" & ".join(headers) + " \\\\")
    lines.append("\\midrule")
    for idx, row in enumerate(results):
        row_out = []
        for col_idx, cell in enumerate(row):
            # Add numbering for pairs
            if col_idx == 0:
                cell = f"{idx+1}. {cell}"
            # Format cointegration score as scientific in latex
            elif col_idx == 1 and isinstance(cell, float):
                base, exp = f"{cell:.2e}".split("e")
                exp = int(exp)
                cell = f"${base}\\times 10^{{{exp}}}$"
            # Theoretical return: show as percent if small, otherwise keep as float
            elif col_idx == 5 and isinstance(cell, float):
                cell = f"{cell*100:.2f}\\%"
                if "-100" in cell:
                  cell = "TLOE*"
            # Format YoY Returns (std) as $a\% \pm b\%$
            elif col_idx == 4 and isinstance(cell, str) and "%" in cell:
                # Convert e.g. '-82.63% +- 30.20%' to latex: $-82.63\% \pm 30.20\%$
                cell = cell.replace("%", "\\%")
                cell = cell.replace("+-", "\\pm")
                cell = f"${cell}$"
                if "-100" in cell:
                  cell = "TLOE*"
            elif col_idx == 6 and isinstance(cell, float):
              cell = f"{cell:.2f}"
            # General float formatting
            elif isinstance(cell, float):
                cell = f"{cell:.5f}"
            # Replace % in any string field (needed for e.g. theoretical return if not float)
            elif isinstance(cell, str) and "%" in cell:
                cell = cell.replace("%", "\\%")
            row_out.append(cell)
        # Join and add row
        lines.append(" & ".join(str(x) for x in row_out) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("}")
    lines.append("\\caption{Model performance and return statistics for all tested pairs.}")
    lines.append("\\end{table}")
    return "\n".join(lines)
