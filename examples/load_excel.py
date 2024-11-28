import sys

import numpy as np
import pandas as pd

sys.path.append("../")
from configs.settings import BASE_DIR

file_path = BASE_DIR / "scripts" / "results.xlsx"

data = pd.read_excel(file_path, sheet_name="image_proc")
print("Image Data:\n", data)


def get_data(data, row="normal", column="ASR"):
    normal_rows = data[data.iloc[:, 0] == row]
    print(f"Rows with {row} in the first column:\n", normal_rows)

    acc_columns = data.columns[data.iloc[0] == column]
    print(f"Columns with {column} in the first row:\n", acc_columns)

    result_data = normal_rows[acc_columns]
    print("Result Data:\n", result_data)
    result_data = (
        result_data.apply(pd.to_numeric, errors="coerce")
        .dropna()
        .values.flatten()
        .tolist()
    )
    return result_data


metric = "BAC"
normal_list = get_data(data, row="normal", column=metric)
noise_list = get_data(data, row="noise", column=metric)
mislabel_list = get_data(data, row="mislabel", column=metric)
print("Normal List:", normal_list, sum(normal_list) / len(normal_list))
print("Noise List:", noise_list, sum(noise_list) / len(noise_list))
print("Mislabel List:", mislabel_list, sum(mislabel_list) / len(mislabel_list))


from scipy.stats import mannwhitneyu

# Mann-Whitney U test - normal vs data noise
stat1, p_value1 = mannwhitneyu(noise_list, normal_list, alternative="two-sided")
print(f"normal vs data noise: statistic value = {stat1}, p value = {p_value1}")

# Mann-Whitney U test - normal vs label noise
stat2, p_value2 = mannwhitneyu(mislabel_list, normal_list, alternative="two-sided")
print(f"normal vs label noise: statistic value = {stat2}, p value = {p_value2}")

alpha = 0.05
if p_value1 < alpha:
    print("normal vs data noise are significantly different")
else:
    print("normal vs data noise are not significantly different")

if p_value2 < alpha:
    print("normal vs label noise are significantly different")
else:
    print("normal vs label noise are not significantly different")
