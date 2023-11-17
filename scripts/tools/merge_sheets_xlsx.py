# %%
import pandas as pd
import os.path as osp
import os
import sys
import numpy as np

# Define the Excel file path
file_path = "exp-grit-all.csv.xlsx"
if len(sys.argv) > 1:
    file_path = sys.argv[1]
print(f"file_path: {file_path}")

# List the sheet names you want to merge
sheets_to_merge = [
    "scores-ciderd.csv.xlsx",
    "scores-meteor.csv.xlsx",
    "scores-spice.csv.xlsx",
    "scores-bleu.csv.xlsx",
    "scores-rouge.csv.xlsx",
    "content.csv.xlsx",
    "clip.csv.xlsx",
]

# Read the first sheet and store it in a DataFrame
merged_data = pd.read_excel(file_path, sheet_name=sheets_to_merge[0])
# Remove empty columns
merged_data = merged_data.dropna(axis=1, how="all")
merged_data.insert(loc=0, column="BaseLogPath", value=merged_data["LogPath"].apply(lambda x: osp.dirname(x)))


# %%
# Iterate through the rest of the sheets and append them to the merged_data DataFrame
def merge_and_check_df(merged_data, sheet, data, check="both"):
    # NOTE: check the docs
    merged_data = merged_data.merge(
        data,
        left_on=["BaseLogPath"],
        right_on=["BaseLogPath"],
        how="left",
        validate="1:1",
        indicator=True,
        suffixes=("", f"_{sheet}"),
    )
    # Check indicator column is all "both", and remove the column
    if isinstance(check, str):
        check = [check]
    if not isinstance(check, list):
        raise ValueError(f"check should be str or list, but got {type(check)}")
    merge_check = merged_data["_merge"] == check[0]
    if len(check) > 1:
        for c in check[1:]:
            merge_check = merge_check | (merged_data["_merge"] == c)
    assert all(merge_check)
    print(f"merge check: {check} has passed.")
    merged_data = merged_data.drop(columns=["_merge"])
    return merged_data


for sheet in sheets_to_merge[1:]:
    try:
        data = pd.read_excel(file_path, sheet_name=sheet)
        # Remove empty columns
        data = data.dropna(axis=1, how="all")
        data.insert(loc=0, column="BaseLogPath", value=data["LogPath"].apply(lambda x: osp.dirname(x)))
    except Exception as e:
        print(f"Error: {e}")
        continue
    print(f"sheet: {sheet}, len: {len(data)}")

    if sheet == "clip.csv.xlsx":
        assert len(data) % 2 == 0
        data_1 = data.iloc[: len(data) // 2]
        data_2 = data.iloc[len(data) // 2 :]
        merged_data = merge_and_check_df(merged_data, f"{sheet}_pred", data_1, check=["both", "left_only"])
        merged_data = merge_and_check_df(merged_data, f"{sheet}_gt", data_2, check=["both", "left_only"])
    else:
        merged_data = merge_and_check_df(merged_data, sheet, data, check="both")

# Preview the merged DataFrame
print(merged_data.head())

# %%
# Save the merged DataFrame to a new Excel file
# merged_data.to_excel('merged_data.xlsx', index=False)
# Save the merged DataFrame to a new Excel file with the header (column names)
# with pd.ExcelWriter("merged_data.xlsx", engine="openpyxl", mode="w") as writer:
#     merged_data.to_excel(writer, index=False)

# %%
# Remove empty columns
merged_data = merged_data.dropna(axis=1, how="all")

# Remove columns whose names start with "Max" or "Min"
merged_data = merged_data.loc[:, ~merged_data.columns.str.startswith(("Max", "Min"))]

# Remove all the columns starts with "LogPath" or "Dataset"
logpath_dataset_mask = merged_data.columns.str.startswith(("LogPath", "Dataset"))
dataset_mask = merged_data.columns.str.startswith(("Dataset"))
first_datset_index = np.where(dataset_mask == True)[0][0]
logpath_dataset_mask[first_datset_index] = False
merged_data = merged_data.loc[:, ~logpath_dataset_mask]


# Remove the suffix
def remove_suffix(value):
    if isinstance(value, str):
        if "+/-" in value:
            return value.split("+/-")[0].strip()
        elif "±" in value:
            return value.split("±")[0].strip()
    return value


try:
    # NOTE: FutureWarning: DataFrame.applymap has been deprecated. Use DataFrame.map instead.
    merged_data = merged_data.map(remove_suffix)
except Exception as e:
    merged_data = merged_data.applymap(remove_suffix)

for column_name in merged_data.columns:
    try:
        # if the first obj can be convert to number, we convert all of them
        merged_data[column_name] = pd.to_numeric(merged_data[column_name])
    except ValueError as e:
        continue

# Save the merged DataFrame to a new Excel file with the header (column names)
file_path_wo_ext, ext = osp.splitext(file_path)
save_file_path = f"{file_path_wo_ext}.merged{ext}"
with pd.ExcelWriter(save_file_path, engine="openpyxl", mode="w") as writer:
    merged_data.to_excel(writer, index=False)

print(f"save_file_path: {save_file_path}")

# %%
