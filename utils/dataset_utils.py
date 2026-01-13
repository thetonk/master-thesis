# Copyright (C) 2025 Spyridon Baltsas
# This file is part of the research project "Cyberattack detection on network level using state-of-the-art deep learning models"
# Licensed under the GNU General Public License v3.0 (GPLv3)
# See the LICENSE file in the project root for full license text.

import os
import hashlib
#import random
import sys
import argparse
import ipaddress
import torch
from torch.utils.data import TensorDataset
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import numpy as np

#np.random.seed(42)

FEATURE_COLUMNS = (
    "Src IP","Src Port","Dst IP","Dst Port","Protocol","Timestamp","Flow Duration","Tot Fwd Pkts","Tot Bwd Pkts","TotLen Fwd Pkts","TotLen Bwd Pkts",
    "Fwd Pkt Len Max","Fwd Pkt Len Min","Fwd Pkt Len Mean","Fwd Pkt Len Std","Bwd Pkt Len Max","Bwd Pkt Len Min","Bwd Pkt Len Mean","Bwd Pkt Len Std","Flow Byts/s",
    "Flow Pkts/s","Flow IAT Mean","Flow IAT Std","Flow IAT Max","Flow IAT Min","Fwd IAT Tot","Fwd IAT Mean","Fwd IAT Std","Fwd IAT Max","Fwd IAT Min","Bwd IAT Tot",
    "Bwd IAT Mean","Bwd IAT Std","Bwd IAT Max","Bwd IAT Min","Fwd PSH Flags","Bwd PSH Flags","Fwd URG Flags","Bwd URG Flags","Fwd Header Len","Bwd Header Len",
    "Fwd Pkts/s","Bwd Pkts/s","Pkt Len Min","Pkt Len Max","Pkt Len Mean","Pkt Len Std","Pkt Len Var","FIN Flag Cnt","SYN Flag Cnt","RST Flag Cnt","PSH Flag Cnt","ACK Flag Cnt",
    "URG Flag Cnt","CWE Flag Count","ECE Flag Cnt","Down/Up Ratio","Pkt Size Avg","Fwd Seg Size Avg","Bwd Seg Size Avg","Fwd Byts/b Avg","Fwd Pkts/b Avg","Fwd Blk Rate Avg",
    "Bwd Byts/b Avg","Bwd Pkts/b Avg","Bwd Blk Rate Avg","Subflow Fwd Pkts","Subflow Fwd Byts","Subflow Bwd Pkts","Subflow Bwd Byts","Init Fwd Win Byts","Init Bwd Win Byts",
    "Fwd Act Data Pkts","Fwd Seg Size Min","Active Mean","Active Std","Active Max","Active Min","Idle Mean","Idle Std","Idle Max","Idle Min"
)

def _prepare_numeric_columns(df: pd.DataFrame, label_column = "Label") -> pd.DataFrame:
    non_numeric_columns = ["Src IP", "Dst IP", "Timestamp", label_column]
    for column in df.columns:
        if column not in non_numeric_columns and not pd.api.types.is_numeric_dtype(df[column]):
            df[column] = pd.to_numeric(df[column], errors="coerce", downcast="float")
    return df


class CSVDataset():
    def __init__(self, dataset_path, label_column, feature_columns=FEATURE_COLUMNS,
                 columns_to_drop=["Timestamp"], chunk_size=1e+6):
        self._chunk_size = chunk_size
        self._label_column = label_column.replace("_", " ")
        self._columns_to_drop = columns_to_drop
        self.dataset_path = dataset_path
        self.categories = None
        self.n_classes = None
        self.n_rows = None
        self.X = None
        self.y = None
        df = pd.read_csv(dataset_path, nrows=0)
        df.columns = df.columns.str.replace("_", " ")
        self._columns_to_drop = self._columns_to_drop + df.columns.difference(list(feature_columns)).to_list()
        df = df.drop(columns=self._columns_to_drop, errors="ignore")
        self.features = df.columns.to_list()
        self.n_features = len(self.features)

    
    @staticmethod
    def _read_csv_in_chunks(file_path, columns_to_drop, chunk_size=1e+6):
        # convert all numeric data from float64 to float32, save memory, as model uses float32
        cols = [col for col in FEATURE_COLUMNS if col not in columns_to_drop]
        dtypes = {
            c: "float32"
            for c in cols
        }
        with pd.read_csv(file_path, chunksize=chunk_size, low_memory=False, dtype=dtypes, usecols=cols, delimiter=",") as csv_reader:
            for chunk in csv_reader:
                chunk: pd.DataFrame
                chunk.columns = chunk.columns.str.replace("_", " ")
                #chunk = _prepare_numeric_columns(chunk)
                print("Chunk shape:",chunk.shape, "Datatypes:", chunk.dtypes)
                yield chunk
        del dtypes, cols

    
    def load(self, balance_classes=True, rows_limit=500e+3):
        rows_limit = int(rows_limit)
        label_column = self._label_column
        tensors = []
        for chunk in CSVDataset._read_csv_in_chunks(self.dataset_path, self._columns_to_drop, chunk_size=self._chunk_size):
            tensors.append(torch.from_numpy(chunk.to_numpy(dtype="float32")))
        x_tensor = torch.cat(tensors, dim=0)
        y_df = pd.read_csv(self.dataset_path, delimiter=",", usecols=[label_column], dtype={label_column: "category"})
        labels: pd.Series = y_df[label_column]
        del y_df
        y_tensor = torch.from_numpy(labels.cat.codes.to_numpy(dtype="int64"))
        if balance_classes:
            num_classes = len(labels.cat.categories)
            minimum_class_samples = labels.cat.codes.value_counts().min()
            required_samples_per_class = int(rows_limit / num_classes)
            if rows_limit > 1:
                rows_limit_per_class = min(minimum_class_samples, required_samples_per_class)
            else:
                rows_limit_per_class = minimum_class_samples
            if rows_limit_per_class < required_samples_per_class:
                print(f"Warning: dataset {self.dataset_path} minority class has less samples than the required! Has: {minimum_class_samples} samples!", file=sys.stderr)
            indexes = []
            for category in labels.cat.categories:
                indexes += labels.index[labels == category].to_series().sample(n=rows_limit_per_class).to_list()
            # shuffle indexes to mix samples of each class
            #random.shuffle(indexes)
            self.X = x_tensor[indexes]
            self.y = y_tensor[indexes]
        else:
            if 0 < rows_limit < x_tensor.shape[0]:
                sss = StratifiedShuffleSplit(n_splits=1, test_size=rows_limit)
                _, indexes = next(sss.split(x_tensor, y_tensor))
                self.X = x_tensor[indexes]
                self.y = y_tensor[indexes]
            else:
                self.X = x_tensor
                self.y = y_tensor
        del x_tensor, y_tensor
        self.categories = dict(enumerate(labels.cat.categories))
        self.n_classes = len(self.categories)
        self.n_rows = self.X.shape[0]


class LoadedTensorDataset():
    def __init__(self, dataset: TensorDataset, n_rows: int, n_features: int, categories: dict, feature_names, dtype=torch.float32):
        self.dataset = dataset
        self.num_rows = n_rows
        self.num_features = n_features
        self.categories = categories
        self.num_classes = len(categories)
        self.feature_names = feature_names
        self.dtype = dtype


def load_datasets_from_dir(dataset_dir, label_column: str, drop_columns: list | None = None,
                           rows_per_dataset: int | None = None, total_rows_limit: int=None, balance_classes: bool= False,
                           as_tensors_list: bool=False) -> LoadedTensorDataset | list[LoadedTensorDataset]:
    # assume that all datasets have same amount of classes and have same label column and features
    # first pass, discover num of classes and feature names
    dataset_list = []
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            filename_path = os.path.join(root, file)
            dataset_list.append(filename_path)
    dataset_list.sort(key=lambda filename: os.path.getsize(filename))
    num_datasets = len(dataset_list)
    print("Number of datasets found: ", num_datasets)
    if drop_columns is None:
        drop_columns = ["Timestamp"]
    csv_dataset = CSVDataset(dataset_list[0], label_column, columns_to_drop=drop_columns, chunk_size=3e+6)
    csv_dataset.load(balance_classes=balance_classes, rows_limit=10)
    dataset_categories = csv_dataset.categories
    feature_names = csv_dataset.features
    num_features = csv_dataset.X.shape[1]
    if rows_per_dataset is None:
        rows_per_dataset = int(total_rows_limit / num_datasets)
    # second pass, merge datasets into a large single dataset
    x = None
    y = None
    tensor_datasets = []
    for i, dataset_path in enumerate(dataset_list):
        print(f"Loading dataset {i+1} ({dataset_path})...")
        csv_dataset = CSVDataset(dataset_path, label_column, columns_to_drop=drop_columns, chunk_size=3e+6)
        csv_dataset.load(balance_classes=balance_classes, rows_limit=rows_per_dataset)
        if as_tensors_list:
            dataset = TensorDataset(csv_dataset.X, csv_dataset.y)
            tensor_datasets.append(LoadedTensorDataset(dataset, csv_dataset.X.shape[0], csv_dataset.X.shape[1],
                                                       csv_dataset.categories, feature_names, csv_dataset.X.dtype))
        else:
            if x is None:
                x, y = csv_dataset.X, csv_dataset.y
            else:
                x = torch.cat((x, csv_dataset.X), dim=0)
                y = torch.cat((y, csv_dataset.y), dim=0)
        del csv_dataset
        print("Done!")
    if as_tensors_list:
        return tensor_datasets
    else:
        dataset = TensorDataset(x, y)
        num_rows = x.shape[0]
        dtype = x.dtype
        del x, y
        return LoadedTensorDataset(dataset, num_rows, num_features, dataset_categories, feature_names, dtype)


def prepare_few_shot_train_test(train_dataset: TensorDataset, test_dataset: TensorDataset,
                                samples_per_class: int, class_values):
    x_test_initial = test_dataset.tensors[0]
    y_test_initial = test_dataset.tensors[1]
    x = train_dataset.tensors[0]
    y = train_dataset.tensors[1]
    # shuffle test set so infused samples are random each time
    random_indices = torch.randperm(x_test_initial.shape[0])
    x_test_initial = x_test_initial[random_indices]
    y_test_initial = y_test_initial[random_indices]
    del random_indices
    infused_indices_list = []
    for class_value in class_values:
        infused_samples_indexes = (y_test_initial == class_value).nonzero(as_tuple=True)[0][:samples_per_class]
        infused_indices_list += [int(i) for i in infused_samples_indexes]
        del infused_samples_indexes
    x_infused = x_test_initial[infused_indices_list]
    y_infused = y_test_initial[infused_indices_list]
    test_indices_mask = torch.ones(y_test_initial.shape[0], dtype=torch.bool)
    test_indices_mask[infused_indices_list] = False
    x_test = x_test_initial[test_indices_mask]
    y_test = y_test_initial[test_indices_mask]
    x = torch.cat((x, x_infused), dim=0)
    y = torch.cat((y, y_infused), dim=0)
    test_dataset = TensorDataset(x_test, y_test)
    del x_test, y_test, x_infused, y_infused, infused_indices_list, test_indices_mask
    print(f"Final # of rows after infusion: {x.shape[0]}")
    train_dataset = TensorDataset(x, y)
    del x_test_initial, y_test_initial, x, y
    return train_dataset, test_dataset


def merge_cicflow_csvs(csvs_directory, merged_csv_path, label_column="Label", chunk_size=1e+6, benign_label=None, multiclass=False):
    label_column = label_column.replace("_", " ")
    dataset_columns = list(FEATURE_COLUMNS) + [label_column]
    missing_fields = ["Src IP", "Src Port", "Dst IP"]
    total_dropped_lines = 0
    header_inserted = False
    sums = None
    counts = None
    print("="*50,"FIRST PASS","="*50)
    for root, _ ,files in os.walk(csvs_directory):
        for file in files:
            csv_file_path = os.path.join(root, file)
            print("Merging file {} to {}...".format(csv_file_path, merged_csv_path))
            with pd.read_csv(csv_file_path, chunksize=chunk_size, low_memory=False, delimiter=",") as csv_reader:
                for chunk in csv_reader:
                    chunk: pd.DataFrame
                    chunk.columns = chunk.columns.str.replace("_", " ")
                    # Remove rows that are identical to the header
                    chunk = chunk[chunk.apply(lambda row: not all(str(row[col]) == col for col in chunk.columns), axis=1)]

                    # drop unnamed columns
                    chunk = chunk.loc[:, ~chunk.columns.str.contains('^Unnamed')]

                    # drop all columns not in the final dataset list
                    chunk = chunk.drop(columns=chunk.columns.difference(dataset_columns))

                    if benign_label is not None:
                        if multiclass:
                            # chunk[label_column] = chunk[label_column].str.replace(r".*scan.*|reconnaissance|analysis", "Scanner", regex=True, case=False)
                            # chunk[label_column] = chunk[label_column].str.replace(r"mirai|okiru|.*c&c.*|torii|bot|botnet", "Botnet", regex=True, case=False)
                            # chunk[label_column] = chunk[label_column].str.replace(r".*d?dos.*|flood", "DoS", regex=True, case=False)
                            # chunk[label_column] = chunk[label_column].str.replace(r"heartbeat|exploits|sql injection|shellcode|fuzzers|infilteration", "Exploit", regex=True, case=False)
                            # chunk[label_column] = chunk[label_column].str.replace(r".*brute ?force.*|sparta", "Brute force", regex=True, case=False)
                            # chunk[label_column] = chunk[label_column].str.replace(r"attack|.*generic.*|theft", "Generic", regex=True, case=False)
                            # chunk[label_column] = chunk[label_column].str.replace(r"worms?|.*download.*|backdoor", "Infection", regex=True, case=False)
                            # chunk[label_column] = chunk[label_column].str.replace(r".*mitm.*", "MITM", regex=True, case=False)
                            s = chunk[label_column].str.lower()
                            conditions = [
                                s.str.contains(r".*scan.*|reconnaissance|analysis", regex=True),
                                s.str.contains(r"mirai|okiru|.*c&c.*|torii|bot|botnet", regex=True),
                                s.str.contains(r".*d?dos.*|flood", regex=True),
                                s.str.contains(r"heartbeat|exploits|sql injection|shellcode|fuzzers|infilteration", regex=True),
                                s.str.contains(r".*brute ?force.*|sparta", regex=True),
                                s.str.contains(r"attack|.*generic.*|theft", regex=True),
                                s.str.contains(r"worms?|.*download.*|backdoor", regex=True),
                                s.str.contains(r".*mitm.*", regex=True),
                            ]
                            del s
                            choices = [
                                "Scanner",
                                "Botnet",
                                "DoS",
                                "Exploit",
                                "Brute force",
                                "Generic",
                                "Infection",
                                "MITM",
                            ]
                            chunk[label_column] = np.select(
                                conditions,
                                choices,
                                default=chunk[label_column]
                            )
                        else:
                            print("Replacing non benign traffic labels to 'Attack' label!")
                            chunk.loc[chunk[label_column] != benign_label, label_column] = "Attack"
                    
                    chunk[label_column] = chunk[label_column].replace({"Benign": "Normal"})
                    # drop rows with dst port and protocol equal to 0
                    bad_rows = chunk[(chunk['Protocol'] == 0) & (chunk['Dst Port'] == 0) & (chunk[label_column] == "Normal")]
                    if not bad_rows.empty:
                        bad_row_count = len(bad_rows)
                        total_dropped_lines += bad_row_count
                        print(f"Dropping {bad_row_count} Benign traffic rows with Protocol=0 and Dst Port=0!")
                    chunk = chunk.drop(bad_rows.index)

                    for field in missing_fields:
                        if field not in chunk:
                            if field == "Src Port":
                                # Use ephemeral port range if missing
                                chunk[field] = np.random.randint(49152, 65536, chunk.shape[0])
                            else:
                                chunk[field] = 0
                            chunk[field] = chunk[field].astype(object)

                    chunk[label_column] = chunk[label_column].astype(object)
                    # Convert IP addresses to numbers
                    try:
                        chunk["Src IP"] = chunk["Src IP"].apply(lambda ip: int(ipaddress.ip_address(ip)))
                        chunk["Dst IP"] = chunk["Dst IP"].apply(lambda ip: int(ipaddress.ip_address(ip)))
                    except ValueError:
                        chunk["Src IP"] = chunk["Src IP"].astype(int)
                        chunk["Dst IP"] = chunk["Src IP"].astype(int)
                    chunk = chunk.reindex(columns=dataset_columns)

                    # Convert numeric data to corresponding numeric pandas datatype
                    chunk = _prepare_numeric_columns(chunk, label_column=label_column)

                    print("Chunk shape:",chunk.shape, "Datatypes:", chunk.dtypes)
                    numeric_chunk = chunk.select_dtypes(include="number")

                    numeric_chunk.replace([np.inf, -np.inf], np.nan, inplace=True)    
                    chunk_sums = numeric_chunk.sum(skipna=True)
                    chunk_counts = numeric_chunk.count()
                    if sums is None or counts is None:
                        sums = chunk_sums
                        counts = chunk_counts
                    else:
                        sums += chunk_sums
                        counts += chunk_counts
                    if label_column != "Label":
                        chunk = chunk.rename(columns={label_column: "Label"})
                    if not header_inserted:
                        chunk.to_csv(merged_csv_path, index=False, header=True, mode="w")
                        header_inserted = True
                    else:
                        chunk.to_csv(merged_csv_path, index=False, header=False, mode="a")
            print(f"Done!")
        
    print("="*40,"SECOND PASS, REPLACING NaN VALUES WITH MEANS","="*40)
    tmp_merged_file = merged_csv_path+".tmp"
    means = sums / counts
    dataset_columns[-1] = "Label"
    label_column = "Label"
    for i,chunk in enumerate(pd.read_csv(merged_csv_path, delimiter=",", chunksize=chunk_size)):
        # Convert numeric data to corresponding numeric pandas datatype
        chunk = _prepare_numeric_columns(chunk, label_column=label_column)
        numeric_cols = chunk.select_dtypes(include="number").columns
        chunk[numeric_cols] = chunk[numeric_cols].replace([np.inf, -np.inf], np.nan)
        for col in numeric_cols:
            if pd.notna(means[col]):
                chunk[col] = chunk[col].fillna(means[col])
            else:
                print(f"Warning! {col} has a mean of NaN! Replacing NaNs of this column with 0!")
                chunk[col] = chunk[col].fillna(0)
        chunk.to_csv(tmp_merged_file, mode="a", header=(i == 0), index=False)

    os.remove(merged_csv_path)
    print("Removing duplicates, please wait...")
    # Memory efficient way to remove duplicates
    with open(tmp_merged_file, "r") as f_in, open(merged_csv_path, "w") as f_out:
        unique_hashes = set()
        for line in f_in:
            line_hash = hashlib.md5(line.encode()).digest()
            if line_hash not in unique_hashes:
                unique_hashes.add(line_hash)
                f_out.write(line)
        del unique_hashes
    os.remove(tmp_merged_file)
    print(f"CSV dataset merge completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("dataset_folder", type=str, help="Dataset folder containing CSV files")
    parser.add_argument("merged_dataset_path", type=str, help="Path to save the merged CSV dataset")
    parser.add_argument("label_column", type=str, help="The name of the column that will be used as class.", default="Label")
    parser.add_argument("benign_label", type=str, help="The label of benign samples", default="Normal")
    parser.add_argument("-m", "--multiclass", action="store_true", help="Use multiclass labeling instead of binary")
    args = parser.parse_args()
    dataset_folder = args.dataset_folder
    merged_dataset_path = args.merged_dataset_path
    label_column = args.label_column
    benign_label = args.benign_label
    use_multiclass = args.multiclass
    merge_cicflow_csvs(dataset_folder, merged_dataset_path, label_column, 2e+6,
                       benign_label, use_multiclass)

