# Copyright (C) 2025 Spyridon Baltsas
# This file is part of the research project "Cyberattack detection on network level using state-of-the-art deep learning models"
# Licensed under the GNU General Public License v3.0 (GPLv3)
# See the LICENSE file in the project root for full license text.

import os
import hashlib
import sys
import argparse
import ipaddress
import numpy as np
import torch
from torch.utils.data import TensorDataset
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import polars as pl

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

def get_dropped_columns(model_type, unified_removal=False) -> list[str]:
    from utils.models import ModelTypes
    dropped_columns = ["Timestamp"]
    if not unified_removal:
        if model_type == ModelTypes.TRANSFORMER:
            dropped_columns += ["Src IP", "Dst IP", "Idle Mean", "Idle Min", "Idle Max"]
        elif model_type == ModelTypes.LSTM:
            dropped_columns += ["Fwd Seg Size Min"]
        else:
            dropped_columns += ["Src IP", "Dst IP", "Idle Mean", "Idle Min", "Idle Max", "Idle Std"]
    else:
        dropped_columns += ["Src IP", "Dst IP", "Fwd Seg Size Min", "Init Bwd Win Byts",
                                    "Idle Mean", "Idle Min", "Idle Max"]
    return dropped_columns

def _prepare_numeric_columns(df: pd.DataFrame, label_column = "Label") -> pd.DataFrame:
    non_numeric_columns = ["Src IP", "Dst IP", "Timestamp", label_column]
    for column in df.columns:
        if column not in non_numeric_columns and not pd.api.types.is_numeric_dtype(df[column]):
            df[column] = pd.to_numeric(df[column], errors="coerce", downcast="float")
    return df


class CSVDataset():
    def __init__(self, dataset_path, label_column, feature_columns=FEATURE_COLUMNS,
                 columns_to_drop=["Timestamp"], chunk_size=1e+6):
        self._chunk_size = int(chunk_size)
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
    def _read_csv_in_batches(file_path, columns_to_drop, chunk_size=int(1e+6)):
        # convert all numeric data from float64 to float32, save memory, as model uses float32
        cols = [col for col in FEATURE_COLUMNS if col not in columns_to_drop]
        lazy_df = (pl.scan_csv(file_path, separator=",", try_parse_dates=False, low_memory=False)
                   .select(cols)
                   .rename(lambda c: c.replace("_", " "))
                   .with_columns(pl.all().cast(pl.Float32)))
        for batch in lazy_df.collect_batches(chunk_size=chunk_size):
            print("Schema: ", batch.schema)
            yield batch
        del cols


    def load(self, balance_classes=True, rows_limit=int(500e+3), normal_bias=False):
        label_column = self._label_column
        self.n_rows = pl.scan_csv(self.dataset_path).select(pl.len()).collect().item()
        x_tensor = torch.empty(size=(self.n_rows, self.n_features), dtype=torch.float32)
        offset = 0
        for chunk in CSVDataset._read_csv_in_batches(self.dataset_path, self._columns_to_drop, self._chunk_size):
            n = len(chunk)
            x_tensor[offset:offset+n] = torch.from_numpy(chunk.to_numpy())
            offset += n
        assert offset == x_tensor.shape[0]
        y_lazy_df = (pl.scan_csv(self.dataset_path, separator=",", low_memory=False, try_parse_dates=False)
                     .select(label_column)
                     .cast({label_column: pl.Categorical})
                     .with_row_index("idx")
                     )
        y_df = y_lazy_df.collect()
        labels = y_df.select(label_column).to_series()
        y_tensor = torch.from_numpy(labels.to_physical().cast(pl.Int64).to_numpy())
        categories = labels.cat.get_categories()
        self.categories = dict(enumerate(categories))
        minimum_class_samples = labels.to_physical().value_counts(name="n").select(pl.col("n").min()).item()
        del labels
        self.n_classes = len(self.categories)
        if balance_classes:
            num_classes = len(categories)
            required_samples_per_class = int(rows_limit / num_classes)
            if rows_limit > 1:
                rows_limit_per_class = min(minimum_class_samples, required_samples_per_class)
            else:
                rows_limit_per_class = minimum_class_samples
            if rows_limit_per_class < required_samples_per_class:
                print(f"Warning: dataset {self.dataset_path} minority class has less samples than the required! Has: {minimum_class_samples} samples!", file=sys.stderr)
            indexes = []
            for category in categories:
                print(category)
                samples = rows_limit_per_class
                if normal_bias and category == "Normal":
                    samples = int(8 * rows_limit_per_class)
                    print("Applying normal sample bias! Normal samples: ", samples)
                indexes += (y_df.filter(pl.col(label_column) == category).select("idx").to_series().sample(n=samples).to_list())
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


def infuse_samples(src_dataset: TensorDataset, dst_dataset: TensorDataset,
                                samples_per_class: int, class_values):
    x_dst_initial = dst_dataset.tensors[0]
    y_dst_initial = dst_dataset.tensors[1]
    x = src_dataset.tensors[0]
    y = src_dataset.tensors[1]
    dst_samples = x_dst_initial.shape[0]
    requested_samples = len(class_values)*samples_per_class
    if (dst_samples < requested_samples):
        print("Warning: Destination dataset has less samples than requested! Requested: ", requested_samples, "available: ", dst_samples)
    # shuffle test set so infused samples are random each time
    random_indices = torch.randperm(x_dst_initial.shape[0])
    x_dst_initial = x_dst_initial[random_indices]
    y_dst_initial = y_dst_initial[random_indices]
    del random_indices
    infused_indices_list = []
    for class_value in class_values:
        infused_samples_indexes = (y_dst_initial == class_value).nonzero(as_tuple=True)[0][:samples_per_class]
        infused_indices_list += [int(i) for i in infused_samples_indexes]
        del infused_samples_indexes
    x_infused = x_dst_initial[infused_indices_list]
    y_infused = y_dst_initial[infused_indices_list]
    dst_indices_mask = torch.ones(y_dst_initial.shape[0], dtype=torch.bool)
    dst_indices_mask[infused_indices_list] = False
    x_dst = x_dst_initial[dst_indices_mask]
    y_dst = y_dst_initial[dst_indices_mask]
    x = torch.cat((x, x_infused), dim=0)
    y = torch.cat((y, y_infused), dim=0)
    dst_dataset = TensorDataset(x_dst, y_dst)
    del x_dst, y_dst, x_infused, y_infused, infused_indices_list, dst_indices_mask
    print(f"Final # of rows after infusion: {x.shape[0]}")
    src_dataset = TensorDataset(x, y)
    del x_dst_initial, y_dst_initial, x, y
    return src_dataset, dst_dataset


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


def extract_class_from_datasets(csv_directory: os.PathLike, label_column: str, extracted_class: str, leftover_csv_dataset: os.PathLike):
    print(f"Extracting class {extracted_class} to separate dataset!")
    #header_inserted = False
    extracted_csv_dataset = os.path.join(os.path.dirname(leftover_csv_dataset), f"{extracted_class.lower()}_extracted.csv")
    for root, _ ,files in os.walk(csv_directory):
        for file in files:
            csv_dataset = os.path.join(root, file)
            lazy_df = pl.scan_csv(csv_dataset, separator=",", try_parse_dates=False)
            lazy_df = lazy_df.select([c for c in lazy_df.columns if not c.startswith("Unnamed")])
            extracted_lazy = lazy_df.filter(pl.col(label_column) == extracted_class)
            leftover_lazy = lazy_df.filter(pl.col(label_column) != extracted_class)
            # Stream to CSV (lazy execution, memory-efficient)
            extracted_lazy.sink_csv(extracted_csv_dataset)
            leftover_lazy.sink_csv(leftover_csv_dataset)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("dataset_folder", type=str, help="Dataset folder containing CSV files")
    parser.add_argument("output_dataset_path", type=str, help="Path to save the unified processed CSV dataset")
    parser.add_argument("label_column", type=str, help="The name of the column that will be used as class.", default="Label")
    parser.add_argument("benign_label", type=str, help="The label of benign samples", default="Normal")
    parser.add_argument("-m", "--multiclass", action="store_true", help="Use multiclass labeling instead of binary")
    parser.add_argument("-x", "--extract-class", type=str, help="Class name to extract to separate file")
    args = parser.parse_args()
    dataset_folder = args.dataset_folder
    output_dataset_path = args.output_dataset_path
    label_column = args.label_column
    benign_label = args.benign_label
    use_multiclass = args.multiclass
    if args.extract_class is None:
        merge_cicflow_csvs(dataset_folder, output_dataset_path, label_column, 2e+6,
                        benign_label, use_multiclass)
    else:
        extract_class_from_datasets(dataset_folder, label_column, args.extract_class, output_dataset_path)

