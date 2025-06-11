import os
import sys
import ipaddress
import multiprocessing
import torch
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import numpy as np

#np.random.seed(42)

FEATURE_COLUMNS = [
    "Src IP","Src Port","Dst IP","Dst Port","Protocol","Timestamp","Flow Duration","Tot Fwd Pkts","Tot Bwd Pkts","TotLen Fwd Pkts","TotLen Bwd Pkts",
    "Fwd Pkt Len Max","Fwd Pkt Len Min","Fwd Pkt Len Mean","Fwd Pkt Len Std","Bwd Pkt Len Max","Bwd Pkt Len Min","Bwd Pkt Len Mean","Bwd Pkt Len Std","Flow Byts/s",
    "Flow Pkts/s","Flow IAT Mean","Flow IAT Std","Flow IAT Max","Flow IAT Min","Fwd IAT Tot","Fwd IAT Mean","Fwd IAT Std","Fwd IAT Max","Fwd IAT Min","Bwd IAT Tot",
    "Bwd IAT Mean","Bwd IAT Std","Bwd IAT Max","Bwd IAT Min","Fwd PSH Flags","Bwd PSH Flags","Fwd URG Flags","Bwd URG Flags","Fwd Header Len","Bwd Header Len",
    "Fwd Pkts/s","Bwd Pkts/s","Pkt Len Min","Pkt Len Max","Pkt Len Mean","Pkt Len Std","Pkt Len Var","FIN Flag Cnt","SYN Flag Cnt","RST Flag Cnt","PSH Flag Cnt","ACK Flag Cnt",
    "URG Flag Cnt","CWE Flag Count","ECE Flag Cnt","Down/Up Ratio","Pkt Size Avg","Fwd Seg Size Avg","Bwd Seg Size Avg","Fwd Byts/b Avg","Fwd Pkts/b Avg","Fwd Blk Rate Avg",
    "Bwd Byts/b Avg","Bwd Pkts/b Avg","Bwd Blk Rate Avg","Subflow Fwd Pkts","Subflow Fwd Byts","Subflow Bwd Pkts","Subflow Bwd Byts","Init Fwd Win Byts","Init Bwd Win Byts",
    "Fwd Act Data Pkts","Fwd Seg Size Min","Active Mean","Active Std","Active Max","Active Min","Idle Mean","Idle Std","Idle Max","Idle Min"
]

def _prepare_numeric_columns(df: pd.DataFrame, label_column = "Label") -> pd.DataFrame:
    non_numeric_columns = ["Src IP", "Dst IP", "Timestamp", label_column]
    for column in df.columns:
        if column not in non_numeric_columns and not pd.api.types.is_numeric_dtype(df[column]):
            df[column] = pd.to_numeric(df[column], errors="coerce", downcast="float")
    return df


class CSVDataset():
    def __init__(self, dataset_path, label_column, feature_columns=FEATURE_COLUMNS, chunk_size=1e+6):
        self._chunk_size = chunk_size
        self._label_column = label_column.replace("_", " ")
        self._columns_to_drop = ["Src IP", "Dst IP", "Timestamp"]
        self.dataset_path = dataset_path
        self.categories = None
        self.X = None
        self.y = None
        df = pd.read_csv(dataset_path, nrows=0)
        df.columns = df.columns.str.replace("_", " ")
        self._columns_to_drop = self._columns_to_drop + df.columns.difference(feature_columns).to_list()
        df = df.drop(columns=self._columns_to_drop, errors="ignore")
        self.features = df.columns.to_list()
    
    @staticmethod
    def _read_csv_in_chunks(file_path, columns_to_drop, chunk_size=1e+6):
        # convert all numeric data from float64 to float32, save memory, as model uses float32
        with pd.read_csv(file_path, chunksize=chunk_size, low_memory=False, delimiter=",") as csv_reader:
            for chunk in csv_reader:
                chunk.columns = chunk.columns.str.replace("_", " ")
                chunk = chunk.drop(columns=columns_to_drop, errors="ignore")
                chunk = _prepare_numeric_columns(chunk)
                chunk = chunk.astype("float32")
                print("Chunk shape:",chunk.shape, "Datatypes:", chunk.dtypes)
                yield chunk

    @staticmethod
    def _chunk_to_tensor(chunk: pd.DataFrame) -> torch.Tensor:
        return torch.tensor(chunk.to_numpy())
    
    def load(self):
        label_column = self._label_column
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            tensors = pool.map(CSVDataset._chunk_to_tensor, CSVDataset._read_csv_in_chunks(self.dataset_path, self._columns_to_drop, chunk_size=self._chunk_size))
        x_tensor = torch.cat(tensors, dim=0).float()
        y_df = pd.read_csv(self.dataset_path, delimiter=",", usecols=[label_column], dtype={label_column: "category"})
        y_tensor = torch.tensor(y_df[label_column].cat.codes.to_numpy(), dtype=torch.int64)
        rows_limit = int(500e+3)
        if x_tensor.shape[0] > rows_limit:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=rows_limit)
            _, indexes = next(sss.split(x_tensor, y_tensor))
            self.X = x_tensor[indexes]
            self.y = y_tensor[indexes]
        else:
            self.X = x_tensor
            self.y = y_tensor
        del x_tensor, y_tensor
        self.categories = dict(enumerate(y_df[label_column].cat.categories))


def merge_cicflow_csvs(csvs_directory, merged_csv_path, label_column="Label", chunk_size=1e+6, bin_benign_label=None):
    label_column = label_column.replace("_", " ")
    dataset_columns = FEATURE_COLUMNS + [label_column]
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
                    chunk.columns = chunk.columns.str.replace("_", " ")
                    # Remove rows that are identical to the header
                    chunk = chunk[chunk.apply(lambda row: not all(str(row[col]) == col for col in chunk.columns), axis=1)]

                    # drop unnamed columns
                    chunk = chunk.loc[:, ~chunk.columns.str.contains('^Unnamed')]

                    # drop all columns not in the final dataset list
                    chunk = chunk.drop(columns=chunk.columns.difference(dataset_columns))

                    if bin_benign_label is not None:
                        print("Replacing non benign traffic labels to 'Attack' label!")
                        chunk.loc[chunk[label_column] != bin_benign_label, label_column] = "Attack"

                    # drop rows with dst port and protocol equal to 0
                    bad_rows = chunk[(chunk['Protocol'] == 0) & (chunk['Dst Port'] == 0) & (chunk[label_column].isin(["Benign", "Normal"]))]
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
                    if sums is None:
                        sums = chunk_sums
                        counts = chunk_counts
                    else:
                        sums += chunk_sums
                        counts += chunk_counts
                    if not header_inserted:
                        chunk.to_csv(merged_csv_path, index=False, header=True, mode="w")
                        header_inserted = True
                    else:
                        chunk.to_csv(merged_csv_path, index=False, header=False, mode="a")
            print(f"Done!")
        
    print("="*40,"SECOND PASS, REPLACING NaN VALUES WITH MEANS","="*40)
    tmp_merged_file = merged_csv_path+".tmp"
    means = sums / counts
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
    os.system(f"awk '!a[$0]++' '{tmp_merged_file}' > '{merged_csv_path}'")
    os.remove(tmp_merged_file)
    print(f"CSV dataset merge completed successfully!")


if __name__ == "__main__":
    HELPTEXT = f"Usage: {sys.argv[0]} DATASET_FOLDER MERGED_DATASET_PATH LABEL_COLUMN BENIGN_LABEL"
    if len(sys.argv) < 5:
        print("Insufficient parameters. Exiting!", file=sys.stderr)
        print(HELPTEXT)
        sys.exit(1)
    else:
        dataset_folder = sys.argv[1]
        merged_dataset_path = sys.argv[2]
        label_column = sys.argv[3]
        benign_label = sys.argv[4]
        merge_cicflow_csvs(dataset_folder, merged_dataset_path, label_column=label_column, chunk_size=2e+6, bin_benign_label=benign_label)
