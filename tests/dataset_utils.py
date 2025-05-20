import os
import ipaddress
import multiprocessing
import torch
import pandas as pd
import numpy as np

DATASET_FOLDER = os.path.join("/home/sbaltsas/Visual Studio Code Projects/thesis/datasets/IoT-23")
MERGED_DATASET_PATH = os.path.join(DATASET_FOLDER, "IoT-23-merged.csv")
np.random.seed(42)

def _prepare_numeric_columns(df: pd.DataFrame, label_column = "Label") -> pd.DataFrame:
    non_numeric_columns = ["Src IP", "Dst IP", "Timestamp", label_column]
    for column in df.columns:
        if column not in non_numeric_columns and not pd.api.types.is_numeric_dtype(df[column]):
            df[column] = pd.to_numeric(df[column], errors="coerce", downcast="float")
    return df


def _read_csv_in_chunks(file_path, label_column="Label", chunk_size=1e+6):
    # convert all numeric data from float64 to float32, save memory, as model uses float32
    with pd.read_csv(file_path, chunksize=chunk_size, low_memory=False, delimiter=",") as csv_reader:
        for chunk in csv_reader:
            chunk = chunk.drop(columns=["Src IP", "Dst IP", "Timestamp", label_column], errors="ignore")
            chunk = _prepare_numeric_columns(chunk)
            chunk = chunk.astype("float32")
            print("Chunk shape:",chunk.shape, "Datatypes:", chunk.dtypes)
            yield chunk


def _chunk_to_tensor(chunk: pd.DataFrame) -> torch.Tensor:
    return torch.tensor(chunk.to_numpy())


def merge_cicflow_csvs(csvs_directory, merged_parquet_path, label_column="Label", chunk_size=1e+6):
    label_column = label_column.replace("_", " ")
    dataset_columns = [
        "Src IP","Src Port","Dst IP","Dst Port","Protocol","Timestamp","Flow Duration","Tot Fwd Pkts","Tot Bwd Pkts","TotLen Fwd Pkts","TotLen Bwd Pkts",
        "Fwd Pkt Len Max","Fwd Pkt Len Min","Fwd Pkt Len Mean","Fwd Pkt Len Std","Bwd Pkt Len Max","Bwd Pkt Len Min","Bwd Pkt Len Mean","Bwd Pkt Len Std","Flow Byts/s",
        "Flow Pkts/s","Flow IAT Mean","Flow IAT Std","Flow IAT Max","Flow IAT Min","Fwd IAT Tot","Fwd IAT Mean","Fwd IAT Std","Fwd IAT Max","Fwd IAT Min","Bwd IAT Tot",
        "Bwd IAT Mean","Bwd IAT Std","Bwd IAT Max","Bwd IAT Min","Fwd PSH Flags","Bwd PSH Flags","Fwd URG Flags","Bwd URG Flags","Fwd Header Len","Bwd Header Len",
        "Fwd Pkts/s","Bwd Pkts/s","Pkt Len Min","Pkt Len Max","Pkt Len Mean","Pkt Len Std","Pkt Len Var","FIN Flag Cnt","SYN Flag Cnt","RST Flag Cnt","PSH Flag Cnt","ACK Flag Cnt",
        "URG Flag Cnt","CWE Flag Count","ECE Flag Cnt","Down/Up Ratio","Pkt Size Avg","Fwd Seg Size Avg","Bwd Seg Size Avg","Fwd Byts/b Avg","Fwd Pkts/b Avg","Fwd Blk Rate Avg",
        "Bwd Byts/b Avg","Bwd Pkts/b Avg","Bwd Blk Rate Avg","Subflow Fwd Pkts","Subflow Fwd Byts","Subflow Bwd Pkts","Subflow Bwd Byts","Init Fwd Win Byts","Init Bwd Win Byts",
        "Fwd Act Data Pkts","Fwd Seg Size Min","Active Mean","Active Std","Active Max","Active Min","Idle Mean","Idle Std","Idle Max","Idle Min"
    ]
    missing_fields = ["Src IP", "Src Port", "Dst IP"]
    dataset_columns.append(label_column)
    total_dropped_lines = 0
    header_inserted = False
    for root, _ ,files in os.walk(csvs_directory):
        sums = None
        counts = None
        for file in files:
            csv_file_path = os.path.join(root, file)
            print("Merging file {} to {}...".format(csv_file_path, MERGED_DATASET_PATH))
            print("="*50,"FIRST PASS","="*50)
            with pd.read_csv(csv_file_path, chunksize=chunk_size, low_memory=False, delimiter=",") as csv_reader:
                for chunk in csv_reader:
                    chunk.columns = chunk.columns.str.replace("_", " ")
                    # Remove rows that are identical to the header
                    chunk = chunk[chunk.apply(lambda row: not all(str(row[col]) == col for col in chunk.columns), axis=1)]
                    # drop flow id, since its useless
                    chunk.drop(columns=["Flow ID"], inplace=True, errors="ignore")

                    # drop unnamed columns
                    chunk = chunk.loc[:, ~chunk.columns.str.contains('^Unnamed')]

                    if label_column != "Label" and "Label" in chunk.columns:
                        chunk.drop(columns=["Label"], inplace=True, errors="ignore")

                    # Convert numeric data to corresponding numeric pandas datatype
                    chunk = _prepare_numeric_columns(chunk, label_column=label_column)

                    # drop rows with dst port and protocol equal to 0
                    bad_rows = chunk[(chunk['Protocol'] == 0) & (chunk['Dst Port'] == 0) & (chunk[label_column] == "Benign")]
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
                    chunk["Src IP"] = chunk["Src IP"].apply(lambda ip: int(ipaddress.ip_address(ip)))
                    chunk["Dst IP"] = chunk["Dst IP"].apply(lambda ip: int(ipaddress.ip_address(ip)))
                    chunk = chunk.reindex(columns=dataset_columns)
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
                        chunk.to_csv(merged_parquet_path, index=False, header=True, mode="w")
                        header_inserted = True
                    else:
                        chunk.to_csv(merged_parquet_path, index=False, header=False, mode="a")
            print("="*50,"SECOND PASS, REPLACING NaN VALUES WITH MEANS","="*50)
            tmp_merged_file = MERGED_DATASET_PATH+".tmp"
            means = sums / counts
            for i,chunk in enumerate(pd.read_csv(MERGED_DATASET_PATH, delimiter=",", chunksize=chunk_size)):
                numeric_cols = chunk.select_dtypes(include="number").columns
                chunk[numeric_cols] = chunk[numeric_cols].replace([np.inf, -np.inf], np.nan)
                for col in numeric_cols:
                    chunk[col].fillna(means[col], inplace=True)
                chunk.to_csv(tmp_merged_file, mode="a", header=(i == 0), index=False)
            os.remove(MERGED_DATASET_PATH)
            os.rename(tmp_merged_file, MERGED_DATASET_PATH)
            print(f"Done!")

    print(f"CSV dataset merge completed successfully! Total dropped lines: {total_dropped_lines}")


def dataset_to_tensor(dataset_path, label_column, chunk_size=1e+6) -> tuple[torch.Tensor, torch.Tensor]:
    label_column = label_column.replace("_", " ")
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        tensors = pool.map(_chunk_to_tensor, _read_csv_in_chunks(dataset_path, label_column=label_column, chunk_size=chunk_size))
    x_tensor = torch.cat(tensors, dim=0).float()
    y_df = pd.read_csv(dataset_path, delimiter=",", usecols=[label_column], dtype={label_column: "category"})
    y_tensor = torch.tensor(y_df[label_column].cat.codes.to_numpy(), dtype=torch.int64)
    return x_tensor, y_tensor


if __name__ == "__main__":
    torch.set_printoptions(threshold=100)
    if os.path.exists(MERGED_DATASET_PATH):
        X, y = dataset_to_tensor(MERGED_DATASET_PATH, "Label", chunk_size=2e+6)
        print(X.shape, X.dtype)
        print(y.shape, y.dtype)
        pass
    else:
        merge_cicflow_csvs(DATASET_FOLDER, MERGED_DATASET_PATH, label_column="Label", chunk_size=2e+6)
