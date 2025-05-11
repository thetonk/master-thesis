import os
import multiprocessing
import torch
import pandas as pd

DATASET_FOLDER = os.path.join("datasets/ids-2018")
MERGED_DATASET_PATH = os.path.join(DATASET_FOLDER, "ids-2018-merged.csv")

non_numeric_columns = ["Src IP", "Dst IP", "Timestamp", "Label"]
dataset_columns = [
    "Src IP","Src Port","Dst IP","Dst Port","Protocol","Timestamp","Flow Duration","Tot Fwd Pkts","Tot Bwd Pkts","TotLen Fwd Pkts","TotLen Bwd Pkts",
    "Fwd Pkt Len Max","Fwd Pkt Len Min","Fwd Pkt Len Mean","Fwd Pkt Len Std","Bwd Pkt Len Max","Bwd Pkt Len Min","Bwd Pkt Len Mean","Bwd Pkt Len Std","Flow Byts/s",
    "Flow Pkts/s","Flow IAT Mean","Flow IAT Std","Flow IAT Max","Flow IAT Min","Fwd IAT Tot","Fwd IAT Mean","Fwd IAT Std","Fwd IAT Max","Fwd IAT Min","Bwd IAT Tot",
    "Bwd IAT Mean","Bwd IAT Std","Bwd IAT Max","Bwd IAT Min","Fwd PSH Flags","Bwd PSH Flags","Fwd URG Flags","Bwd URG Flags","Fwd Header Len","Bwd Header Len",
    "Fwd Pkts/s","Bwd Pkts/s","Pkt Len Min","Pkt Len Max","Pkt Len Mean","Pkt Len Std","Pkt Len Var","FIN Flag Cnt","SYN Flag Cnt","RST Flag Cnt","PSH Flag Cnt","ACK Flag Cnt",
    "URG Flag Cnt","CWE Flag Count","ECE Flag Cnt","Down/Up Ratio","Pkt Size Avg","Fwd Seg Size Avg","Bwd Seg Size Avg","Fwd Byts/b Avg","Fwd Pkts/b Avg","Fwd Blk Rate Avg",
    "Bwd Byts/b Avg","Bwd Pkts/b Avg","Bwd Blk Rate Avg","Subflow Fwd Pkts","Subflow Fwd Byts","Subflow Bwd Pkts","Subflow Bwd Byts","Init Fwd Win Byts","Init Bwd Win Byts",
    "Fwd Act Data Pkts","Fwd Seg Size Min","Active Mean","Active Std","Active Max","Active Min","Idle Mean","Idle Std","Idle Max","Idle Min","Label"
]

def _prepare_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    for column in df.columns:
        if column not in non_numeric_columns and not pd.api.types.is_numeric_dtype(df[column]):
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df

def _read_csv_in_chunks(file_path, chunk_size=1e+6):
    with pd.read_csv(file_path, chunksize=chunk_size, low_memory=False, delimiter=",") as csv_reader:
        for chunk in csv_reader:
            chunk = chunk.drop(columns=["Src IP", "Dst IP", "Timestamp", "Label"], errors="ignore")
            chunk = _prepare_numeric_columns(chunk)
            print("Chunk shape:",chunk.shape, "Datatypes:", chunk.dtypes)
            yield chunk

def _chunk_to_tensor(chunk: pd.DataFrame) -> torch.Tensor:
    return torch.tensor(chunk.to_numpy())

def merge_cicflow_csvs(csvs_directory, merged_csv_path):
    total_dropped_lines = 0
    header_inserted = False
    missing_fields = ["Src IP", "Src Port", "Dst IP"]
    for root, _ ,files in os.walk(csvs_directory):
        for file in files:
            csv_file_path = os.path.join(root, file)
            print("Merging file {} to {}...".format(csv_file_path, MERGED_DATASET_PATH))
            with pd.read_csv(csv_file_path, chunksize=1e+6, low_memory=False, delimiter=",") as csv_reader:
                for chunk in csv_reader:
                    # Remove rows that are identical to the header
                    chunk = chunk[chunk.apply(lambda row: not all(str(row[col]) == col for col in chunk.columns), axis=1)]
                    
                    # drop flow id, since its useless
                    chunk.drop(columns=["Flow ID"], inplace=True, errors="ignore")

                    # Convert numeric data to corresponding numeric pandas datatype
                    chunk = _prepare_numeric_columns(chunk)

                    # drop rows with dst port and protocol equal to 0
                    bad_rows = chunk[(chunk['Protocol'] == 0) & (chunk['Dst Port'] == 0) & (chunk['Label'] == "Benign")]
                    if not bad_rows.empty:
                        bad_row_count = len(bad_rows)
                        total_dropped_lines += bad_row_count
                        print(f"Dropping {bad_row_count} Benign traffic rows with Protocol=0 and Dst Port=0!")
                    chunk = chunk.drop(bad_rows.index)

                    for field in missing_fields:
                        if field not in chunk:
                            chunk[field] = 0
                            chunk[field] = chunk[field].astype(object)
                    chunk = chunk.reindex(columns=dataset_columns)
                    print("Chunk shape:",chunk.shape, "Datatypes:", chunk.dtypes)
                    if not header_inserted:
                        chunk.to_csv(merged_csv_path, header=True, mode="w", index=False)
                        header_inserted = True
                    else:
                        chunk.to_csv(merged_csv_path, header=False, mode="a", index=False)
            print(f"Done!")

    print(f"CSV dataset merge completed successfully! Total dropped lines: {total_dropped_lines}")

def dataset_to_tensor(dataset_path) -> torch.Tensor:
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        tensors = pool.map(_chunk_to_tensor, _read_csv_in_chunks(dataset_path))
    return torch.cat(tensors, dim=0)

if __name__ == "__main__":
    torch.set_printoptions(threshold=100)
    if os.path.exists(MERGED_DATASET_PATH):
        tensor = dataset_to_tensor(MERGED_DATASET_PATH)
        print(tensor.shape, tensor.dtype)
    else:
        merge_cicflow_csvs(DATASET_FOLDER, MERGED_DATASET_PATH)