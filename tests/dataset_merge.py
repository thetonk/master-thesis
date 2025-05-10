import pandas as pd
import os

DATASET_FOLDER = os.path.join("datasets/ids-2018")
MERGED_DATASET_PATH = os.path.join(DATASET_FOLDER, "ids-2018-merged.csv")
HEADER_INSERTED = False

missing_fields = ["Src IP", "Src Port", "Dst IP"]
columns_order = [
    "Src IP","Src Port","Dst IP","Dst Port","Protocol","Timestamp","Flow Duration","Tot Fwd Pkts","Tot Bwd Pkts","TotLen Fwd Pkts","TotLen Bwd Pkts",
    "Fwd Pkt Len Max","Fwd Pkt Len Min","Fwd Pkt Len Mean","Fwd Pkt Len Std","Bwd Pkt Len Max","Bwd Pkt Len Min","Bwd Pkt Len Mean","Bwd Pkt Len Std","Flow Byts/s",
    "Flow Pkts/s","Flow IAT Mean","Flow IAT Std","Flow IAT Max","Flow IAT Min","Fwd IAT Tot","Fwd IAT Mean","Fwd IAT Std","Fwd IAT Max","Fwd IAT Min","Bwd IAT Tot",
    "Bwd IAT Mean","Bwd IAT Std","Bwd IAT Max","Bwd IAT Min","Fwd PSH Flags","Bwd PSH Flags","Fwd URG Flags","Bwd URG Flags","Fwd Header Len","Bwd Header Len",
    "Fwd Pkts/s","Bwd Pkts/s","Pkt Len Min","Pkt Len Max","Pkt Len Mean","Pkt Len Std","Pkt Len Var","FIN Flag Cnt","SYN Flag Cnt","RST Flag Cnt","PSH Flag Cnt","ACK Flag Cnt",
    "URG Flag Cnt","CWE Flag Count","ECE Flag Cnt","Down/Up Ratio","Pkt Size Avg","Fwd Seg Size Avg","Bwd Seg Size Avg","Fwd Byts/b Avg","Fwd Pkts/b Avg","Fwd Blk Rate Avg",
    "Bwd Byts/b Avg","Bwd Pkts/b Avg","Bwd Blk Rate Avg","Subflow Fwd Pkts","Subflow Fwd Byts","Subflow Bwd Pkts","Subflow Bwd Byts","Init Fwd Win Byts","Init Bwd Win Byts",
    "Fwd Act Data Pkts","Fwd Seg Size Min","Active Mean","Active Std","Active Max","Active Min","Idle Mean","Idle Std","Idle Max","Idle Min","Label"
]

for root, _ ,files in os.walk("datasets/ids-2018"):
    for file in files:
        csv_file_path = os.path.join(root, file)
        print("Merging file {} to {}...".format(csv_file_path, MERGED_DATASET_PATH))
        with pd.read_csv(csv_file_path, chunksize=1e+6, low_memory=False, delimiter=",") as csv_reader:
            for chunk in csv_reader:
                chunk.drop(columns=["Flow ID"], inplace=True, errors="ignore")
                for field in missing_fields:
                    if field not in chunk:
                        chunk[field] = 0
                        chunk[field].astype(object)
                chunk = chunk.reindex(columns=columns_order)
                print("Chunk shape:",chunk.shape)
                if not HEADER_INSERTED:
                    chunk.to_csv(MERGED_DATASET_PATH, header=True, mode="w", index=False)
                    HEADER_INSERTED = True
                else:
                    chunk.to_csv(MERGED_DATASET_PATH, header=False, mode="a", index=False)
        print("Done!")

print("CSV dataset merge completed successfully!")