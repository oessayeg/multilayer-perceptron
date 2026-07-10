import sys
import pandas as pd

TRAIN_RATIO = 0.8


def split_dataset(filepath: str, train_ratio: float = TRAIN_RATIO) -> None:
    df = pd.read_csv(filepath, header=None)
    df = df.sample(frac=1, random_state=1).reset_index(drop=True)

    split_index = int(len(df) * train_ratio)
    train_df = df.iloc[:split_index]
    validate_df = df.iloc[split_index:]

    train_df.to_csv("dataset_train.csv", index=False, header=False)
    validate_df.to_csv("dataset_validate.csv", index=False, header=False)

    print(f"Total samples  : {len(df)}")
    print(f"Training set   : {len(train_df)} rows -> dataset_train.csv")
    print(f"Validation set : {len(validate_df)} rows -> dataset_validate.csv")


if __name__ == "__main__":
    try:
        if len(sys.argv) != 2:
            print("Usage: python split_dataset.py <dataset.csv>")
            sys.exit(1)

        split_dataset(sys.argv[1])
    except Exception as e:
        print(f"Error: {e}")
