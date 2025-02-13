import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(
    path: str = "dataset/2. zillow_cleaned.csv",
    # n_samples: int = 1000,
    test_size: float = 0.2,
    random_state: int = 42,
):

    df = pd.read_csv(path, encoding="utf-8")

    # if n_samples is not None:
    #     df = df.sample(n=n_samples, random_state=random_state)

    X = df[
        [
            "city",
            "single",
            "submarket",  # for ml prediction
            "address",  # for llm generation
            "parking",
            "bathroom",
            "bedroom",
            "age",
            "living",
        ]
    ]
    y = df[["duration"]]
    desc = df[["description"]]
    zpid = df[["zpid"]]

    (
        X_train,
        X_test,
        y_train,
        y_test,
        desc_train,
        desc_test,
        zpid_train,
        zpid_test,
    ) = train_test_split(
        X, y, desc, zpid, test_size=test_size, random_state=random_state
    )

    df_words = pd.concat(
        [
            X_train.reset_index(drop=True),
            desc_train.reset_index(drop=True),
            y_train.reset_index(drop=True),
        ],
        axis=1,
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        desc_train,
        desc_test,
        zpid_train,
        zpid_test,
        df_words,
    )
