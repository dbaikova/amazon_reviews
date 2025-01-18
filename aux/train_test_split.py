import numpy as np
import pandas as pd


def global_temporal_split(df, split_ratio=0.8, exclude_cold_start_users=False):
    df = df.sort_values(by="timestamp")

    split_index = int(len(df) * split_ratio)
    split_date = df.iloc[split_index]["timestamp"]

    # temporal split
    train_df = df[df["timestamp"] < split_date].copy()
    test_df = df[df["timestamp"] >= split_date].copy()

    # ensure all users in the test set are in the train set
    test_users = set(test_df["user_id"])
    train_users = set(train_df["user_id"])

    # identify cold-start users in the test set and remove them
    if exclude_cold_start_users:
        cold_start_users = test_users - train_users
        test_df = test_df[~test_df["user_id"].isin(cold_start_users)].reset_index(
            drop=True
        )
        return (
            train_df.reset_index(drop=True),
            test_df.reset_index(drop=True),
            cold_start_users,
        )

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def temporal_split_users_in_both_sets(df, split_ratio=0.8):
    train_list = []
    test_list = []

    for user_id, group in df.groupby("user_id"):
        group = group.sort_values(by="timestamp")

        # calculate the split index
        split_index = int(len(group) * split_ratio)

        # split
        train = group.iloc[:split_index]
        test = group.iloc[split_index:]

        train_list.append(train)
        test_list.append(test)

    train_df = pd.concat(train_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)

    return train_df, test_df


def temporal_split_users_with_cold_start(
    df, split_ratio=0.8, cold_start_ratio=0.1, random_seed=42
):
    """
    Includes a portion of cold-start users only in the test set
    """
    np.random.seed(random_seed)
    unique_users = df["user_id"].unique()

    # cold-start users
    n_cold_start = int(len(unique_users) * cold_start_ratio)
    cold_start_users = np.random.choice(unique_users, size=n_cold_start, replace=False)
    non_cold_start_users = list(set(unique_users) - set(cold_start_users))

    train_list = []
    test_list = []

    # process non-cold-start users
    for user_id in non_cold_start_users:
        group = df[df["user_id"] == user_id].sort_values(by="timestamp")

        split_index = int(len(group) * split_ratio)

        train = group.iloc[:split_index]
        test = group.iloc[split_index:]

        train_list.append(train)
        test_list.append(test)

    # add interactions of cold-start users to the test set
    cold_start_test = df[df["user_id"].isin(cold_start_users)]
    test_list.append(cold_start_test)

    train_df = pd.concat(train_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)

    return train_df, test_df
