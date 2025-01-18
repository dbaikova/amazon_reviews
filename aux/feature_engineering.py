def calculate_rolling_stats(df, group_by_col="user_id"):
    df = df.sort_values([group_by_col, "timestamp"])

    suffix = "user" if group_by_col == "user_id" else "product"
    count_col = f"rolling_review_count_{suffix}"
    avg_col = f"rolling_avg_rating_{suffix}"

    # rolling count
    df[count_col] = df.groupby(group_by_col).cumcount() + 1

    # rolling average rating
    df[avg_col] = (
        df.groupby(group_by_col)["rating"]
        .expanding()
        .mean()
        .reset_index(level=0, drop=True)
    )

    return df
