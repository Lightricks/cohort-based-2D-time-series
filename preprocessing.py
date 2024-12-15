import numpy as np
import pandas as pd

from datetime import datetime
from dateutil.relativedelta import relativedelta


def create_date_range(start_date, end_date, step='M'):
    """
    Generate a range of dates between two dates with a distance of 1 month apart,
    including both the start and end dates.

    Args:
    - start_date (str): The start date in "YYYY-MM-DD" format.
    - end_date (str): The end date in "YYYY-MM-DD" format.

    Returns:
    - list of datetime.date: List of dates from start to end, 1 month apart.
    """
    # ensure that the step is valid
    if step not in ['Y', 'M', 'D', 'W', 'H', 'm', 's']:
        raise ValueError("Invalid step value. Please use 'M', 'D', 'Y', 'W', 'H', 'm', or 's'.")

    # ensure times are valid
    if not isinstance(start_date, datetime):
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
    if not isinstance(end_date, datetime):
        end_date = datetime.strptime(end_date, "%Y-%m-%d")

    # Transforms the start and end dates into the correct format for each type - date for year, month, day, and datetime for hour, minute, second
    if step in ['Y', 'M', 'D']:
        start = datetime.strptime(str(start_date.date()), "%Y-%m-%d").date()
        end = datetime.strptime(str(end_date.date()), "%Y-%m-%d").date()
    else:
        start = start_date
        end = end_date

    current = start
    date_range = [start]

    while current < end:
        if step == 'Y':
            current += relativedelta(years=1)
        elif step == 'M':
            current += relativedelta(months=1)
        elif step == 'D':
            current += relativedelta(days=1)
        elif step == 'W':
            current += relativedelta(weeks=1)
        elif step == 'H':
            current += relativedelta(hours=1)
        elif step == 'm':
            current += relativedelta(minutes=1)
        elif step == 's':
            current += relativedelta(seconds=1)

        if current <= end:
            date_range.append(current)

    return date_range


def add_time0_data(df, on_columns, step='M'):
    """
    Add time0 data to the DataFrame
    :param df: DataFrame
    :param on_columns: list of columns to merge on
    :param time_column: column to use for time
    :return: DataFrame
    """
    month0_data = df[df['time_since_attribution'] == 0]
    # Add month0 data
    df = pd.merge(
        df,
        month0_data,
        on=on_columns,
        suffixes=("", "_time0"),
        how="left",
    )

    if step=='Y':
        df["time_purchase"] = df.apply(
            lambda x: x.cohort + pd.DateOffset(years=x.time_since_attribution), axis=1
        )
    elif step=='M':
        df["time_purchase"] = df.apply(
            lambda x: x.cohort + pd.DateOffset(months=x.time_since_attribution), axis=1
        )
    elif step=='D':
        df["time_purchase"] = df.apply(
            lambda x: x.cohort + pd.DateOffset(days=x.time_since_attribution), axis=1
        )
    elif step=='W':
        df["time_purchase"] = df.apply(
            lambda x: x.cohort + pd.DateOffset(weeks=x.time_since_attribution), axis=1
        )
    elif step=='H':
        df["time_purchase"] = df.apply(
            lambda x: x.cohort + pd.DateOffset(hours=x.time_since_attribution), axis=1
        )
    elif step=='m':
        df["time_purchase"] = df.apply(
            lambda x: x.cohort + pd.DateOffset(minutes=x.time_since_attribution), axis=1
        )
    elif step=='s':
        df["time_purchase"] = df.apply(
            lambda x: x.cohort + pd.DateOffset(seconds=x.time_since_attribution), axis=1
        )

    return df


def add_future_records(df, id_columns, pred_columns='revenue', fill=False):
    """
    Add future records to a DataFrame based on unique values in specified columns.
    :param df: DataFrame to add future records to
    :param id_columns: Columns to use for creating unique values
    :param pred_columns: Column to fill with future records
    :param fill: Whether to fill NaN values with 0
    :return: DataFrame with future records added
    """
    df.rename(columns={pred_columns: 'revenue'}, inplace=True)

    # Dynamically create lists of unique values for each ID column
    unique_values = [df[col].unique().tolist() for col in id_columns if "time_since_attribution" not in col]
    # ensure that for time_since_attribution, we have all the times from 0, to adjusted_df.time_since_attribution.max() + 1
    unique_values.append(np.arange(0, df.time_since_attribution.max() + 1).tolist())

    # Create a MultiIndex from the product of unique values
    idx = pd.MultiIndex.from_product(
        unique_values,
        names=id_columns,
    )

    # Reindex the DataFrame and fill NaN values with 0
    full_df = df.set_index(id_columns).reindex(idx).reset_index()

    # drop combination of attribution_month-vertical-purchase_platform-country that all their
    # net_proceeds_incl_projected_trials are 0 or none
    full_df = full_df.groupby([val for val in id_columns if "since_attribution" not in val]
                              ).filter(
        lambda x: (x.revenue.sum() != 0)
                  | (x.revenue.isna().sum() != x.shape[0])
    )
    full_df.sort_values(
        by=id_columns,
        inplace=True,
    )

    # fill NaN values:
    if fill:
        full_df["revenue"] = full_df["revenue"].fillna(0)

    return full_df