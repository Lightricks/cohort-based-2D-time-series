import logging
from logging import Logger

logging.basicConfig(level=logging.INFO)
Logger: Logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import pmdarima as pm
from threadpoolctl import ThreadpoolController

from config import features, target_value, time_to_predict
from concurrent.futures import ProcessPoolExecutor
import multiprocessing




def arima_train_pred(train, train_pvt, features, rolling_time, horizon_steps, logger= Logger):
    """
    This function is used to train the ARIMA model and make predictions for the given time.
    :param horizon_steps: int, the number of steps ahead to forecast
    :param train: DataFrame, the training data
    :param train_pvt: DataFrame, the training data in pivot table format
    :param features:  list of features
    :param rolling_time: int, the time for which the predictions are to be made
    :param logger: the logger object
    :return: the predictions
    """
    controller = ThreadpoolController()
    y = train_pvt.iloc[:-rolling_time, rolling_time]
    added_features = (
        train[["cohort"] + features]
        .drop_duplicates()
        .astype({val: "float" for val in features})
    )
    x_train = pd.merge(
        pd.DataFrame(train_pvt.iloc[:-rolling_time, (rolling_time - 1)]),
        added_features,
        on="cohort",
    ).set_index("cohort")
    x_train.columns = x_train.columns.astype(str)
    x_pred = pd.merge(
        pd.DataFrame(train_pvt.iloc[-rolling_time:, (rolling_time - 1)]),
        added_features,
        on="cohort",
    ).set_index("cohort")
    x_pred.columns = x_pred.columns.astype(str)
    # if there is enough data to train the model with seasonality of horizon_steps do so otherwise train
    # without a seasonality
    try:
        with controller.limit(limits=1, user_api="blas"):
            ts_model = pm.auto_arima(
                y,
                X=x_train,
                start_p=0,
                start_q=0,
                max_p=horizon_steps,
                max_q=horizon_steps,
                m=horizon_steps,
                start_P=0,
                error_action="ignore",
                n_jobs=1,
            )
            pred_new = ts_model.predict(
                X=x_pred, n_periods=int(rolling_time), return_conf_int=True, alpha=0.05
            )
    except ValueError as e:
        logger.debug(f"Retrain ARIMA model due to {e}")
        with controller.limit(limits=1, user_api="blas"):
            ts_model = pm.auto_arima(
                y,
                X=x_train,
                start_p=0,
                start_q=0,
                max_p=horizon_steps,
                max_q=horizon_steps,
                start_P=0,
                seasonal_test=True,
                error_action="ignore",
                n_jobs=1,
            )
            pred_new = ts_model.predict(
                X=x_pred, n_periods=int(rolling_time), return_conf_int=True, alpha=0.05
            )
    return pred_new


def filter_subgroups(df, sub_group_columns, sub_group):
    """
    This function is used to filter the dataframe for a given sub_group.
    :param df:
    :param sub_group_columns:
    :param sub_group:
    :return:
    """
    # Ensure sub_group_columns is a list
    if isinstance(sub_group_columns, str):
        sub_group_columns = [sub_group_columns]

    # Ensure sub_group is a tuple
    if not isinstance(sub_group, tuple):
        sub_group = (sub_group,)

    # Create the filtering condition
    condition = True
    for column, value in zip(sub_group_columns, sub_group):
        condition &= (df[column] == value)

    return df[condition]


def sub_group_pair_inference(
    sub_group,
    sub_group_columns,
    processed_df,
    prediction_time,
    horizon_steps,
    logger= Logger,
):
    """
    This function is used to process the data for a given gender and make
    predictions for the given prediction time. It uses the ARIMA model to make
    predictions for the given time and returns the predictions.
    :param horizon_steps: int, the number of steps ahead to forecast
    :param processed_df: the dataframe containing the processed data
    :param sub_group: a tuple containing the sub_group identity.
    :param sub_group_columns: the columns that define the sub_group
    :param prediction_time:  the time for which the predictions are to be made
    :param logger:  the logger object
    :return:  the dataframe containing the predictions
    """
    logger.info(f"Started {sub_group} iteration")
    # although cutoff represent *prediction* time we cut off by purchase_time
    # as it represents the time we know
    cutoff = pd.to_datetime(prediction_time)
    sub_group_revs_dat = filter_subgroups(
        processed_df, sub_group_columns, sub_group
    )
    sub_group_revs_dat["time_purchase"] = pd.to_datetime(
        sub_group_revs_dat["time_purchase"]
    )
    train = sub_group_revs_dat[sub_group_revs_dat["time_purchase"] < cutoff]
    pred_df = sub_group_revs_dat[sub_group_revs_dat["time_purchase"] >= cutoff]

    train_pvt_reg = train.pivot_table(
        index="cohort",
        columns="time_since_attribution",
        values=target_value,
    )
    # prevent attempts of prediction when lacking enough past data]
    # (at least settings.time_to_predict time)
    if (train_pvt_reg > 0).sum(axis=1).count() < time_to_predict + 2:
        logger.info(
            f"not enough past data, skipping {sub_group, str(cutoff)}"
        )
        return pd.DataFrame()
    train_pvt_lower_bounds = train_pvt_reg.copy()
    train_pvt_upper_bounds = train_pvt_reg.copy()

    skip_rest_of_windows = False
    for rolling_time in np.arange(1, time_to_predict + 1):
        # for prediction
        if skip_rest_of_windows:
            return pd.DataFrame()
        pred_new_reg = arima_train_pred(
            train=train,
            train_pvt=train_pvt_reg,
            features=features,
            rolling_time=rolling_time,
            horizon_steps=horizon_steps,
            logger=logger
        )
        # if pred_new_reg contains negative values, logging and continue
        if (pred_new_reg[0] < 0).any():
            logger.info(
                f"negative prediction for {sub_group, cutoff}, skipping"
            )
            return pd.DataFrame()
        train_pvt_reg.iloc[-rolling_time:, rolling_time] = pred_new_reg[0]
        if (
            rolling_time == 1
        ):  # for the first iteration take the upper and lower bound of the prediction
            train_pvt_lower_bounds.iloc[-rolling_time:, rolling_time] = pred_new_reg[1][:, 0]
            train_pvt_upper_bounds.iloc[-rolling_time:, rolling_time] = pred_new_reg[1][:, 1]
        else:
            train_pvt_lower_bounds.iloc[-rolling_time:, rolling_time] = None
            train_pvt_upper_bounds.iloc[-rolling_time:, rolling_time] = None

    train_preds = pd.melt(
        train_pvt_reg.reset_index(),
        id_vars="cohort",
        value_vars=np.array(train_pvt_reg.columns),
    )
    train_preds.rename(columns={"value": "predictions"}, inplace=True)
    pred_df = pred_df.merge(train_preds, on=["cohort", "time_since_attribution"])

    train_preds_lower = pd.melt(
        train_pvt_lower_bounds.reset_index(),
        id_vars="cohort",
        value_vars=np.array(train_pvt_lower_bounds.columns),
    )
    train_preds_lower.rename(columns={"value": "lower_bound_prediction"}, inplace=True)
    pred_df = pred_df.merge(train_preds_lower, on=["cohort", "time_since_attribution"])

    train_preds_upper = pd.melt(
        train_pvt_upper_bounds.reset_index(),
        id_vars="cohort",
        value_vars=np.array(train_pvt_upper_bounds.columns),
    )
    train_preds_upper.rename(columns={"value": "upper_bound_prediction"}, inplace=True)
    pred_df = pred_df.merge(train_preds_upper, on=["cohort", "time_since_attribution"])

    pred_df[
        "prediction_time"
    ] = cutoff  # prediction is done once a time is over and all knowledge is known

    # if train_pvt_reg contains any row that is not constantly increasing,
    # logging and continue
    if (train_pvt_reg.diff(axis=1) < 0).any(axis=1).any():
        logger.info(
            f"decreasing prediction for {sub_group, cutoff}, skipping"
        )
        return pd.DataFrame()

    logger.info(f"Finished {sub_group, cutoff} iteration")
    return pred_df



def run_inference(processed_df, prediction_time, sub_group_column, horizon_steps, logger= Logger):
    """
    This function is used to run the inference for all the subscriptions and countries
    in the given dataframe and make predictions for the given prediction time.
    It uses the sub_group_pair_inference function to make predictions for each
    sub_group and returns the predictions.
    :param horizon_steps: int, the number of steps ahead to forecast
    :param sub_group_column: the column that defines the sub_group
    :param processed_df: the dataframe containing the processed data
    :param prediction_time: the time for which the predictions are to be made
    :param logger: the logger object
    :return: the dataframe containing the predictions
    """
    # Create a list of all combinations
    sub_group_pairs = [
        sub_group
        for sub_group, _ in processed_df.groupby(
            sub_group_column
        )
    ]

    # Use ProcessPoolExecutor to run the processing in parallel
    all_dfs = []
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count() - 1) as executor:
        future_to_pair = {
            executor.submit(
                sub_group_pair_inference,
                sub_group,
                sub_group_column,
                processed_df,
                prediction_time,
                horizon_steps,
                logger
            ): sub_group
            for sub_group in sub_group_pairs
        }
        import concurrent.futures

        for future in concurrent.futures.as_completed(future_to_pair):
            sub_group = future_to_pair[future]
            try:
                result_df = future.result()
                all_dfs.append(result_df)
            except Exception as exc:
                logger.error(
                    f"Error processing {sub_group}: {exc}"
                )

    logger.info("Finished cohort inference")
    return pd.concat(all_dfs)
