# enable running the script from the command line
import argparse
import logging

from model import run_inference
from config import sub_group_columns

logging.basicConfig(level=logging.INFO)
Logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd

from datetime import datetime
from dateutil.relativedelta import relativedelta



def main(data_path, save_path, prediction_time, horizon_steps, step_unit):
    if prediction_time is None:
        prediction_time = datetime.now().strftime("%Y-%m-%d")
    if horizon_steps is None:
        horizon_steps = 12
    if step_unit is None:
        step_unit = 'M'

    if step_unit in ['Y', 'M', 'D']:
        prediction_time = datetime.strptime(prediction_time, "%Y-%m-%d").date()

    df  = pd.read_csv(data_path)

    Logger.info(f"Data loaded from {data_path}")

    # run model
    Logger.info(f"Running model for prediction date {prediction_time}, horizon steps {horizon_steps}, step unit {step_unit}")

    results = run_inference(
        processed_df=df,
        prediction_time=prediction_time,
        sub_group_column=sub_group_columns,
        horizon_steps=horizon_steps,
        logger=Logger)

    Logger.info(f"Model run completed")

    #try to save the results
    try:
        results.to_csv(save_path, index=False)
        Logger.info(f"Results saved to {save_path}")
    except Exception as e:
        Logger.error(f"Error saving results: {e}")
        raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 2d ARIMA")


    def is_date(s):
        try:
            datetime.strptime(s, "%Y-%m-%d")
            return s
        except ValueError:
            msg = f"Not a valid date: '{s}'."
            raise argparse.ArgumentTypeError(msg)

    def is_step_unit(s):
        if s not in ['Y', 'M', 'D', 'W', 'H', 'm', 's']:
            msg = f"Invalid step value. Please use 'M', 'D', 'Y', 'W', 'H', 'm', or 's'."
            raise argparse.ArgumentTypeError(msg)
        return s


    parser.add_argument(
        "--prediction_time",
        "-pt",
        type=is_date,
        metavar="PREDICTION_TIME",
        help="The time for which the prediction is done, in the format of '%Y-%m-%d'",
        required=False,
        default=datetime.now().strftime("%Y-%m-%d"),
    )

    parser.add_argument(
        "--horizon_steps",
        "-hs",
        type=int,
        metavar="HORIZON_STEPS",
        help="The number of steps ahead to forecast",
        required=False,
        default=None,
    )

    parser.add_argument(
        "--step_unit",
        "-su",
        type=is_step_unit,
        metavar="STEP_UNIT",
        help="The unit of the step, one of 'Y', 'M', 'D', 'W', 'H', 'm', 's'",
        required=False,
        default='M',
    )

    parser.add_argument(
        "--data_path",
        "-dp",
        type=str,
        metavar="DATA_PATH",
        help="The path to the data",
        required=True,
    )

    parser.add_argument(
        "--save_path",
        "-sp",
        type=str,
        metavar="SAVE_PATH",
        help="The path to save the results",
        required=True,
    )



    args = parser.parse_args()
    args_dict = vars(args)

    main(**args_dict)
