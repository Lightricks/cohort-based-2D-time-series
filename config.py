first_record = "2017-01-01"
max_month_since_attribution = 12
id_columns = ["gender", "cohort", "months_since_attribution"]
sub_group_columns = ["gender"]
target_value = "gf_actual"
features = ["revenue_time0", "monthly_subs_time0", "annual_subs_time0"]
time_to_predict = 12


data_url = "https://raw.githubusercontent.com/ThoughtfulData/tds-datasets/master/subscription_revenue_forecasting.csv"
