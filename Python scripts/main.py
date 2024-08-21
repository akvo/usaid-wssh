# Author: Maurits Kruisheer
# Organization: Akvo Foundation
# Project: USAID WSSH
# Date: April - July 2024

# Description (150 words): This script transforms the data downloaded from Pardee's International Futures (IF) model to be ingested into powerBI. From many CSV's (input) to 1 CSV (output).
# The IF's data is being downloaded as CSVs. Here, each row shows a year (e.g. 2019-2050, 32 rows) and each column a combination of Country, Indicator, Status and Scenario). 
# For PowerBI to make graphs, each row should only contain 1 value. Therefore, the data is being transformed so that 1 row contains the Value, Year, Country, Indicator, Status and Scenario.
# Unit is also added. On top of these, the script generates 2 more columns: Type and dataset. Type indicates whether the Scenario is impacting access to 'Sanitation' or 'Water' (or both). 
# For example, the scenario 'SanInc_2x' increases the Sanitation access with 2x. Type then shows 'Sanitation'. The variable 'dataset' indicates whether the 'Value' is calculated directly ...
# from the IFs model, or whether it shows the difference between a Value and the Reference/Base scenario. 

import pandas as pd
from pathlib import Path
import numpy as np
import re  
import itertools
from functions import transform_IFs_data, get_year_full_values, get_difference_values, calculate_progress_rates

# Initialize the file path, and define the filter conditions
folder = Path(__file__).resolve().parent.parent / "input data/IFs (news)"
out_folder = Path(__file__).resolve().parent.parent / "output data"
conversion_table_path = Path(__file__).resolve().parent.parent / "input data"

out_folder.mkdir(parents=True, exist_ok=True)

filter_countries = pd.read_csv(conversion_table_path / 'conversion_table_countries.csv')["old_name"].tolist()

# Transform the IFs data into 1 CSV called 'BasicIndicators.csv'
abs_df = transform_IFs_data(folder, out_folder, conversion_table_path, filter_countries, 2051)

# Get the "Year Full" values (to be updated, so that YearFull.py can be removed)
year_full_access = get_year_full_values(abs_df, filter_countries, conversion_table_path, out_folder)

# Progress rates 
start_year, end_year = 2020, 2030
progress_rates, progress_rates_diff = calculate_progress_rates(abs_df, start_year, end_year, out_folder)

# Filter years to 2019-2050 for the diff_df (to keep below 100MB)
abs_df = abs_df[abs_df['Year'] <= 2050]

# Get difference values between each Scenario and the Base scenario
diff_df = get_difference_values(abs_df, out_folder)
