# Author: Maurits Kruisheer
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

#%% Initializing the file path, and define the filter conditions

# Define folder paths relative to the script location
folder = Path(__file__).resolve().parent.parent / "input data/IFs/Basic Indicators"
out_folder = Path(__file__).resolve().parent.parent / "output data"
conversion_table_path = Path(__file__).resolve().parent.parent / "input data/conversion_table_scenarios.csv"

# Create the new folder if it does not exist
out_folder.mkdir(parents=True, exist_ok=True)

# Filter Countries
filter_countries = ["Congo; Dem. Republic of the", "Ethiopia", "Ghana", "Guatemala", "Haiti", "India", "Indonesia", "Kenya",
                    "Liberia", "Madagascar", "Malawi", "Mali", "Mozambique", "Nepal", "Nigeria", "Philippines", "Rwanda",
                    "Senegal", "Sudan South", "Tanzania", "Uganda", "Zambia"]

# Filter Years
filter_years = [2030, 2050]

#%% Functions

# This function adjusts the value units based on the 
def adjust_scale_based_on_unit(value, unit):
    unit_lower = unit.lower()  # Convert the unit to lowercase for case-insensitive comparison
    if 'million' in unit_lower:
        return value * 1e6
    elif 'billion' in unit_lower:
        return value * 1e9
    elif 'thousand' in unit_lower:
        return value * 1e3
    else:
        return value

# This function determines whether the scenario is 'Water' or 'Sanitation' (or 'Base')
def determine_type(scenario):
    if 'ater' in scenario:
        return 'Water'
    elif 'anit' in scenario:
        return 'Sanitation'
    else:
        return 'Base'

# This function filters Water-Sanitation (Indicator-Scenario) & Sanitation-Water (Indicator-Scenario)
def remove_WASH_false_doubles(df):
    # Add a new column 'WASH_mismatches' with boolean values
    df['WASH_mismatches'] = df.apply(
        lambda row: True if not re.search(r'water|sanitation', row['Indicator'], re.IGNORECASE) else row['Type'] in row['Indicator'] if row['Scenario'] != 'Base' else True, axis=1)
    
    # Filter out the rows where 'WASH_mismatches' is False
    df = df[df['WASH_mismatches']]
    
    # Drop the 'WASH_mismatches' column as it's no longer needed
    df.drop(columns=['WASH_mismatches'], inplace=True)
    
    return df

#%% Transform the IFs data into 1 CSV called 'BasicIndicators.csv'

# Initialize an empty DataFrame to store the final transformed data
abs_df = pd.DataFrame()

for file in sorted(folder.glob('*.csv')):  # This will find all CSV files in the directory
    print(file.name)
    
    # read CSV and remove  the erroneously imported ';' icons
    csv = pd.read_csv(file, header=None)
    csv.replace(';', '', regex=True, inplace=True)
    
    # create empty dataframe
    melted = pd.DataFrame()

    # Extract metadata
    file_years = csv.iloc[0:, 0].dropna().tolist()  # Extract years from the file
    file_indicators = csv.iloc[0, 1:].tolist()  # First row after header for indicators
    file_countries = csv.iloc[1, 1:].tolist()   # Second row for countries
    status = csv.iloc[2, 1:].tolist()   # Third row for status
    file_unit = csv.iloc[4, 1:].tolist()        # Fifth row for unit of analysis
    file_scenario = csv.iloc[5, 1:].tolist() # Sixth row for calculation technique
    
    # Remove the numbers in square brackets from file_indicators
    file_indicators = [re.sub(r'\[\d+\]', '', indicator) for indicator in file_indicators]
    
    # Process the CSV
    # Remove the first column and the first five rows as they contain metadata
    csv_cleaned = csv.drop(csv.columns[0], axis=1).drop(index=range(6))
    
    # Replace semicolons with an empty string
    csv_cleaned.replace(';', '', regex=True, inplace=True)
    
    # Reset index after dropping rows
    csv_cleaned.reset_index(drop=True, inplace=True)

    df_numeric = csv_cleaned.apply(pd.to_numeric, errors='coerce')
    
    # Optional: Fill NaN values with a default value, for example, 0
    csv_cleaned = df_numeric.fillna(0)    
    
    # Transpose the DataFrame to get values aligned with columns correctly
    csv_transposed = csv_cleaned.transpose().reset_index(drop=True)
    
    # Melt the transposed DataFrame to convert it into a long format
    melted = csv_transposed.melt(var_name='Row', value_name='Value')
    melted.drop('Row', axis=1, inplace=True)  # Drop the Row column if not needed
    
    # Check if data is correct
    values = melted['Value']
    print("76th Percentile of all values:", values.quantile(q=0.25))
    print("Mean of all values:", values.mean())
    print("98th Percentile of all values:", values.quantile(q=0.75))
             
    # Now, add the metadata as new columns. This assumes each value corresponds correctly.
    # Adjust as needed based on actual data structure
    melted['Year'] = pd.Series(file_years).repeat(len(file_indicators)).tolist()
    melted['Indicator'] = file_indicators * len(file_years)  
    melted['Unit'] = file_unit * len(file_years)
    melted['Status'] = status * len(file_years)
    melted['Country'] = file_countries * len(file_years)
    melted['Scenario'] = file_scenario * len(file_years)
    
    # Apply the function to the 'Scenario' column to create a new 'Type' column
    melted['Type'] = melted['Scenario'].apply(determine_type)
          
    # Filter the DataFrame for the specified countries and years
    filtered_df = melted[(melted['Country'].isin(filter_countries))]
    
    # Append the filtered data to the final DataFrame
    abs_df = pd.concat([abs_df, filtered_df], ignore_index=True)


# Load conversion table from CSV and change the Indicator names
conversion_table = pd.read_csv(conversion_table_path)
conversion_dict = dict(zip(conversion_table['Indicator'], conversion_table['New_Indicator']))

# Apply the conversion to change indicator names
abs_df['Indicator'] = abs_df['Indicator'].map(conversion_dict).fillna(abs_df['Indicator'])

# Apply the remove_WASH_false_doubles function to filter the DataFrame
abs_df = remove_WASH_false_doubles(abs_df)

# Export CSV: absolute values
abs_file_path = out_folder / 'BasicIndicators_abs.csv'
abs_df.to_csv(abs_file_path, index=False)

#%% Get the "Year Full" values (to be updated, so that YearFull.py can be removed)

# After all files are processed, calculate the access year outside of the loop
access_indicators = [
    "Sanitation Services, Access, percent of population",
    "Water Services, Access, percent of population"
]

all_countries_access = pd.DataFrame(list(itertools.product(filter_countries, access_indicators)), columns=['Country', 'Indicator'])

# Filtering the access to WASH indicators
access_df = abs_df[(abs_df['Indicator'].isin(access_indicators)) & (abs_df['Scenario'] == 'Base')]

# Calculate the total access by summing 'Limited', 'Basic', and 'SafelyManaged' for each country and year
pivot_df = access_df.pivot_table(index=['Indicator', 'Country', 'Year'], columns='Status', values='Value', aggfunc=np.sum).reset_index()

# Add up the relevant columns to get total access percentage
pivot_df['TotalAccess'] = pivot_df[['Basic', 'SafelyManaged']].sum(axis=1)

# Now determine the first year where 'TotalAccess' exceeds 99% for each country
access_year_df = pivot_df[pivot_df['TotalAccess'] > 99].groupby(['Indicator', 'Country'])['Year'].min().reset_index()
access_year_df.rename(columns={'Year': 'YearOf99PctAccess'}, inplace=True)

# Perform the left join on both 'Country' and 'Indicator'
all_countries_access = all_countries_access.merge(access_year_df, on=['Country', 'Indicator'], how='left')

# Where YearOf99PctAccess is NaN, replace with "after 2100"
all_countries_access['YearOf99PctAccess'].fillna('after 2100', inplace=True)

# Drop the 'Indicator_y' column if it exists
if 'Indicator_y' in all_countries_access.columns:
    all_countries_access = all_countries_access.drop(columns=['Indicator_y'])

# Keep only the necessary columns: 'Country', 'Indicator', and 'YearOf99PctAccess'
year_full_access = all_countries_access[['Country', 'Indicator', 'YearOf99PctAccess']]

# Export CSV: yearFull
fullAccess_file_path = out_folder / 'YearFull_access.csv'
year_full_access.to_csv(fullAccess_file_path, index=False)

#%% Get difference values

# Define mappings for absolute and relative indicators as per your specification
relative_indicators = [
    "GDP (MER) - Billion Dollars",
    "GDP per Capita (PPP) - Thousand Dollars"
]

# Assuming the reference scenario is called 'Reference'
ref_df = abs_df[abs_df['Scenario'] == 'Base']

# Create empty difference dataframe
diff_df = pd.DataFrame()

# Compute differences for each scenario compared to the reference
for scenario in abs_df['Scenario'].unique():
    if scenario != 'Base':
        print(scenario)

        scen_df = abs_df[abs_df['Scenario'] == scenario]
        merged_df = pd.merge(scen_df, ref_df, on=['Year', 'Indicator', 'Country'], suffixes=('', '_ref'))

        # Initialize 'Difference' column to avoid KeyError
        merged_df['Difference'] = np.nan

        # Calculate differences
        for indicator in merged_df['Indicator'].unique():
            print(indicator)
            condition = merged_df['Indicator'] == indicator
            if indicator in relative_indicators:
                merged_df.loc[condition, 'Difference'] = \
                    ((merged_df['Value'] - merged_df['Value_ref']) / merged_df['Value_ref']) * 100
            else:
                merged_df.loc[condition, 'Difference'] = \
                    merged_df['Value'] - merged_df['Value_ref']

        # Rename 'Difference' column to 'Value' and reorder columns
        filtered_dfb = merged_df[['Difference', 'Year', 'Indicator', 'Unit', 'Status', 'Country', 'Scenario', 'Type']]

        filtered_dfb.rename(columns={'Difference': 'Value'}, inplace=True)


        # Append to the difference DataFrame
        diff_df = pd.concat([diff_df, filtered_dfb], ignore_index=True)


# Export CSV: difference
diff_file_path = out_folder / 'BasicIndicators_dif.csv'
diff_df.to_csv(diff_file_path, index=False)

#%% Calculate change needed

# # Define WATSAN_names list
# WATSAN_names = [
#  'Sanitation Services, Access, percent of population',
#  'Sanitation Services, Access, Number of people',
#  'Water Services, Access, percent of population',
#  'Water Services, Access, Number of people'
# ]

# # Filtering rows where 'Indicator' column matches one of the values in WATSAN_names
# watsan_access_df = abs_df[abs_df['Indicator'].isin(WATSAN_names)]

# # Calculate the increase needed to reach full WASH in 2030 or 2050

# # Set starting parameters
# start = 2024
# horizon_2030 = 2030
# horizon_2050 = 2050

# # Calculate years left to horizon
# years_left_2030 = horizon_2030 - start
# years_left_2050 = horizon_2050 - start

# # Select full WASH at 2030 & 2050
# fullAccess_scenarios = [
#     "FullWat_2030", 
#     "FullWat_2050", 
#     "FullSan_2030", 
#     "FullSan_2050", 
#     "WaterFull_20"
# ]

# # Only focus on full WASH at 2030 & 2050
# final_filtered_df = watsan_access_df[watsan_access_df['Scenario'].isin(fullAccess_scenarios)]

# # Ensure 'Value' column is numeric
# final_filtered_df['Value'] = pd.to_numeric(final_filtered_df['Value'], errors='coerce')

# # Filter data for the years 2020 and 2030
# data_2020 = final_filtered_df[final_filtered_df['Year'] == 2020]
# data_2030 = final_filtered_df[final_filtered_df['Year'] == 2030]

# # Merge data for 2020 and 2030
# merged_data_2020_2030 = pd.merge(data_2020, data_2030, on=['Country', 'Type', 'Scenario'], suffixes=('_2020', '_2030'))

# # Calculate the average annual difference between 2020 and 2030
# merged_data_2020_2030['Average_Annual_Difference'] = (merged_data_2020_2030['Value_2030'] - merged_data_2020_2030['Value_2020']) / 10

# # Get the value at the start year (2024)
# start_values = final_filtered_df[final_filtered_df['Year'] == start]
# start_values = start_values[['Country', 'Type', 'Scenario', 'Value']]
# start_values.rename(columns={'Value': 'Value_Start_2024'}, inplace=True)

# # Calculate the access gap (100 - value at start year 2024)
# access_gap = start_values.copy()
# access_gap['Access_Gap'] = 100 - access_gap['Value_Start_2024']

# # Merge the access gap with the average annual difference
# merged_data = pd.merge(access_gap, merged_data_2020_2030[['Country', 'Type', 'Scenario', 'Average_Annual_Difference']], on=['Country', 'Type', 'Scenario'])

# # Calculate the years needed for full access
# merged_data['Years_Needed_For_Full_Access'] = merged_data['Access_Gap'] / merged_data['Average_Annual_Difference']

# # Calculate the factor increase needed for 2030 and 2050
# merged_data['Factor_Increase_2030'] = merged_data['Years_Needed_For_Full_Access'] / years_left_2030
# merged_data['Factor_Increase_2050'] = merged_data['Years_Needed_For_Full_Access'] / years_left_2050

# # Determine the final increase needed for 2030 and 2050
# merged_data['Final_Increase_2030'] = merged_data.apply(
#     lambda row: 'there\'s been a deterioration of WASH access over the last years' if row['Average_Annual_Difference'] < 0 
#     else 'no extra increase needed' if row['Factor_Increase_2030'] <= 1 
#     else row['Factor_Increase_2030'], 
#     axis=1
# )

# merged_data['Final_Increase_2050'] = merged_data.apply(
#     lambda row: 'there\'s been a deterioration of WASH access over the last years' if row['Average_Annual_Difference'] < 0 
#     else 'no extra increase needed' if row['Factor_Increase_2050'] <= 1 
#     else row['Factor_Increase_2050'], 
#     axis=1
# )

# # Create the changeNeeded dataframe
# changeNeeded = merged_data[['Country', 'Type', 'Scenario', 'Final_Increase_2030', 'Final_Increase_2050']]

# # Round numeric values to 2 decimals
# changeNeeded['Change_Needed_2030'] = pd.to_numeric(changeNeeded['Final_Increase_2030'], errors='coerce').round(2).fillna(changeNeeded['Final_Increase_2030'])
# changeNeeded['Change_Needed_2050'] = pd.to_numeric(changeNeeded['Final_Increase_2050'], errors='coerce').round(2).fillna(changeNeeded['Final_Increase_2050'])

# # Drop the old columns and keep the renamed ones
# changeNeeded = changeNeeded[['Country', 'Type', 'Scenario', 'Change_Needed_2030', 'Change_Needed_2050']]

# # Export the changeNeeded DataFrame as a new CSV
# change_needed_file_path = Path(out_folder) / 'ChangeNeeded.csv'
# changeNeeded.to_csv(change_needed_file_path, index=False)
