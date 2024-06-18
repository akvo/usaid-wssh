import pandas as pd
import numpy as np
import re
import itertools
from pathlib import Path

def adjust_scale_based_on_unit(value, unit):
    """
    Adjusts the value based on the unit provided.
    
    Parameters:
    - value (float): The numerical value to adjust.
    - unit (str): The unit of the value (e.g., 'million', 'billion', 'thousand').
    
    Returns:
    - float: The adjusted value.
    """
    unit_lower = unit.lower()  # Convert the unit to lowercase for case-insensitive comparison
    if 'million' in unit_lower:
        return value * 1e6
    elif 'billion' in unit_lower:
        return value * 1e9
    elif 'thousand' in unit_lower:
        return value * 1e3
    else:
        return value

def determine_type(scenario):
    """
    Determines whether the scenario is related to 'Water' or 'Sanitation' or is a 'Base' scenario.
    
    Parameters:
    - scenario (str): The scenario description.
    
    Returns:
    - str: 'Water', 'Sanitation', or 'Base' based on the scenario.
    """
    if 'ater' in scenario:
        return 'Water'
    elif 'anit' in scenario:
        return 'Sanitation'
    else:
        return 'Base'

def remove_WASH_false_doubles(df):
    """
    Filters out rows where the Indicator and Scenario combination does not match Water-Sanitation criteria.
    
    Parameters:
    - df (DataFrame): The DataFrame to filter.
    
    Returns:
    - DataFrame: The filtered DataFrame.
    """
    # Add a new column 'WASH_mismatches' with boolean values
    df['WASH_mismatches'] = df.apply(
        lambda row: True if not re.search(r'water|sanitation', row['Indicator'], re.IGNORECASE) else row['Type'] in row['Indicator'] if row['Scenario'] != 'Base' else True, axis=1)
    
    # Filter out the rows where 'WASH_mismatches' is False
    df = df[df['WASH_mismatches']]
    
    # Drop the 'WASH_mismatches' column as it's no longer needed
    df.drop(columns=['WASH_mismatches'], inplace=True)
    
    return df

def transform_IFs_data(folder, out_folder, conversion_table_path, filter_countries):
    """
    Transforms the IFs data into a single CSV file called 'BasicIndicators_abs.csv'.
    
    Parameters:
    - folder (Path): The directory containing the input CSV files.
    - out_folder (Path): The directory to save the output CSV files.
    - conversion_table_path (Path): The path to the conversion table CSV file.
    - filter_countries (list): List of countries to filter.
    
    Returns:
    - DataFrame: The transformed data.
    """
    abs_df = pd.DataFrame()  # Initialize an empty DataFrame to store the final transformed data

    for file in sorted(folder.glob('*.csv')):  # This will find all CSV files in the directory
        print(file.name)
        
        # read CSV and remove the erroneously imported ';' icons
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

    return abs_df

def get_year_full_values(abs_df, filter_countries, out_folder):
    """
    Calculates the "Year Full" values for each country and indicator and saves the result to 'YearFull_access.csv'.
    
    Parameters:
    - abs_df (DataFrame): The DataFrame containing the transformed data.
    - filter_countries (list): List of countries to filter.
    - out_folder (Path): The directory to save the output CSV files.
    
    Returns:
    - DataFrame: The DataFrame with the year full values.
    """
    access_indicators = [
        "Sanitation Services, Access, percent of population",
        "Water Services, Access, percent of population"
    ]

    # Filtering the access to WASH indicators
    access_df = abs_df[(abs_df['Indicator'].isin(access_indicators)) & (abs_df['Scenario'] == 'Base')]

    # Calculate the total access by summing 'Limited', 'Basic', and 'SafelyManaged' for each country and year
    pivot_df = access_df.pivot_table(index=['Indicator', 'Country', 'Year'], columns='Status', values='Value', aggfunc=np.sum).reset_index()

    # Add up the relevant columns to get total access percentage
    pivot_df['TotalAccess'] = pivot_df[['Basic', 'SafelyManaged']].sum(axis=1)

    # Now determine the first year where 'TotalAccess' exceeds 99% for each country
    access_year_df = pivot_df[pivot_df['TotalAccess'] > 99].groupby(['Indicator', 'Country'])['Year'].min().reset_index()
    access_year_df.rename(columns={'Year': 'YearOf99PctAccess'}, inplace=True)

    all_countries_access = pd.DataFrame(list(itertools.product(filter_countries, access_indicators)), columns=['Country', 'Indicator'])
    
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

    return year_full_access

def get_difference_values(abs_df, out_folder):
    """
    Calculates the difference values between scenarios and the reference scenario and saves the result to 'BasicIndicators_dif.csv'.
    
    Parameters:
    - abs_df (DataFrame): The DataFrame containing the transformed data.
    - relative_indicators (list): List of indicators that should be calculated as relative differences.
    - out_folder (Path): The directory to save the output CSV files.
    
    Returns:
    - DataFrame: The DataFrame with the difference values.
    """
    # Assuming the reference scenario is called 'Reference'
    ref_df = abs_df[abs_df['Scenario'] == 'Base']

    # Create empty difference dataframe
    diff_df = pd.DataFrame()
    
    # Define mappings for absolute and relative indicators as per your specification
    relative_indicators = [
        "GDP (MER) - Billion Dollars",
        "GDP per Capita (PPP) - Thousand Dollars"
    ]

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

            # Ensure the Value column has a maximum of 5 digits
            filtered_dfb['Value'] = filtered_dfb['Value'].apply(lambda x: round(x, 5))

            # Append to the difference DataFrame
            diff_df = pd.concat([diff_df, filtered_dfb], ignore_index=True)

    # Export CSV: difference
    diff_file_path = out_folder / 'BasicIndicators_dif.csv'
    diff_df.to_csv(diff_file_path, index=False)

    return diff_df
