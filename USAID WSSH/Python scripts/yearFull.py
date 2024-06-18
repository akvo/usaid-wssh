import pandas as pd
from pathlib import Path
import numpy as np

# Get the current script directory
script_dir = Path(__file__).resolve().parent

# Define the input and output folders relative to the script directory
input_folder = Path(__file__).resolve().parent.parent / "input data/IFs/Basic Indicators"
out_folder = Path(__file__).resolve().parent.parent / "output data"

# Create the new folder if it does not exist
input_folder.mkdir(parents=True, exist_ok=True)
out_folder.mkdir(parents=True, exist_ok=True)

# filter Countries
filter_countries = ["Congo; Dem. Republic of the", "Ethiopia", "Ghana", "Guatemala", "Haiti", "India", "Indonesia", "Kenya",
                    "Liberia", "Madagascar", "Malawi", "Mali", "Mozambique", "Nepal", "Nigeria", "Philippines", "Rwanda",
                    "Senegal", "Sudan South", "Tanzania", "Uganda", "Zambia"]

# load data
for file in sorted(Path(input_folder).glob('*.csv')):  # This will find all CSV files in the directory
    print(file.name)
    
    # Create a DataFrame to track the year of reaching 99% access for all countries
    all_countries_access = pd.DataFrame(filter_countries, columns=['Country'])

    # Initialize an empty DataFrame to store the final transformed data
    final_df = pd.DataFrame()
    
    csv = pd.read_csv(file, header=None)
    
    # Extract metadata
    file_years = csv.iloc[0:, 0].dropna().astype(int).tolist()  # Extract years from the file and convert to integers
    file_indicators = csv.iloc[0, 1:].tolist()  # First row after header for indicators
    file_countries = csv.iloc[1, 1:].tolist()   # Second row for countries
    status = csv.iloc[2, 1:].tolist()   # Third row for status
    file_unit = csv.iloc[4, 1:].tolist()        # Fifth row for unit of analysis
    file_calculation = csv.iloc[5, 1:].tolist() # Sixth row for calculation technique
    
    # Process the CSV
    # Remove the first column and the first five rows as they contain metadata
    csv_cleaned = csv.drop(csv.columns[0], axis=1).drop(index=range(6))
    
    # Reset index after dropping rows
    csv_cleaned.reset_index(drop=True, inplace=True)

    df_numeric = csv_cleaned.apply(pd.to_numeric, errors='coerce')
    
    # Optional: Fill NaN values with a default value, for example, 0
    csv_cleaned = df_numeric.fillna(0)    
    
    # Transpose the DataFrame to get values aligned with columns correctly
    csv_transposed = csv_cleaned.transpose().reset_index(drop=True)
    
    # Melt the transposed DataFrame to convert it into a long format
    melted = csv_transposed.melt(var_name='Row', value_name='Value')
    melted.drop('Row', axis=1, inplace=True)
    
    # Add the metadata as new columns
    melted['Year'] = pd.Series(file_years).repeat(len(file_indicators)).tolist()
    melted['Indicator'] = file_indicators * len(file_years)  
    melted['Unit'] = file_unit * len(file_years)
    melted['Status'] = status * len(file_years)
    melted['Country'] = file_countries * len(file_years)
    melted['Calculation'] = file_calculation * len(file_years)
    
    # Filter the DataFrame for the specified countries and years
    filtered_df = melted[(melted['Country'].isin(filter_countries))]
    
    # Append the filtered data to the final DataFrame
    final_df = pd.concat([final_df, filtered_df], ignore_index=True)

    # After all files are processed, calculate the access year outside of the loop
    # Calculate the total access by summing 'Limited', 'Basic', and 'SafelyManaged' for each country and year
    pivot_df = final_df.pivot_table(index=['Country', 'Year'], columns='Status', values='Value', aggfunc=np.sum).reset_index()
    
    # Add up the relevant columns to get total access percentage
    pivot_df['TotalAccess'] = pivot_df[['Basic', 'SafelyManaged']].sum(axis=1)
    
    # Now determine the first year where 'TotalAccess' exceeds 99% for each country
    access_year_df = pivot_df[pivot_df['TotalAccess'] > 99].groupby('Country')['Year'].min().reset_index()
    access_year_df.rename(columns={'Year': 'YearOf99PctAccess'}, inplace=True)
    
        # Merge all_countries_access with the years found using a left join
    all_countries_access = all_countries_access.merge(access_year_df, on='Country', how='left')
    
    # Where YearOf99PctAccess is NaN, replace with "after 2100"
    all_countries_access['YearOf99PctAccess'].fillna('after 2100', inplace=True)
    
    # Merge all
    water_sanitation = file.stem.split("_")[0]
    
# Now export access_year_df as a new file
new_file_path = Path(out_folder) / f'access_year_{water_sanitation}.csv'
all_countries_access.to_csv(new_file_path, index=False)
        
