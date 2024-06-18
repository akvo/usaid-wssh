import pandas as pd
from pathlib import Path
import numpy as np
import re  # Import the regular expressions module

# Get the current script directory
script_dir = Path(__file__).resolve().parent

# Define the main folder and output folder relative to the script directory
main_folder = script_dir.parent 
output_folder = main_folder / "output data"

# Create the new output folder if it does not exist
output_folder.mkdir(parents=True, exist_ok=True)

# Correctly constructing the file path
file_path = main_folder / 'input data' / 'JMP' / 'JMP_2023_WLD.xlsx'

sheets = ['Water', 'Sanitation']

for sheet_name in sheets:
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)

    # Replace '-' with np.nan
    df.replace('-', np.nan, inplace=True)

    # Remove < or > symbols from all values, keeping only numbers
    df = df.replace(to_replace=[r'^<', r'^>'], value='', regex=True)

    # Manually set the first row as headers, combining with the second row if necessary.
    headers = []
    for col in range(len(df.columns)):
        header = str(df.iloc[0, col]) + ' - ' + str(df.iloc[1, col]) if pd.notnull(df.iloc[1, col]) else df.iloc[0, col]
        headers.append(header)

    # Drop the rows used for headers.
    df = df.drop(index=[0, 1])
    
    # Set the new headers.
    df.columns = headers
    
    # Add the 'Type' column filled with the sheet name.
    df['Type'] = sheet_name
    
    # Export the final DataFrame as a new CSV
    new_file_path = output_folder / f'JMP_{sheet_name}.csv'
    df.to_csv(new_file_path, index=False)
