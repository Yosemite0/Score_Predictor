import os
import glob
import json
import pandas as pd

def main():
    # Get the folder one level up; adjust path relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.join(script_dir, "..")
    
    # Find all CSV files in the parent directory
    csv_pattern = os.path.join(parent_dir, "*.csv")
    csv_files = glob.glob(csv_pattern)
    
    desc = {}
    
    for csv_file in csv_files:
        try:
            # Read a sample to get column names
            df = pd.read_csv(csv_file, nrows=100)
        except Exception as e:
            print(f"Error processing '{csv_file}': {e}")
            continue
        
        # Just create a list of column names
        columns_list = list(df.columns)
        
        # Use only the CSV file name (not the full path)
        csv_basename = os.path.basename(csv_file)
        desc[csv_basename] = columns_list
    
    # Save collected description data into desc.json in this script's directory
    output_path = os.path.join(script_dir, "desc.json")
    with open(output_path, "w") as out_file:
        json.dump(desc, out_file, indent=4)
    
    print(f"Column descriptions saved to: {output_path}")

if __name__ == "__main__":
    main()
