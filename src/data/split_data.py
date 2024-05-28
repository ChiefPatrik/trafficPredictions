import os
import pandas as pd
from sklearn.model_selection import train_test_split

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, '..', '..', 'data')
merged_data_path = os.path.join(data_dir, 'merged')

# Iterate over all files in merged directory
for filename in os.listdir(merged_data_path):
    if filename.endswith('_data.csv'):
        if(filename == 'reference_data.csv'):
            continue
        # Read the file into a DataFrame
        df = pd.read_csv(os.path.join(merged_data_path, filename))

        # Split the DataFrame into train and test sets
        train, test = train_test_split(df, test_size=0.1, random_state=1234)

        base_filename = filename[:-9]    # Remove '_data.csv' (9 characters)

        # Save the train set as 'X_train.csv' and the test set as 'X_test.csv'
        train.to_csv(os.path.join(merged_data_path, f'{base_filename}_train.csv'), index=False)
        test.to_csv(os.path.join(merged_data_path, f'{base_filename}_test.csv'), index=False)

print("Data splited and saved successfully for all regions!")