import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_data(data_path, train_output_path, test_output_path, test_size=0.2, random_state=42):
    # Load data
    data = pd.read_csv(data_path, sep='\t')

    # Filter data
    positive_samples = data[data['Is_Nature_Product'] == 1]
    negative_samples = data[data['Is_Nature_Product'] == 0].sample(n=int(1.5 * len(positive_samples)), random_state=random_state)
    filtered_data = pd.concat([positive_samples, negative_samples])

    # Split the data into training and testing sets
    train_data, test_data = train_test_split(filtered_data, test_size=test_size, random_state=random_state)

    # Write out train_data and test_data to files
    train_data.to_csv(train_output_path, sep='\t', index=False)
    test_data.to_csv(test_output_path, sep='\t', index=False)

    return train_data, test_data