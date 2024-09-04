import numpy as np
import os

def handle_missing_values(data, color_column_index):
    # Remove the 'color' column by its index
    data = np.delete(data, color_column_index, axis=1)
    
    # Drop rows with any missing values
    mask = ~np.any(data == '', axis=1)
    data = data[mask]
    
    return data

def one_hot_encode(data, column_index, categories):
    # Create the one-hot encoded matrix
    one_hot = np.zeros((data.shape[0], len(categories)))
    
    # Perform the one-hot encoding
    for i, category in enumerate(categories):
        one_hot[:, i] = (data[:, column_index] == category).astype(int)
    
    # Remove the original column
    data_without_column = np.delete(data, column_index, axis=1)
    # Combine the one-hot encoded columns with the rest of the data
    data_with_one_hot = np.hstack([data_without_column[:, :column_index], one_hot, data_without_column[:, column_index:]])
    
    return data_with_one_hot

def preprocess_kilometers(data, kilometers_column_index):
    # Process the 'km' column
    cleaned_kilometers = np.array([float(value.replace(' km', '').replace(',', '')) for value in data[:, kilometers_column_index]])
    
    # Replace the 'km' column in place with the cleaned kilometers
    data[:, kilometers_column_index] = cleaned_kilometers

    return data


def extract_numeric_value(value):
    number_part = ''.join([char for char in value if char.isdigit() or char == '.'])
    
    try:
        return float(number_part)
    except ValueError:
        return None

def process_motor(data, motor_column_index):
    valid_rows = []

    for i, value in enumerate(data[:, motor_column_index]):
        motor_value = extract_numeric_value(value)
        
        if motor_value is not None:
            data[i, motor_column_index] = motor_value
            valid_rows.append(data[i])  

    valid_data = np.array(valid_rows)

    return valid_data


def preprocess_data(data):
    # Step 1: Handle missing values and remove 'color' column (assuming 'color' is at index 3)
    data = handle_missing_values(data, 3)
    
    # Step 2: Keep 'id' column as it is (assuming 'id' is at index 0)
    
    # Step 3: Perform one-hot encoding on 'tipo' (assuming 'tipo' is at index 1)
    categories = ['Hilux SW4', 'Corolla Cross', 'RAV4']
    data = one_hot_encode(data, 1, categories)
    
     # Step 4: Extract the 'year' column (index 4), convert it to float, and reshape it
    year_column = data[:, 4].astype(float).reshape(-1, 1)
    
    # Step 5: Perform one-hot encoding on 'fuel_type' (assuming 'fuel_type' is at index 5)
    fuel_categories = ['Nafta', 'Diésel', 'Nafta/GNC', 'Híbrido/Nafta', 'Eléctrico']
    data = one_hot_encode(data, 5, fuel_categories)

    # Step 6: Perform one-hot encoding on 'transmission' (assuming 'transmission' is at index 10)
    transmission_categories = ['Automática', 'Manual', 'Automática secuencial']
    data = one_hot_encode(data, 10, transmission_categories)

    # Step 7: Process 'motor' (assuming 'motor' is at index 13)
    data = process_motor(data, 13)

    # Step 8: Process 'km' (assuming 'km' is at index 14)
    data = preprocess_kilometers(data, 14)

    # Step 9: Perform one-hot encoding on 'seller_type' (assuming 'seller_type' is at index 15)
    seller_categories = ['concesionaria', 'particular', 'tienda']
    data = one_hot_encode(data, 15, seller_categories)

    # Step 10: Keep 'price' column as it is (assuming 'price' is at index 18)


    return data



# Load and preprocess data
dev_data = np.genfromtxt('/Users/victoria/Desktop/5tocuatrimestre/ml/tps/tp2ml/machineLearning/data/raw/toyota_dev.csv', delimiter=',', skip_header=1, dtype=str)
test_data = np.genfromtxt('/Users/victoria/Desktop/5tocuatrimestre/ml/tps/tp2ml/machineLearning/data/raw/toyota_test.csv', delimiter=',', skip_header=1, dtype=str)

# Process data
dev_data_processed = preprocess_data(dev_data)
test_data_processed = preprocess_data(test_data)

# Save processed data
np.savetxt('/Users/victoria/Desktop/5tocuatrimestre/ml/tps/tp2ml/machineLearning/data/processed/toyota_dev_processed.csv', dev_data_processed, delimiter=',', fmt='%s')
np.savetxt('/Users/victoria/Desktop/5tocuatrimestre/ml/tps/tp2ml/machineLearning/data/processed/toyota_test_processed.csv', test_data_processed, delimiter=',', fmt='%s')

