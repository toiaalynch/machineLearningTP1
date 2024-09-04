import numpy as np
from sklearn.preprocessing import StandardScaler

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
    data = handle_missing_values(data, 3)    
    categories = ['Hilux SW4', 'Corolla Cross', 'RAV4']
    data = one_hot_encode(data, 1, categories)
    year_column = data[:, 4].astype(float).reshape(-1, 1)
    fuel_categories = ['Nafta', 'Diésel', 'Nafta/GNC', 'Híbrido/Nafta', 'Eléctrico']
    data = one_hot_encode(data, 5, fuel_categories)
    transmission_categories = ['Automática', 'Manual', 'Automática secuencial']
    data = one_hot_encode(data, 10, transmission_categories)
    data = process_motor(data, 13)
    data = preprocess_kilometers(data, 14)
    seller_categories = ['concesionaria', 'particular', 'tienda']
    data = one_hot_encode(data, 15, seller_categories)
    return data

def normalize_data(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data)


dev_data = np.genfromtxt('/Users/victoria/Desktop/5tocuatrimestre/ml/tps/tp2ml/machineLearning/data/raw/toyota_dev.csv', delimiter=',', skip_header=1, dtype=str)
test_data = np.genfromtxt('/Users/victoria/Desktop/5tocuatrimestre/ml/tps/tp2ml/machineLearning/data/raw/toyota_test.csv', delimiter=',', skip_header=1, dtype=str)

dev_data_processed = preprocess_data(dev_data)
test_data_processed = preprocess_data(test_data)

np.savetxt('/Users/victoria/Desktop/5tocuatrimestre/ml/tps/tp2ml/machineLearning/data/processed/toyota_dev_processed.csv', dev_data_processed, delimiter=',', fmt='%s')
np.savetxt('/Users/victoria/Desktop/5tocuatrimestre/ml/tps/tp2ml/machineLearning/data/processed/toyota_test_processed.csv', test_data_processed, delimiter=',', fmt='%s')

dev_data_normalized = normalize_data(dev_data_processed)
test_data_normalized = normalize_data(test_data_processed)

# Guardar los datos procesados y normalizados
np.savetxt('/Users/victoria/Desktop/5tocuatrimestre/ml/tps/tp2ml/machineLearning/data/processed/toyota_dev_processed_normalized.csv', dev_data_normalized, delimiter=',', fmt='%.6f')
np.savetxt('/Users/victoria/Desktop/5tocuatrimestre/ml/tps/tp2ml/machineLearning/data/processed/toyota_test_processed_normalized.csv', test_data_normalized, delimiter=',', fmt='%.6f')


