import numpy as np

def handle_missing_values(data):
    mask = np.all(data != '', axis=1)
    data = data[mask]
    return data

def one_hot_encode(data, column_index, categories):
    one_hot = np.zeros((data.shape[0], len(categories)))
    for i, category in enumerate(categories):
        one_hot[:, i] = np.array([1 if category in value else 0 for value in data[:, column_index]])
    return one_hot

def preprocess_kilometers(data, kilometers_column_index):
    cleaned_kilometers = np.array([float(value.replace(' km', '').replace(',', '')) for value in data[:, kilometers_column_index]])
    return cleaned_kilometers.reshape(-1, 1)

# def preprocess_motor(data, motor_column_index):
#     cleaned_motor = []
#     for value in data[:, motor_column_index]:
#         if "INYECCION MULTI PUNTO" in value:
#             cleaned_motor.append(5)
#         else:
#             number_part = ''.join(filter(str.isdigit, value))
#             cleaned_motor.append(float(number_part) if number_part else np.nan)
#     return np.array(cleaned_motor).reshape(-1, 1)

def preprocess_motor(data, motor_column_index):
    cleaned_motor = []
    for value in data[:, motor_column_index]:
        if "INYECCION MULTI PUNTO" in value:
            cleaned_motor.append(5.0)
        else:
            number_part = ''.join([char for char in value if char.isdigit() or char == '.'])
            cleaned_motor.append(float(number_part) if number_part else np.nan)
    return np.array(cleaned_motor).reshape(-1, 1)

def preprocess_data(data):
    data = handle_missing_values(data)
    
    # Keep id and year as is
    processed_data = data[:, [0, 2]].astype(float)
    
    # Convert 'Tipo' to numeric values
    type_map = {'Hilux SW4': 1, 'Corolla Cross': 2, 'Otro': 3}
    tipo = np.array([type_map.get(value, 3) for value in data[:, 1]]).reshape(-1, 1)
    processed_data = np.hstack((processed_data, tipo))
    
    # One-Hot Encoding for 'Color'
    color_categories = ['Plateado', 'Blanco', 'Gris', 'Negro', 'Marrón', 'Rojo', 'Gris oscuro', 'Azul']
    color_encoded = one_hot_encode(data, 3, color_categories)
    processed_data = np.hstack((processed_data, color_encoded))
    
    # One-Hot Encoding for 'Tipo de Combustible'
    fuel_categories = ['Nafta', 'Diésel', 'GNC', 'Híbrido', 'Eléctrico']
    fuel_encoded = one_hot_encode(data, 4, fuel_categories)
    processed_data = np.hstack((processed_data, fuel_encoded))
    
    # One-Hot Encoding for 'Transmisión'
    transmission_categories = ['Automática', 'Manual']
    transmission_encoded = one_hot_encode(data, 5, transmission_categories)
    processed_data = np.hstack((processed_data, transmission_encoded))
    
    # Process 'Motor'
    motor_encoded = preprocess_motor(data, 6)
    processed_data = np.hstack((processed_data, motor_encoded))
    
    # Process 'Kilómetros'
    kilometers_encoded = preprocess_kilometers(data, 7)
    processed_data = np.hstack((processed_data, kilometers_encoded))
    
    # One-Hot Encoding for 'Tipo de Vendedor'
    seller_categories = ['concesionaria', 'particular', 'tienda']
    seller_encoded = one_hot_encode(data, 8, seller_categories)
    processed_data = np.hstack((processed_data, seller_encoded))
    
    # Add 'Precio'
    precio = data[:, 9].astype(float).reshape(-1, 1)
    processed_data = np.hstack((processed_data, precio))
    
    return processed_data

# Load and preprocess data
dev_data = np.genfromtxt('/Users/victoria/Desktop/5tocuatrimestre/ml/tps/tp2ml/machineLearning/data/raw/toyota_dev.csv', delimiter=',', skip_header=1, dtype=str)
test_data = np.genfromtxt('/Users/victoria/Desktop/5tocuatrimestre/ml/tps/tp2ml/machineLearning/data/raw/toyota_test.csv', delimiter=',', skip_header=1, dtype=str)

# Process data
dev_data_processed = preprocess_data(dev_data)
test_data_processed = preprocess_data(test_data)

# Save processed data
np.savetxt('/Users/victoria/Desktop/5tocuatrimestre/ml/tps/tp2ml/machineLearning/data/processed/toyota_dev_processed.csv', dev_data_processed, delimiter=',', fmt='%s')
np.savetxt('/Users/victoria/Desktop/5tocuatrimestre/ml/tps/tp2ml/machineLearning/data/processed/toyota_test_processed.csv', test_data_processed, delimiter=',', fmt='%s')
