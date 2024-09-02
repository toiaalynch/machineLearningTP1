import numpy as np




def handle_missing_values(data):
    """
    Elimina las filas que contienen valores faltantes en cualquier columna.

    :param data: Un array de NumPy con los datos.
    :return: Array de NumPy con las filas sin valores faltantes.
    """
    # Crear una máscara para las filas que no tienen valores faltantes
    mask = np.all(data != '', axis=1)
    
    # Aplicar la máscara para filtrar las filas
    data = data[mask]
    
    return data

def one_hot_encoder(data, column_index, unique_values=None):
    """
    Aplica One-Hot Encoding a una columna categórica en un array de NumPy.

    :param data: Un array de NumPy con los datos.
    :param column_index: Índice de la columna a la que se aplicará One-Hot Encoding.
    :param unique_values: Valores únicos a codificar. Si es None, se calcularán a partir de los datos.
    :return: Nuevo array de NumPy con la columna original reemplazada por sus correspondientes dummies.
    """
    if unique_values is None:
        unique_values = np.unique(data[:, column_index])
        
    one_hot = np.zeros((data.shape[0], len(unique_values)))
    
    for i, unique_value in enumerate(unique_values):
        one_hot[:, i] = (data[:, column_index] == unique_value).astype(int)
    
    # Eliminar la columna original y agregar las columnas one-hot
    data = np.delete(data, column_index, axis=1)
    data = np.hstack((data, one_hot))
    
    return data

def preprocess_kilometers_column(data, kilometers_column_index):
    """
    Limpia la columna 'Kilómetros', eliminando ' km' y convirtiendo a float.
    
    :param data: Un array de NumPy con los datos.
    :param kilometers_column_index: Índice de la columna de kilómetros.
    :return: Array de NumPy con la columna de kilómetros limpiada y convertida a float.
    """
    cleaned_kilometers = []
    for value in data[:, kilometers_column_index]:
        if isinstance(value, str):
            cleaned_value = value.replace(' km', '').replace(',', '')
            try:
                cleaned_value = float(cleaned_value)
            except ValueError:
                cleaned_value = np.nan
        else:
            cleaned_value = float(value)
        
        cleaned_kilometers.append(cleaned_value)
    
    data[:, kilometers_column_index] = np.array(cleaned_kilometers, dtype=float)
    return data


def preprocess_data(data):
    """
    Realiza el preprocesamiento completo de los datos según las especificaciones dadas.

    :param data: Un array de NumPy con los datos.
    :return: Array de NumPy con los datos preprocesados.
    """

# id,Tipo,Año,Color,Tipo de combustible,Transmisión,Motor,Kilómetros,Tipo de vendedor,Precio

    # Eliminar filas con valores faltantes
    data = handle_missing_values(data)

    # Convertir la columna 'Tipo' a valores numéricos
    # type_map = {'Hilux SW4': 1, 'Corolla Cross': 2, 'Otro': 3}
    # data[:, 1] = np.array([type_map.get(x, 3) for x in data[:, 1]])

    # # One-Hot Encoding para 'Color'
    # data = one_hot_encoder(data, column_index=3)

    # # One-Hot Encoding para 'Tipo de Combustible'
    # # One-Hot Encoding para 'Tipo de Combustible'
    # fuel_map = ['Nafta', 'Diésel', 'Híbrido', 'Nafta/Híbrido']  # Asegúrate de que estos valores sean los únicos posibles
    # data = one_hot_encoder(data, column_index=4, unique_values=fuel_map)

    # # One-Hot Encoding para 'Transmisión'
    # transmission_map = ['Automática', 'Manual']  # Asegúrate de que estos valores sean los únicos posibles
    # data = one_hot_encoder(data, column_index=5, unique_values=transmission_map)

    # # cambio para para 'motor'
    # data[:, 6] = np.where(data[:, 6] == 'INYECCION MULTI PUNTO', '5', data[:, 6])

    # # Limpiar la columna 'Kilómetros', eliminando ' km' y convirtiendo a float
    # data = preprocess_kilometers_column(data, 7)  # Índice de la columna "Kilómetros"

    # # One-Hot Encoding para 'Tipo de Vendedor'
    # seller_type_map = ['particular', 'concesionaria', 'tienda']  # Asegúrate de que estos valores sean los únicos posibles
    # data = one_hot_encoder(data, column_index=8, unique_values=seller_type_map)

    return data

