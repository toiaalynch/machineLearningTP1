import numpy as np

def handle_missing_values(data, strategy='mean', columns=None, fill_value=None):
    """
    Maneja valores faltantes en un array de NumPy.

    :param data: Un array de NumPy con los datos.
    :param strategy: Estrategia para manejar los valores faltantes. 
                     'mean', 'median', 'mode', 'constant', o 'drop'.
    :param columns: Lista de índices de columnas a las que se aplica la estrategia. Si es None, se aplica a todo el array.
    :param fill_value: Valor para rellenar si la estrategia es 'constant'.
    :return: Array de NumPy con los valores faltantes manejados.
    """
    if columns is None:
        columns = np.arange(data.shape[1])
    
    for col in columns:
        col_data = data[:, col]
        mask = np.isnan(col_data)
        
        if strategy == 'mean':
            data[mask, col] = np.nanmean(col_data)
        elif strategy == 'median':
            data[mask, col] = np.nanmedian(col_data)
        elif strategy == 'mode':
            # Nota: np.nanmode no existe, tendrías que implementarlo o usar una alternativa
            mode_value = np.bincount(col_data[~mask].astype(int)).argmax()
            data[mask, col] = mode_value
        elif strategy == 'constant':
            if fill_value is None:
                raise ValueError("fill_value must be specified for 'constant' strategy")
            data[mask, col] = fill_value
        elif strategy == 'drop':
            data = data[~mask]
        else:
            raise ValueError("Unknown strategy type")
    
    return data

def one_hot_encoder(data, column_index):
    """
    Aplica One-Hot Encoding a una columna categórica en un array de NumPy.

    :param data: Un array de NumPy con los datos.
    :param column_index: Índice de la columna a la que se aplicará One-Hot Encoding.
    :return: Nuevo array de NumPy con la columna original reemplazada por sus correspondientes dummies.
    """
    unique_values = np.unique(data[:, column_index])
    one_hot = np.zeros((data.shape[0], len(unique_values)))
    
    for i, unique_value in enumerate(unique_values):
        one_hot[:, i] = (data[:, column_index] == unique_value).astype(int)
    
    # Eliminar la columna original y agregar las columnas one-hot
    data = np.delete(data, column_index, axis=1)
    data = np.hstack((data, one_hot))
    
    return data

def normalize(data, columns=None):
    """
    Normaliza las columnas especificadas del array de NumPy para que estén en el rango [0, 1].

    :param data: Un array de NumPy con los datos.
    :param columns: Lista de índices de columnas a normalizar. Si es None, se normalizan todas las columnas.
    :return: Array de NumPy con las columnas normalizadas.
    """
    if columns is None:
        columns = np.arange(data.shape[1])
    
    for col in columns:
        col_min = np.min(data[:, col])
        col_max = np.max(data[:, col])
        data[:, col] = (data[:, col] - col_min) / (col_max - col_min)
    
    return data
