import pandas as pd
import numpy as np
import os


# Funcion para crear la barra lateral
def leer_csv():

    # Listas de dispositivos y temáticas
    dispositivos = ['Celular', 'Computadora', 'Internet']
    orientacion = ['Economia', 'Edad', 'Escolaridad', 'Frecuencia_Uso', 'Sexo', 'Miscelaneo']

    # Generar la lista de nombres de archivos CSV
    csv_files = [f'TICS_{dispositivo}_{tema}.csv' for dispositivo in dispositivos for tema in orientacion]
    # Insertar los nombres de los archivos específicos al principio de la lista
    csv_files.insert(0, 'TICS_General.csv')
    csv_files.insert(1, 'TICS_Computadora_Entidad.csv')
    csv_files.insert(2, 'TICS_Internet_Entidad.csv')
    csv_files.insert(3, 'TICS_Celular_Entidad.csv')

    # Lista para almacenar los nombres de los dataframes
    dataframe_names = []

    # Diccionario para almacenar los dataframes con sus nombres
    dataframes = {}

    # Función para leer los CSV y guardarlos como dataframes
    def load_csv_files(csv_files):

        directorio_actual = os.getcwd()
        csv_dir = os.path.join(directorio_actual, 'CSV')
        if directorio_actual != r'C:\Users\Emilio\Desktop\streamlit\CSV':
            os.chdir(csv_dir)

        print(os.getcwd())

        for file in csv_files:
            # Crear un nombre para el dataframe a partir del nombre del archivo sin la extensión
            dataframe_name = file.split('.')[0]
            # Leer el CSV y guardarlo como un dataframe
            df = pd.read_csv(file, index_col=0)
            # Almacenar el dataframe en el diccionario
            dataframes[dataframe_name] = df
            # Guardar el nombre del dataframe en la lista
            dataframe_names.append(dataframe_name)

    # Llamar a la función para cargar los archivos CSV
    load_csv_files(csv_files)

    def process_dataframes(dataframes):
        def convert_spaced_int(value):
            if isinstance(value, str) and ' ' in value:
                return int(''.join(value.split()))
            return int(value)

        total_null_values = 0

        for name, df in dataframes.items():
            # Contar valores nulos antes de cualquier operación
            null_values_count = df.isnull().sum().sum()
            total_null_values += null_values_count

            # Reemplazar valores nulos por 0
            df.fillna(0, inplace=True)

            # Convertir todos los valores a enteros usando map en lugar de applymap
            if 'TICS_Computadora_Entidad' not in name and 'TICS_Internet_Entidad' not in name and 'TICS_Celular_Entidad' not in name:
                df = df.map(convert_spaced_int)

            # Eliminar la columna "total"
            if 'total' in df.columns:
                df.drop(columns=['total'], inplace=True)

            # Realizar otras transformaciones necesarias para la estacionariedad
            # (Por ejemplo: eliminación de valores atípicos, tratamiento de datos faltantes, etc.)

            # Actualizar el dataframe en el diccionario
            dataframes[name] = df

    # Llamar a la función para procesar los dataframes
    process_dataframes(dataframes)

    directorio_actual = os.getcwd()
    streamlit_dir = os.path.abspath(os.path.join(directorio_actual, '..'))  # Subir un nivel
    os.chdir(streamlit_dir)

    return dataframes, dispositivos
