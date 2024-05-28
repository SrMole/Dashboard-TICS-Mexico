import os
import geopandas as gpd
import matplotlib.pyplot as plt
import streamlit as st

def mapa(dataframes):

    # Función para graficar el mapa de calor para un año y tipo de TIC específicos
    def plot_map(year, tic_type):

        directorio_actual = os.getcwd()
        csv_dir = os.path.join(directorio_actual, 'CSV')
        if directorio_actual != 'C:\\Users\\Emilio\\Desktop\\streamlit\\CSV':
            os.chdir(csv_dir)

        # Leer el archivo shapefile de México
        mexico_shapefile = gpd.read_file('Shapefile/gdb_ref_esta_ine_2009.shp')

        # Seleccionar el DataFrame correspondiente al tipo de TIC
        if tic_type == 'Computadora':
            tic_df = dataframes['TICS_Computadora_Entidad']
        elif tic_type == 'Internet':
            tic_df = dataframes['TICS_Internet_Entidad']
        elif tic_type == 'Celular':
            tic_df = dataframes['TICS_Celular_Entidad']
        else:
            st.error("Tipo de TIC no válido.")
            return
        
        # Fusionar el shapefile con el DataFrame para el año dado
        merged_data = mexico_shapefile.merge(tic_df[[year]], left_on='abrv_edo', right_index=True)

        # Graficar el mapa de calor para el año dado
        fig, ax = plt.subplots(figsize=(10, 6))  # Ajusta el tamaño de la figura
        fig.patch.set_alpha(0)  # Hacer la figura transparente
        ax = merged_data.plot(column=year, cmap='Greens', linewidth=0.8, ax=ax, edgecolor='0.8', 
                            legend=True, vmin=25, vmax=90)
        ax.axis('off')  # Eliminar ejes x e y
        ax.set_facecolor('none')  # Hacer el fondo del mapa transparente

        # Personalizar la leyenda
        legend = ax.get_legend()
        if legend:
            legend.set_bbox_to_anchor((1, 0.5))
            frame = legend.get_frame()
            frame.set_height(1)
            frame.set_alpha(0)  # Hacer el fondo de la leyenda transparente
            frame.set_edgecolor('none')  # Sin borde

            for text in legend.get_texts():
                text.set_color('white')
                text.set_fontweight('bold')

        # Personalizar los números de la barra de colores
        colorbar = ax.get_figure().get_axes()[1]
        colorbar.tick_params(colors='white', labelsize='medium')  # Ajusta el tamaño de la letra
        for text in colorbar.get_yticklabels():
            text.set_fontweight('bold')

        # Deshabilitar el ajuste automático de tamaño en Streamlit
        st.set_option('deprecation.showPyplotGlobalUse', False)
        # Mostrar el gráfico en Streamlit
        st.pyplot(fig)

    # CSS para personalizar los botones y texto
    st.markdown('''
        <style>
            .stRadio > div {flex-direction: row;}
            .stRadio > div > label {
                color: white;
                font-weight: bold;
                transition: all 0.2s ease;
                cursor: pointer;
                margin-right: 5px;
                margin-bottom: 5px; /* Reducir el margen inferior */
                margin-left: 5px; /* Agregar un margen lateral */
                background-color: transparent; /* Cambio el fondo del botón a transparente */
                border: 2px solid white; /* Añado un borde blanco */
                border-radius: 5px; /* Añado bordes redondeados */
                padding: 5px 5px; /* Añado espaciado interno para mejorar la apariencia */
                text-align: center; /* Centrar el texto */
            }
            .stRadio > div > label > div {
                color: white;
                font-weight: bold;
            }
            .stRadio > div > label:hover {
                color: #4CAF50;
                background-color: transparent; /* Cambio el fondo del botón a transparente */
                border-color: #4CAF50; /* Cambio el color del borde al hacer hover */
            }
            .stRadio > div > label input[checked] + div {
                background-color: transparent;  /* Cambio el color de fondo del botón seleccionado a transparente */
                color: white;
                border-color: #4CAF50; /* Cambio el color del borde del botón seleccionado */
            }
            body {
                background-color: #000;
            }
        </style>
    ''', unsafe_allow_html=True)

    st.title("Mapa de Calor de Mexico - Uso de TIC")

    # Dividir la página en dos columnas
    col1, col2 = st.columns([1, 1])

    with col1:
        with st.container():
            year_buttons = st.radio("###### **Año:**", ['2015', '2016', '2017', '2018', '2020', '2021', '2022'], horizontal=True)
    with col2:
        with st.container():
            tic_buttons = st.radio("###### **TIC:**", ['Computadora', 'Internet', 'Celular'], horizontal=True)

    # Llamar a la función plot_map cuando se cambian los valores de los widgets
    plot_map(year_buttons, tic_buttons)

    directorio_actual = os.getcwd()
    streamlit_dir = os.path.abspath(os.path.join(directorio_actual, '..'))  # Subir un nivel
    os.chdir(streamlit_dir)
