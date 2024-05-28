import streamlit as st
import geopandas as gpd
from funciones.Leer_csv import leer_csv
from funciones.Graficas import histograma, serie_temporal, descomposicion, diferenciacion, suavizado, autocorrelacion, graficas_segun, arima, arima_residuos, sarima, sarima_residuos, reg_lineal
from funciones.Mapa import mapa
import os

# Funcion main
def main():
    
    # Configurar el ancho y el alto de la página
    st.set_page_config(
        layout="wide",  # Puedes elegir "wide" para un ancho completo o "centered" para centrar el contenido
        page_title="Dashboard",  # Título de la página en el navegador
        page_icon=":chart_with_upwards_trend:",  # Icono de la página en el navegador
        initial_sidebar_state="expanded",  # Puedes elegir "expanded" para que la barra lateral esté expandida inicialmente
        menu_items={
            "Get Help": None,
            "Report a bug": "https://github.com/streamlit/streamlit/issues",
            "About": "https://streamlit.io/about",
        }
    )

    # Leer datasets
    dataframes, dispositivo = leer_csv()

    # Crear la barra lateral con título centrado
    st.sidebar.markdown(
        "<h1 class='centered-title'>Tendencias del Uso de las TICS en México</h1>",
        unsafe_allow_html=True
    )
    for i in range(3): st.sidebar.markdown("")

    # Establecer el estilo de la barra lateral
    st.markdown(
        """
        <style>
        /* Centrar el título de la barra lateral */
        .centered-title {
            text-align: left;
            font-weight: bold;
            color: #ffffff; /* Texto en blanco */
            font-family: Arial;
            margin-top: 50px; /* Agregar margen superior grande */
        }
        /* Estilo para el contenido de la barra lateral */
        .css-1d391kg .css-1v3fvcr {
            background-color: #ffffff; /* Fondo blanco */
        }
        /* Estilo para los enlaces */
        .centered-link {
            display: block;
            text-align: left; /* Alinear texto a la izquierda */
            color: #ffffff !important; /* Texto en blanco */
            text-decoration: none !important;
            font-weight: bold;
            margin-bottom: 1rem;
        }
        .centered-link:hover {
            color: #2C6E49 !important; /* Puedes agregar un color de hover si lo deseas */
        }
        /* Estilo para los subheaders */
        .css-1d391kg .css-1v3fvcr .subheader {
            color: #ffffff; /* Texto en blanco */
        }
        /* Estilo para la lista */
        .custom-list {
            padding-left: 0;
            list-style-type: none;
            margin-top: 0;
        }
    </style>

        """,
        unsafe_allow_html=True
    )

    # Agregar enlaces a cada gráfica en la barra lateral
    st.sidebar.markdown("<ul class='custom-list'><li><a href='#mapa-de-calor-de-mexico-uso-de-tic' class='centered-link'>Mapa de Calor</a></li></ul>", unsafe_allow_html=True)
    st.sidebar.markdown("<ul class='custom-list'><li><a href='#histogramas-uso-de-tics' class='centered-link'>Histogramas</a></li></ul>", unsafe_allow_html=True)
    st.sidebar.markdown("<ul class='custom-list'><li><a href='#series-temporales-uso-de-tics' class='centered-link'>Series Temporales</a></li></ul>", unsafe_allow_html=True)
    st.sidebar.markdown("<ul class='custom-list'><li><a href='#deteccion-de-anomalias-uso-de-tics' class='centered-link'>Anomalias</a></li></ul>", unsafe_allow_html=True)
    st.sidebar.markdown("<ul class='custom-list'><li><a href='#diferenciacion-uso-de-tics' class='centered-link'>Diferenciacion</a></li></ul>", unsafe_allow_html=True)
    st.sidebar.markdown("<ul class='custom-list'><li><a href='#suavizado-exponencial-uso-de-tics' class='centered-link'>Suavizado</a></li></ul>", unsafe_allow_html=True)
    st.sidebar.markdown("<ul class='custom-list'><li><a href='#autocorrelacion-y-autocorrelacion-parcial-uso-de-tics' class='centered-link'>Autocorrelacion y Autocorrelacion Parcial</a></li></ul>", unsafe_allow_html=True)
    st.sidebar.markdown("<ul class='custom-list'><li><a href='#graficas-de-barras-filtros-uso-de-tic' class='centered-link'>Graficas de Barras</a></li></ul>", unsafe_allow_html=True)
    st.sidebar.markdown("<ul class='custom-list'><li><a href='#predicciones-de-arima-uso-de-tics' class='centered-link'>ARIMA</a></li></ul>", unsafe_allow_html=True)
    st.sidebar.markdown("<ul class='custom-list'><li><a href='#residuos-de-arima-uso-de-tics' class='centered-link'>ARIMA Residuos</a></li></ul>", unsafe_allow_html=True)
    st.sidebar.markdown("<ul class='custom-list'><li><a href='#predicciones-de-sarima-uso-de-tics' class='centered-link'>SARIMA</a></li></ul>", unsafe_allow_html=True)
    st.sidebar.markdown("<ul class='custom-list'><li><a href='#residuos-de-sarima-uso-de-tics' class='centered-link'>SARIMA Residuos</a></li></ul>", unsafe_allow_html=True)
    st.sidebar.markdown("<ul class='custom-list'><li><a href='#predicciones-de-regresion-lineal-uso-de-tics' class='centered-link'>Regresion Lineal</a></li></ul>", unsafe_allow_html=True)

    for i in range(10): st.sidebar.markdown("")
    st.sidebar.markdown("### **Integrantes**")
    st.sidebar.markdown('''
                        **Manzano Morales Jesús Emilio**
                        **Sanchez Heredia Ana Isabel**
                        **Rafael Zamora Guerrero**
            ''')

    mapa(dataframes)
    for i in range(8): st.write("")
    histograma(dataframes)
    for i in range(8): st.write("")
    serie_temporal(dataframes)
    for i in range(8): st.write("")
    descomposicion(dataframes)
    for i in range(8): st.write("")
    diferenciacion(dataframes)
    for i in range(8): st.write("")
    suavizado(dataframes)
    for i in range(8): st.write("")
    autocorrelacion(dataframes)
    for i in range(8): st.write("")
    graficas_segun(dataframes, dispositivo)
    for i in range(8): st.write("")
    arima(dataframes)
    for i in range(8): st.write("")
    arima_residuos(dataframes)
    for i in range(8): st.write("")
    sarima(dataframes)
    for i in range(8): st.write("")
    sarima_residuos(dataframes)
    for i in range(8): st.write("")
    reg_lineal(dataframes)

# Llamar main
if __name__ == "__main__":
    main()
