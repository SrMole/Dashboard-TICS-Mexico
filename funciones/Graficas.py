import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.graph_objs as go
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
import warnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX

def histograma(dataframes):
    st.title("Histogramas - Uso de TICS")

    # Buscar el DataFrame 'TICS_General' en el diccionario
    df_general = dataframes.get('TICS_General')

    # Obtener las variables únicas (columnas) del DataFrame
    variables = df_general.columns

    # Dividir el espacio en tres columnas
    col1, col2, col3 = st.columns(3)

    # Iterar sobre las variables y generar un histograma para cada una
    for i, variable in enumerate(variables):
        # Extraer el nombre específico de la variable
        nombre_variable = variable.split('_')[-1]

        # Crear una nueva figura para cada variable
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df_general[variable], name=nombre_variable, marker_color='green', nbinsx=6, marker_line_width=1))
        fig.update_layout(title=f'Distribución de Usuarios con {nombre_variable.capitalize()} a lo largo de 2009 - 2022', xaxis_title=f'{nombre_variable.capitalize()}', yaxis_title='Frecuencia')

        # Determinar en qué columna colocar el histograma
        if i % 3 == 0:
            col = col1
        elif i % 3 == 1:
            col = col2
        else:
            col = col3

        # Mostrar el histograma en la columna correspondiente
        col.plotly_chart(fig, use_container_width=True)

def serie_temporal(dataframes):

    st.title("Series Temporales - Uso de TICS")

    # Buscar el DataFrame 'TICS_General' en el diccionario
    df_general = dataframes.get('TICS_General')

    # Obtener las variables únicas (columnas) del DataFrame
    variables = df_general.columns

    # Dividir el espacio en tres columnas
    col1, col2, col3 = st.columns(3)

    # Iterar sobre las variables y generar una gráfica para cada una
    for i, variable in enumerate(variables):
        # Extraer el nombre específico de la variable
        nombre_variable = variable.split('_')[-1]

        # Crear una nueva figura para cada variable
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_general.index, y=df_general[variable], mode='lines+markers', name=nombre_variable, line=dict(color='green')))
        fig.update_layout(title=f'Usuarios con {nombre_variable.capitalize()} por año a nivel Nacional', xaxis_title='Año', yaxis_title='Cantidad de usuarios')

        # Determinar en qué columna colocar la gráfica
        if i % 3 == 0:
            col = col1
        elif i % 3 == 1:
            col = col2
        else:
            col = col3

        # Mostrar la gráfica en la columna correspondiente
        col.plotly_chart(fig, use_container_width=True)

def descomposicion(dataframes):
    st.title("Deteccion de Anomalias - Uso de TICS")

    # Seleccionar el DataFrame 'TICS_General'
    TICS_General = dataframes['TICS_General']

    # Convertir el índice a tipo datetime con el formato especificado
    TICS_General.index = pd.to_datetime(TICS_General.index, format='%Y')

    # Crear una figura de Plotly vacía
    fig = go.Figure()

    # Diccionario para almacenar los nombres originales y los nombres simplificados
    def capitalize_words(name):
        return ' '.join([word.capitalize() for word in name.split('_')])

    column_mapping = {col: capitalize_words(col) for col in TICS_General.columns}

    # Iterar sobre las columnas de TICS_General
    for tic in TICS_General.columns:
        # Suavizado exponencial para obtener la tendencia
        model = ExponentialSmoothing(TICS_General[tic], trend='add', seasonal=None).fit(optimized=True)
        tendencia = model.fittedvalues

        # Calcular los residuos
        residuales = TICS_General[tic] - tendencia

        # Detectar anomalías
        std_residuales = np.std(residuales)
        mean_residuales = np.mean(residuales)
        umbral_superior = mean_residuales + 2 * std_residuales
        umbral_inferior = mean_residuales - 2 * std_residuales
        anomalías = residuales[(residuales > umbral_superior) | (residuales < umbral_inferior)]

        # Agregar la serie temporal original al gráfico
        fig.add_trace(go.Scatter(
            x=TICS_General.index, y=TICS_General[tic], mode='lines',
            name=f'Serie Temporal - {column_mapping[tic]}', line=dict(color='green'), visible=(tic == list(TICS_General.columns)[0])
        ))

        # Agregar la tendencia al gráfico
        fig.add_trace(go.Scatter(
            x=TICS_General.index, y=tendencia, mode='lines',
            name=f'Tendencia - {column_mapping[tic]}', line=dict(color='orange'), visible=(tic == list(TICS_General.columns)[0])
        ))

        # Agregar los residuos al gráfico
        fig.add_trace(go.Scatter(
            x=TICS_General.index, y=residuales, mode='lines',
            name=f'Residuales - {column_mapping[tic]}', line=dict(color='green'), visible=(tic == list(TICS_General.columns)[0])
        ))

        # Agregar las anomalías al gráfico
        fig.add_trace(go.Scatter(
            x=anomalías.index, y=anomalías, mode='markers',
            name=f'Anomalías - {column_mapping[tic]}', marker=dict(color='red'), visible=(tic == list(TICS_General.columns)[0])
        ))

        # Agregar los umbrales al gráfico
        fig.add_trace(go.Scatter(
            x=TICS_General.index, y=[umbral_superior]*len(TICS_General), mode='lines',
            name=f'Umbral Superior - {column_mapping[tic]}', line=dict(color='red', dash='dash'), visible=(tic == list(TICS_General.columns)[0])
        ))

        fig.add_trace(go.Scatter(
            x=TICS_General.index, y=[umbral_inferior]*len(TICS_General), mode='lines',
            name=f'Umbral Inferior - {column_mapping[tic]}', line=dict(color='red', dash='dash'), visible=(tic == list(TICS_General.columns)[0])
        ))

    # Definir botones para cambiar la visibilidad de las series y actualizar el título
    buttons = []
    for tic in TICS_General.columns:
        visibility = []
        for trace in fig.data:
            if (trace.name.startswith(f'Serie Temporal - {column_mapping[tic]}') or
                trace.name.startswith(f'Tendencia - {column_mapping[tic]}') or
                trace.name.startswith(f'Residuales - {column_mapping[tic]}') or
                trace.name.startswith(f'Anomalías - {column_mapping[tic]}') or
                trace.name.startswith(f'Umbral Superior - {column_mapping[tic]}') or
                trace.name.startswith(f'Umbral Inferior - {column_mapping[tic]}')):
                visibility.append(True)
            else:
                visibility.append(False)

        buttons.append(dict(
            label=column_mapping[tic].split(' ')[-1],
            method='update',
            args=[{'visible': visibility},
                  {'title': f"<b>Detección de Anomalías en {column_mapping[tic]}</b>"}]
        ))

    # Establecer el título predeterminado
    default_title = f"<b>Detección de Anomalías en {column_mapping[list(TICS_General.columns)[0]]}</b>"
    fig.update_layout(
        title=default_title,
        updatemenus=[dict(
            type="dropdown",
            direction="down",
            buttons=buttons,
            showactive=True,
        )],
        xaxis_title="Año",
        yaxis_title="Cantidad de usuarios"
    )

    # Mostrar el gráfico en Streamlit
    st.plotly_chart(fig, use_container_width=True)


def diferenciacion(dataframes):
    st.title("Diferenciacion - Uso de TICS")

    # Seleccionar el DataFrame 'TICS_General'
    TICS_General = dataframes['TICS_General']

    # Convertir el índice a tipo datetime con el formato especificado
    TICS_General.index = pd.to_datetime(TICS_General.index, format='%Y')

    # Crear una figura de Plotly vacía
    fig = go.Figure()

    # Diccionario para almacenar los nombres originales y los nombres simplificados
    def capitalize_words(name):
        return ' '.join([word.capitalize() for word in name.split('_')])

    column_mapping = {col: capitalize_words(col) for col in TICS_General.columns}

    # Iterar sobre las columnas de TICS_General
    for tic in TICS_General.columns:
        # Obtener la serie temporal y su diferencia
        serie_temporal = TICS_General[tic]
        serie_diferenciada = serie_temporal.diff().dropna()

        # Agregar la serie temporal al gráfico
        fig.add_trace(go.Scatter(
            x=serie_temporal.index, y=serie_temporal, mode='lines', 
            name=f'Serie Temporal - {column_mapping[tic]}', line=dict(color='green'), visible=(tic == list(TICS_General.columns)[0])
        ))

        # Agregar la serie diferenciada al gráfico
        fig.add_trace(go.Scatter(
            x=serie_diferenciada.index, y=serie_diferenciada, mode='lines', 
            name=f'Diferenciada - {column_mapping[tic]}', line=dict(color='red'), visible=(tic == list(TICS_General.columns)[0])
        ))

    # Definir botones para cambiar la visibilidad de las series y actualizar el título
    buttons = []
    for tic in TICS_General.columns:
        visibility = []
        for trace in fig.data:
            if trace.name.startswith(f'Serie Temporal - {column_mapping[tic]}') or trace.name.startswith(f'Diferenciada - {column_mapping[tic]}'):
                visibility.append(True)
            else:
                visibility.append(False)

        buttons.append(dict(
            label=column_mapping[tic].split(' ')[-1],
            method='update',
            args=[{'visible': visibility},
                  {'title': f"<b>Diferenciación de {column_mapping[tic]}</b>"}]
        ))

    # Establecer el título predeterminado
    default_title = f"<b>Diferenciación de {column_mapping[list(TICS_General.columns)[0]]}</b>"
    fig.update_layout(
        title=default_title,
        updatemenus=[dict(
            type="dropdown",
            direction="down",
            buttons=buttons,
            showactive=True,
        )],
        xaxis_title="Año",
        yaxis_title="CAntidad de usuarios"
    )

    # Mostrar el gráfico en Streamlit
    st.plotly_chart(fig, use_container_width=True)

def suavizado(dataframes):
    st.title("Suavizado Exponencial - Uso de TICS")

    # Suponiendo que 'TICS_General' es tu DataFrame
    TICS_General = dataframes['TICS_General']

    # Crear una figura de Plotly
    fig = go.Figure()

    # Iterar sobre las columnas del DataFrame y aplicar el suavizado exponencial simple
    for i, columna in enumerate(TICS_General.columns):
        # Obtener el nombre para el botón y la serie temporal de la columna actual
        nombre_limpio = columna.split('_')[1].capitalize()  # Obtener el nombre sin 'usuario_'
        serie_temporal = TICS_General[columna]
        
        # Aplicar suavizado exponencial simple
        serie_suavizada = serie_temporal.ewm(alpha=0.2).mean()
        
        # Establecer visibilidad por defecto
        visible = 'legendonly' if i > 0 else True
        
        # Agregar la serie temporal original y suavizada al gráfico
        fig.add_trace(go.Scatter(
            x=serie_temporal.index, y=serie_temporal,
            mode='lines', name=f'Serie Temporal - {nombre_limpio}', line=dict(color='green'), visible=visible
        ))

        fig.add_trace(go.Scatter(
            x=serie_suavizada.index, y=serie_suavizada,
            mode='lines', name=f'Suavizado Exponencial - {nombre_limpio}', line=dict(color='red'), visible=visible
        ))

    # Definir botones desplegables para seleccionar las series temporales a mostrar
    buttons = []
    for columna in TICS_General.columns:
        nombre_limpio = columna.split('_')[1].capitalize()  # Obtener el nombre sin 'usuario_'
        buttons.append(dict(
            label=nombre_limpio,
            method='update',
            args=[{'visible': [True if col.name.startswith(f'Serie Temporal - {nombre_limpio}') or col.name.startswith(f'Suavizado Exponencial - {nombre_limpio}') else False for col in fig.data]},
                  {'title': f"<b>Suavizado Exponencial de Usuarios con {nombre_limpio}</b>"}]
        ))

    # Establecer el diseño del gráfico
    fig.update_layout(
        title="<b>Suavizado Exponencial de Series Temporales</b>",
        xaxis_title="Año",
        yaxis_title="Valor",
        xaxis=dict(showgrid=True, zeroline=False),
        yaxis=dict(showgrid=True, zeroline=False),
        updatemenus=[dict(
            type="dropdown",
            direction="down",
            buttons=buttons,
            showactive=True,
        )]
    )

    # Mostrar el gráfico en Streamlit
    st.plotly_chart(fig, use_container_width=True)


def autocorrelacion(dataframes):
    # Suponiendo que 'TICS_General' es tu DataFrame
    TICS_General = dataframes['TICS_General']
    
    # Diccionario para mapear nombres de columnas a nombres amigables
    friendly_names = {
        'usuarios_computadora': 'Computadora',
        'usuarios_celular': 'Celular',
        'usuarios_internet': 'Internet'
    }

    # Streamlit app
    st.title('Autocorrelacion y Autocorrelacion Parcial - Uso de TICS')

    # Crear una lista de categorías (columnas) para los botones
    categories = TICS_General.columns

    # Crear las figuras de ACF y PACF para cada categoría
    acf_figs = {}
    pacf_figs = {}

    for columna in categories:
        serie_temporal = TICS_General[columna]
        
        # Obtener el tamaño de la muestra
        n = len(serie_temporal)
        
        # Calcular el intervalo de confianza (asumiendo una distribución normal)
        intervalo_confianza = 1.96 / np.sqrt(n)  # Para un nivel de confianza del 95%

        # Calcular la autocorrelación y la autocorrelación parcial
        acf_vals = acf(serie_temporal, nlags=13)
        pacf_vals = pacf(serie_temporal, nlags=7)

        # Graficar la autocorrelación
        fig_acf = go.Figure()
        fig_acf.add_trace(go.Bar(x=list(range(len(acf_vals))), y=acf_vals, name='ACF', marker_color='green'))
        fig_acf.add_trace(go.Scatter(
            x=list(range(len(acf_vals))),
            y=acf_vals,
            error_y=dict(
                type='data',
                array=[intervalo_confianza] * len(acf_vals),
                visible=True,
                color='rgba(0, 128, 0, 0.5)',
            ),
            mode='lines+markers',
            name='Intervalo de Confianza'
        ))
        fig_acf.update_layout(title=f'<b>Autocorrelación de Usuarios con {friendly_names[columna]}</b>', xaxis_title='Lags', yaxis_title='Autocorrelación')

        # Graficar la autocorrelación parcial
        fig_pacf = go.Figure()
        fig_pacf.add_trace(go.Bar(x=list(range(len(pacf_vals))), y=pacf_vals, name='PACF', marker_color='green'))
        fig_pacf.add_trace(go.Scatter(
            x=list(range(len(pacf_vals))),
            y=pacf_vals,
            error_y=dict(
                type='data',
                array=[intervalo_confianza] * len(pacf_vals),
                visible=True,
                color='rgba(0, 128, 0, 0.5)',
            ),
            mode='lines+markers',
            name='Intervalo de Confianza'
        ))
        fig_pacf.update_layout(title=f'<b>Autocorrelación Parcial de Usuarios con {friendly_names[columna]}</b>', xaxis_title='Lags', yaxis_title='Autocorrelación Parcial')

        acf_figs[columna] = fig_acf
        pacf_figs[columna] = fig_pacf

    # Crear un botón para cada categoría para controlar la visibilidad
    buttons_acf = []
    buttons_pacf = []

    for i, category in enumerate(categories):
        visibility_acf = [False] * len(categories)
        visibility_pacf = [False] * len(categories)
        visibility_acf[i] = True
        visibility_pacf[i] = True
        
        buttons_acf.append(dict(
            label=friendly_names[category],
            method='update',
            args=[{'visible': visibility_acf},
                {'title': f'<b>Autocorrelación de Usuarios con {friendly_names[category]}</b>'}]
        ))
        
        buttons_pacf.append(dict(
            label=friendly_names[category],
            method='update',
            args=[{'visible': visibility_pacf},
                {'title': f'<b>Autocorrelación Parcial de Usuarios con {friendly_names[category]}</b>'}]
        ))

    # Crear figura principal para ACF con botones
    fig_acf = go.Figure()
    for col in categories:
        fig_acf.add_traces(acf_figs[col].data)
    fig_acf.update_layout(
        updatemenus=[dict(
            type="dropdown",
            direction="down",
            buttons=buttons_acf,
            showactive=True,
        )],
        title=f'<b>Autocorrelación de Usuarios con {friendly_names["usuarios_computadora"]}</b>',
        xaxis_title='Lags',
        yaxis_title='Autocorrelación',
        barmode='group',
    )
    fig_acf.update_traces(visible=False)
    fig_acf.data[0].visible = True  # Mostrar solo la primera traza por defecto

    # Crear figura principal para PACF con botones
    fig_pacf = go.Figure()
    for col in categories:
        fig_pacf.add_traces(pacf_figs[col].data)
    fig_pacf.update_layout(
        updatemenus=[dict(
            type="dropdown",
            direction="down",
            buttons=buttons_pacf,
            showactive=True,
        )],
        title=f'<b>Autocorrelación Parcial de Usuarios con {friendly_names["usuarios_computadora"]}</b>',
        xaxis_title='Lags',
        yaxis_title='Autocorrelación Parcial',
        barmode='group',
    )
    fig_pacf.update_traces(visible=False)
    fig_pacf.data[0].visible = True  # Mostrar solo la primera traza por defecto

    # Dividir la página en dos columnas
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(fig_acf)

    with col2:
        st.plotly_chart(fig_pacf)

    
def graficas_segun(dataframes, dispositivo):

    st.title("Graficas de Barras - Filtros Uso de TIC")
    
    # Crear una función para generar las figuras
    def create_figure(dataframes, dispositivo):
        # Filtrar los dataframes para excluir los específicos
        filtered_dataframes = {k: v for k, v in dataframes.items() if k not in ['TICS_Computadora_Entidad', 'TICS_Internet_Entidad', 'TICS_Celular_Entidad']}
        
        # Extraer los dataframes específicos del dispositivo
        dispositivo_dfs = {k: v for k, v in filtered_dataframes.items() if dispositivo in k}

        # Crear una figura vacía
        fig = go.Figure()

        # Lista de categorías (temáticas)
        categories = list(set([k.split('_')[2] for k in dispositivo_dfs.keys()]))

        # Paleta de colores
        palette = sns.color_palette("Greens", 10)
        colors = [rgb2hex(color) for color in palette]

        # Añadir los trazos para cada categoría
        traces_by_category = {category: [] for category in categories}
        color_idx = 0
        for name, df in dispositivo_dfs.items():
            category = name.split('_')[2]
            for col in df.columns:
                trace = go.Bar(
                    x=df.index,
                    y=df[col],
                    name=f'{category} - {col}',
                    visible=(category == categories[0]),  # Hacer visible solo la primera categoría inicialmente
                    marker_color=colors[color_idx % len(colors)]  # Asignar color de la paleta
                )
                fig.add_trace(trace)
                traces_by_category[category].append(trace)
                color_idx += 1

        # Crear un botón para cada categoría para controlar la visibilidad
        buttons = []
        for category in categories:
            visibility = [False] * len(fig.data)
            for trace in traces_by_category[category]:
                visibility[fig.data.index(trace)] = True
            
            buttons.append(dict(
                label=category,
                method='update',
                args=[{'visible': visibility},
                    {'title': f'<b>Usuarios con {dispositivo} por {category}</b>'}]  # Hacer el título en negrita
            ))

        # Añadir los botones a la figura
        fig.update_layout(
            updatemenus=[dict(
                type="dropdown",
                direction="down",
                buttons=buttons,
                showactive=True,
            )]
        )

        # Configurar diseño de la gráfica
        fig.update_layout(
            title=f'<b>Usuarios con {dispositivo} por {categories[0]}</b>',  # Hacer el título en negrita
            xaxis_title='Año',
            yaxis_title='Cantidad de usuarios',
            barmode='group'
        )

        return fig

    # Generar las figuras para cada dispositivo
    fig_celular = create_figure(dataframes, 'Celular')
    fig_computadora = create_figure(dataframes, 'Computadora')
    fig_internet = create_figure(dataframes, 'Internet')

    # Crear una función para mostrar las gráficas en Streamlit
    def show_plotly_figure(figure):
        st.plotly_chart(figure, use_container_width=True)

    # Mostrar las figuras en Streamlit
    show_plotly_figure(fig_celular)
    show_plotly_figure(fig_computadora)
    show_plotly_figure(fig_internet)

def arima(dataframes):
    # Título sin negrita
    st.title("Predicciones de ARIMA - Uso de TICS")

    # Seleccionar el DataFrame 'TICS_General'
    TICS_General = dataframes['TICS_General']

    # Convertir el índice a tipo datetime con el formato especificado
    TICS_General.index = pd.to_datetime(TICS_General.index, format='%Y')

    # Inferir la frecuencia si no está especificada en el índice
    if not TICS_General.index.freq:
        TICS_General.index.freq = pd.infer_freq(TICS_General.index)

    # Renombrar las columnas reemplazando '_' con ' con ' y capitalizando
    TICS_General.columns = [col.replace('_', ' con ').capitalize() for col in TICS_General.columns]

    # Rango de años para el slider
    anio_inicio = TICS_General.index[-1].year + 1
    anio_final = 2030

    # Crear figuras para cada serie temporal y almacenarlas en un diccionario
    figuras = {}
    predicciones_dict = {}
    fechas_predicciones_dict = {}

    for columna in TICS_General.columns:
        # Obtener la serie temporal de la columna actual
        serie_temporal = TICS_General[columna]

        # Aplicar el modelo ARIMA
        modelo = ARIMA(serie_temporal, order=(1, 1, 1))
        modelo_ajustado = modelo.fit()

        # Predecir los valores futuros hasta el año 2030
        total_steps = anio_final - anio_inicio + 1
        predicciones = modelo_ajustado.forecast(steps=total_steps)

        # Crear un rango de fechas para las predicciones
        fechas_predicciones = pd.date_range(start=f'{anio_inicio}', periods=total_steps, freq='YE')

        # Almacenar las predicciones y las fechas de predicción
        predicciones_dict[columna] = predicciones
        fechas_predicciones_dict[columna] = fechas_predicciones

        # Crear la figura para la serie temporal actual
        fig = go.Figure()

        # Agregar traza para la serie temporal
        fig.add_trace(go.Scatter(x=serie_temporal.index, y=serie_temporal, mode='lines', name=f'Serie Temporal - {columna}', line=dict(color='green')))

        # Agregar traza para las predicciones, inicialmente visibles
        fig.add_trace(go.Scatter(
            x=fechas_predicciones,
            y=predicciones,
            mode='lines',
            name=f'Predicciones - {columna}',
            line=dict(color='orange'),
            visible=True
        ))

        # Crear el slider
        slider_steps = []
        for anio in range(anio_inicio, anio_final + 1):
            max_index = anio - anio_inicio + 1
            step = dict(
                method="update",
                args=[{"x": [[*serie_temporal.index, *fechas_predicciones[:max_index]]],
                       "y": [[*serie_temporal, *predicciones[:max_index]]],
                       "visible": [True, True]},
                      {"title": f"<b>Predicciones hasta el año {anio}</b>"}]
            )
            slider_steps.append(step)

        sliders = [dict(
            active=0,
            currentvalue={"prefix": "Visualizar hasta el año: "},
            pad={"t": 50},
            steps=slider_steps
        )]

        fig.update_layout(
            sliders=sliders,
            title=f"Predicciones ARIMA para {columna.title()}",
            xaxis_title="Año",
            yaxis_title="Valor"
        )

        # Almacenar la figura en el diccionario
        figuras[columna] = fig

    # Crear un botón desplegable para seleccionar la serie temporal
    tic_seleccionada = st.selectbox('Selecciona la TIC a visualizar:', TICS_General.columns)

    # Mostrar la figura correspondiente a la serie temporal seleccionada, ampliada a toda la página
    st.plotly_chart(figuras[tic_seleccionada], use_container_width=True)

def arima_residuos(dataframes):
    st.title("Residuos de ARIMA - Uso de TICS")

    # Seleccionar el DataFrame 'TICS_General'
    TICS_General = dataframes['TICS_General']

    # Convertir el índice a tipo datetime con el formato especificado
    TICS_General.index = pd.to_datetime(TICS_General.index, format='%Y')

    # Inferir la frecuencia si no está especificada en el índice
    if not TICS_General.index.freq:
        TICS_General.index.freq = pd.infer_freq(TICS_General.index)

    # Renombrar las columnas reemplazando '_' con ' con ' y capitalizando
    TICS_General.columns = [col.replace('_', ' con ').capitalize() for col in TICS_General.columns]

    # Rango de años para el slider
    anio_inicio = TICS_General.index[-1].year + 1
    anio_final = 2030

    TICS_General.index.freq = pd.infer_freq(TICS_General.index)

    # Crear figuras para cada serie temporal
    figuras = {}
    predicciones_dict = {}
    fechas_predicciones_dict = {}

    for columna in TICS_General.columns:
        # Obtener la serie temporal de la columna actual
        serie_temporal = TICS_General[columna]

        # Aplicar el modelo ARIMA
        modelo = ARIMA(serie_temporal, order=(1, 1, 1))
        modelo_ajustado = modelo.fit()

        # Predecir los valores futuros hasta el año 2030
        total_steps = anio_final - anio_inicio + 1
        predicciones = modelo_ajustado.forecast(steps=total_steps)

        # Crear un rango de fechas para las predicciones
        fechas_predicciones = pd.date_range(start=f'{anio_inicio}', periods=total_steps, freq='YE')

        # Almacenar las predicciones y las fechas de predicción
        predicciones_dict[columna] = predicciones
        fechas_predicciones_dict[columna] = fechas_predicciones

        # Análisis de residuos para las predicciones
        residuos_predicciones = modelo_ajustado.resid

        # Crear la figura para la serie temporal actual
        fig = go.Figure()

        # Agregar traza para la serie temporal
        fig.add_trace(go.Scatter(x=serie_temporal.index, y=serie_temporal, mode='lines', name=f'Serie Temporal - {columna}', line=dict(color='green')))

        # Agregar traza para las predicciones, inicialmente ocultas
        fig.add_trace(go.Scatter(
            x=fechas_predicciones,
            y=residuos_predicciones,
            mode='markers',
            name=f'Residuos de Predicciones - {columna}',
            marker=dict(color='red')
        ))

        # Crear el slider
        slider_steps = []
        for anio in range(anio_inicio, anio_final + 1):
            max_index = anio - anio_inicio + 1
            step = dict(
                method="update",
                args=[{"x": [[*serie_temporal.index, *fechas_predicciones[:max_index]]],
                    "y": [[*serie_temporal, *residuos_predicciones[:max_index]]],
                    "visible": [True, True]},
                    {"title": f"<b>Predicciones hasta el año {anio}</b>"}]
            )
            slider_steps.append(step)

        sliders = [dict(
            active=0,
            currentvalue={"prefix": "Visualizar hasta el año: "},
            pad={"t": 50},
            steps=slider_steps
        )]

        fig.update_layout(
            sliders=sliders,
            title=f"Residuos de Predicciones ARIMA para {columna.title()}",
            xaxis_title="Año",
            yaxis_title="Residuo"
        )

        # Almacenar la figura
        figuras[columna] = fig

    # Selectbox para escoger la serie temporal
    selected_series = st.selectbox("Selecciona una TIC para visualizae", list(TICS_General.columns))

    # Mostrar la gráfica correspondiente a la serie seleccionada, usando todo el ancho de la página
    st.plotly_chart(figuras[selected_series], use_container_width=True)

def sarima(dataframes):
    st.title("Predicciones de SARIMA - Uso de TICS")

    # Seleccionar el DataFrame 'TICS_General'
    TICS_General = dataframes['TICS_General']

    # Convertir el índice a tipo datetime con el formato especificado
    TICS_General.index = pd.to_datetime(TICS_General.index, format='%Y')

    # Inferir la frecuencia si no está especificada en el índice
    if not TICS_General.index.freq:
        TICS_General.index.freq = pd.infer_freq(TICS_General.index)

    # Renombrar las columnas reemplazando '_' con ' con ' y capitalizando
    TICS_General.columns = [col.replace('_', ' con ').capitalize() for col in TICS_General.columns]

    # Rango de años para el slider
    anio_inicio = TICS_General.index[-1].year + 1
    anio_final = 2030

    # Crear figuras para cada serie temporal
    figuras = {}
    predicciones_dict = {}
    fechas_predicciones_dict = {}

    for columna in TICS_General.columns:
        # Obtener la serie temporal de la columna actual
        serie_temporal = TICS_General[columna]

        # Aplicar el modelo SARIMAX
        modelo = SARIMAX(serie_temporal, order=(1, 1, 1), seasonal_order=(1, 1, 1, 2))
        modelo_ajustado = modelo.fit(disp=False)

        # Predecir los valores futuros hasta el año 2030
        total_steps = anio_final - anio_inicio + 1
        predicciones = modelo_ajustado.forecast(steps=total_steps)

        # Crear un rango de fechas para las predicciones
        fechas_predicciones = pd.date_range(start=f'{anio_inicio}', periods=total_steps, freq='YE')

        # Almacenar las predicciones y las fechas de predicción
        predicciones_dict[columna] = predicciones
        fechas_predicciones_dict[columna] = fechas_predicciones

        # Crear la figura para la serie temporal actual
        fig = go.Figure()

        # Agregar traza para la serie temporal
        fig.add_trace(go.Scatter(x=serie_temporal.index, y=serie_temporal, mode='lines', name=f'Serie Temporal - {columna}',line=dict(color='green')))

        # Agregar traza para las predicciones
        fig.add_trace(go.Scatter(
            x=fechas_predicciones,
            y=predicciones,
            mode='lines',
            name=f'Predicciones - {columna}',
            line=dict(color='orange')
        ))

        # Crear el slider
        slider_steps = []
        for anio in range(anio_inicio, anio_final + 1):
            max_index = anio - anio_inicio + 1
            step = dict(
                method="update",
                args=[{"x": [[*serie_temporal.index, *fechas_predicciones[:max_index]]],
                       "y": [[*serie_temporal, *predicciones[:max_index]]],
                       "visible": [True, True]},
                      {"title": f"<b>Predicciones hasta el año {anio}</b>"}]
            )
            slider_steps.append(step)

        sliders = [dict(
            active=0,
            currentvalue={"prefix": "Visualizar hasta el año: "},
            pad={"t": 50},
            steps=slider_steps
        )]

        fig.update_layout(
            sliders=sliders,
            title=f"Predicciones SARIMA para {columna.title()}",
            xaxis_title="Año",
            yaxis_title="Valor"
        )

        # Almacenar la figura
        figuras[columna] = fig

    # Selectbox para escoger la serie temporal
    selected_series = st.selectbox("Selecciona una TIC para visualizar", list(TICS_General.columns), key='select_series')

    # Mostrar la gráfica correspondiente a la serie seleccionada, usando todo el ancho de la página
    st.plotly_chart(figuras[selected_series], use_container_width=True)

def sarima_residuos(dataframes):
    st.title("Residuos de SARIMA - Uso de TICS")

    # Seleccionar el DataFrame 'TICS_General'
    TICS_General = dataframes['TICS_General']

    # Convertir el índice a tipo datetime con el formato especificado
    TICS_General.index = pd.to_datetime(TICS_General.index, format='%Y')

    # Inferir la frecuencia si no está especificada en el índice
    if not TICS_General.index.freq:
        TICS_General.index.freq = pd.infer_freq(TICS_General.index)

    # Renombrar las columnas reemplazando '_' con ' con ' y capitalizando
    TICS_General.columns = [col.replace('_', ' con ').capitalize() for col in TICS_General.columns]

    anio_inicio = TICS_General.index[-1].year + 1
    anio_final = 2030

    # Crear figuras para cada serie temporal
    figuras = {}
    predicciones_dict = {}
    fechas_predicciones_dict = {}

    for columna in TICS_General.columns:
        # Obtener la serie temporal de la columna actual
        serie_temporal = TICS_General[columna]

        # Aplicar el modelo SARIMAX
        modelo = SARIMAX(serie_temporal, order=(1, 1, 1), seasonal_order=(1, 1, 1, 2))
        modelo_ajustado = modelo.fit(disp=False)

        # Predecir los valores futuros hasta el año 2030
        total_steps = anio_final - anio_inicio + 1
        predicciones = modelo_ajustado.forecast(steps=total_steps)

        # Crear un rango de fechas para las predicciones
        fechas_predicciones = pd.date_range(start=f'{anio_inicio}', periods=total_steps, freq='YE')

        # Almacenar las predicciones y las fechas de predicción
        predicciones_dict[columna] = predicciones
        fechas_predicciones_dict[columna] = fechas_predicciones

        # Análisis de residuos para las predicciones
        residuos_predicciones = modelo_ajustado.resid

        # Crear la figura para la serie temporal actual
        fig = go.Figure()

        # Agregar traza para la serie temporal
        fig.add_trace(go.Scatter(x=serie_temporal.index, y=serie_temporal, mode='lines', name=f'Serie Temporal - {columna}', line=dict(color='green')))

        # Agregar traza para las predicciones, inicialmente ocultas
        fig.add_trace(go.Scatter(
            x=fechas_predicciones,
            y=residuos_predicciones,
            mode='markers',
            name=f'Residuos de Predicciones - {columna}',
            marker=dict(color='red')
        ))

        # Crear el slider
        slider_steps = []
        for anio in range(anio_inicio, anio_final + 1):
            max_index = anio - anio_inicio + 1
            step = dict(
                method="update",
                args=[{"x": [[*serie_temporal.index, *fechas_predicciones[:max_index]]],
                    "y": [[*serie_temporal, *residuos_predicciones[:max_index]]],
                    "visible": [True, True]},
                    {"title": f"<b>Predicciones hasta el año {anio}</b>"}]
            )
            slider_steps.append(step)

        sliders = [dict(
            active=0,
            currentvalue={"prefix": "Visualizar hasta el año: "},
            pad={"t": 50},
            steps=slider_steps
        )]

        fig.update_layout(
            sliders=sliders,
            title=f"Residuos de Predicciones ARIMA para {columna.title()}",
            xaxis_title="Año",
            yaxis_title="Residuo"
        )

        # Almacenar la figura
        figuras[columna] = fig

    # Selectbox para escoger la serie temporal a visualizar
    opcion = st.selectbox('Seleccione la TIC a visualizar:', list(TICS_General.columns))

    # Mostrar la figura correspondiente a la opción seleccionada
    st.plotly_chart(figuras[opcion], use_container_width=True)

def reg_lineal(dataframes):
    # Título sin negrita
    st.title("Predicciones de Regresion Lineal - Uso de TICS")

    # Seleccionar el DataFrame 'TICS_General'
    TICS_General = dataframes['TICS_General']

    # Convertir el índice a tipo datetime con el formato especificado
    TICS_General.index = pd.to_datetime(TICS_General.index, format='%Y')

    # Inferir la frecuencia si no está especificada en el índice
    if not TICS_General.index.freq:
        TICS_General.index.freq = pd.infer_freq(TICS_General.index)

    # Renombrar las columnas reemplazando '_' con ' con ' y capitalizando
    TICS_General.columns = [col.replace('_', ' con ').capitalize() for col in TICS_General.columns]

    # Rango de años para el slider
    anio_inicio = TICS_General.index[-1].year + 1
    anio_final = 2030

    # Inicializar diccionarios para almacenar figuras, predicciones y fechas
    figuras_dict = {}
    predicciones_dict = {}
    fechas_predicciones_dict = {}

    for columna in TICS_General.columns:
        # Obtener la serie temporal de la columna actual
        serie_temporal = TICS_General[columna]

        # Convertir serie_temporal a una Serie de Pandas si no lo es actualmente
        serie_temporal = pd.Series(serie_temporal)

        # Aplicar regresión lineal
        X = np.array(range(len(serie_temporal))).reshape(-1, 1)
        y = np.array(serie_temporal)
        regresion_lineal = LinearRegression().fit(X, y)

        # Predecir los valores futuros hasta el año 2030 con regresión lineal
        total_steps = anio_final - anio_inicio + 1
        predicciones_regresion = regresion_lineal.predict(np.array(range(len(serie_temporal), len(serie_temporal) + total_steps)).reshape(-1, 1))

        # Almacenar las predicciones de regresión y las fechas de predicción
        predicciones_dict[columna] = predicciones_regresion
        fechas_predicciones = pd.date_range(start=f'{anio_inicio}', periods=total_steps, freq='YE')
        fechas_predicciones_dict[columna] = fechas_predicciones

        # Crear la figura para la serie temporal actual
        fig = go.Figure()

        # Agregar traza para la serie temporal
        fig.add_trace(go.Scatter(x=serie_temporal.index, y=serie_temporal, mode='lines', name=f'Serie Temporal - {columna}', line=dict(color='green')))

        # Agregar traza para las predicciones de regresión
        fig.add_trace(go.Scatter(
            x=fechas_predicciones,
            y=predicciones_regresion,
            mode='lines',
            name=f'Predicciones de Regresión - {columna}',
            line=dict(color='orange')
        ))

        # Crear el slider
        slider_steps = []
        for anio in range(anio_inicio, anio_final + 1):
            max_index = anio - anio_inicio + 1
            step = dict(
                method="update",
                args=[{"x": [[*serie_temporal.index, *fechas_predicciones[:max_index]]],
                       "y": [[*serie_temporal, *predicciones_regresion[:max_index]]],
                       "visible": [True, True]},
                      {"title": f"<b>Predicciones hasta el año {anio}</b>"}]
            )
            slider_steps.append(step)

        sliders = [dict(
            active=0,
            currentvalue={"prefix": "Visualizar hasta el año: "},
            pad={"t": 50},
            steps=slider_steps
        )]

        fig.update_layout(
            sliders=sliders,
            title=f"Predicciones de Regresión Lineal para {columna.title()}",
            xaxis_title="Año",
            yaxis_title="Valor"
        )

        # Almacenar la figura
        figuras_dict[columna] = fig

    # Selector para elegir la serie temporal
    columna_seleccionada = st.selectbox('Selecciona una TIC para visualizar:', list(TICS_General.columns))

    # Mostrar la figura correspondiente, ocupando todo el ancho de la página
    st.plotly_chart(figuras_dict[columna_seleccionada], use_container_width=True)