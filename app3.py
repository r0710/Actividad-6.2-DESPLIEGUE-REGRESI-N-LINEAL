#####################
# Importamos librerias
import matplotlib.pyplot as plt
import streamlit as st 
import plotly.express as px
import pandas as pd
import io
import numpy as np
from sklearn.linear_model import LinearRegression
import seaborn as sns

#####################
# Definimos la instancia
#################
# Creamos la función de carga de datos
@st.cache_resource
def load_data():
    df = pd.read_csv('Socio_NNCOMPLETO.csv')
    df['Usuario'] = df['Usuario'].str.upper()
    usuarios_validos = ['VALENTIN', 'YAEL DAVID', 'YEREMI YAZMIN ', 'RAMIRO ISAI']
    df = df[df['Usuario'].isin(usuarios_validos)]
    columnas = ['Usuario', 'Administrador','botón correcto', 'tiempo de interacción', 'mini juego',
                'número de interacción', 'color presionado', 'dificultad', 'fecha',
                'Juego', 'tiempo de sesión']
    df = df[columnas]
    return df, columnas

st.set_page_config(page_title= "Wuupi Duupi", layout= "wide")

@st.cache_resource
def load_numeric_data():
    df_numeric = pd.read_csv('DATANN.csv')
    return df_numeric

st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://i.pinimg.com/originals/ee/58/97/ee58978ecfe73a9ad982290b0ba19a84.jpg");
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.image("cropped-menta (1).png")

##################
# Cargo los datos obtenidos de la función 'load_data'
df, Lista = load_data()
k_gr = ['bar', 'area', 'pie']

#####################
# CREACIÓN DEL DASHBOARD

st.sidebar.title('Wuupi')  # Puedes cambiar el título si quieres
View = st.sidebar.selectbox(label='Tipo de Análisis', options=['Extracción de características',
                                            'Regresión Lineal','Regresión No Lineal', 'Regresión Logísitca', 'ANOVA'])


# CONTENIDO DE LA VISTA 1
if View == 'Extracción de características':
    Variable_Cat = st.sidebar.selectbox(label = 'Variable', options = Lista[1:])   # Quitamos 'Usuario' de la selección
    tipo_grafica = st.sidebar.selectbox(label = 'Tipo de grafica', options = k_gr)   

    st.title('Extracción de características')
    

    # Creamos los 4 contenedores de comparación, uno por usuario
    usuarios = df['Usuario'].unique()
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    # Usuario: VALENTIN
    with col1:
        df_valentin = df[df['Usuario'] == 'VALENTIN']
        tabla_valentin = df_valentin[Variable_Cat].value_counts().reset_index()
        tabla_valentin.columns = ['categorias', 'frecuencia']
        st.subheader('Valentin')
        if tipo_grafica == 'bar':
            fig1 = px.bar(data_frame=tabla_valentin, x='categorias', y='frecuencia', title=f'Frecuencia de {Variable_Cat}',
                          color='frecuencia', color_continuous_scale='Blues')
        elif tipo_grafica == 'area':
            fig1 = px.area(data_frame=tabla_valentin, x='categorias', y='frecuencia', title=f'Frecuencia de {Variable_Cat}',
                           color='frecuencia')
        elif tipo_grafica == 'pie':
            fig1 = px.pie(data_frame=tabla_valentin, names='categorias', values='frecuencia', title=f'Frecuencia de {Variable_Cat}',
                          color_discrete_sequence=px.colors.sequential.Blues)
        fig1.update_layout(height=300)
        st.plotly_chart(fig1, use_container_width=True)

    # Usuario: YAEL DAVID
    with col2:
        df_yael = df[df['Usuario'] == 'YAEL DAVID']
        tabla_yael = df_yael[Variable_Cat].value_counts().reset_index()
        tabla_yael.columns = ['categorias', 'frecuencia']
        st.subheader('Yael David')
        if tipo_grafica == 'bar':
            fig2 = px.bar(data_frame=tabla_yael, x='categorias', y='frecuencia', title=f'Frecuencia de {Variable_Cat}',
                          color='frecuencia', color_continuous_scale='Blues')
        elif tipo_grafica == 'area':
            fig2 = px.area(data_frame=tabla_yael, x='categorias', y='frecuencia', title=f'Frecuencia de {Variable_Cat}',
                           color='frecuencia')
        elif tipo_grafica == 'pie':
            fig2 = px.pie(data_frame=tabla_yael, names='categorias', values='frecuencia', title=f'Frecuencia de {Variable_Cat}',
                          color_discrete_sequence=px.colors.sequential.Blues)
        fig2.update_layout(height=300)
        st.plotly_chart(fig2, use_container_width=True)

    # Usuario: YEREMI YAZMIN
    with col3:
        df_yeremi = df[df['Usuario'] == 'YEREMI YAZMIN ']
        tabla_yeremi = df_yeremi[Variable_Cat].value_counts().reset_index()
        tabla_yeremi.columns = ['categorias', 'frecuencia']
        st.subheader('Yeremi Yazmin')
        if tipo_grafica == 'bar':
            fig3 = px.bar(data_frame=tabla_yeremi, x='categorias', y='frecuencia', title=f'Frecuencia de {Variable_Cat}',
                          color='frecuencia', color_continuous_scale='Blues')
        elif tipo_grafica == 'area':
            fig3 = px.area(data_frame=tabla_yeremi, x='categorias', y='frecuencia', title=f'Frecuencia de {Variable_Cat}',
                           color='frecuencia')
        elif tipo_grafica == 'pie':
            fig3 = px.pie(data_frame=tabla_yeremi, names='categorias', values='frecuencia', title=f'Frecuencia de {Variable_Cat}',
                          color_discrete_sequence=px.colors.sequential.Blues)
        fig3.update_layout(height=300)
        st.plotly_chart(fig3, use_container_width=True)

    # Usuario: RAMIRO ISAI
    with col4:
        df_ramiro = df[df['Usuario'] == 'RAMIRO ISAI']
        tabla_ramiro = df_ramiro[Variable_Cat].value_counts().reset_index()
        tabla_ramiro.columns = ['categorias', 'frecuencia']
        st.subheader('Ramiro Isai')
        if tipo_grafica == 'bar':
            fig4 = px.bar(data_frame=tabla_ramiro, x='categorias', y='frecuencia', title=f'Frecuencia de {Variable_Cat}',
                          color='frecuencia', color_continuous_scale='Blues')
        elif tipo_grafica == 'area':
            fig4 = px.area(data_frame=tabla_ramiro, x='categorias', y='frecuencia', title=f'Frecuencia de {Variable_Cat}',
                           color='frecuencia')
        elif tipo_grafica == 'pie':
            fig4 = px.pie(data_frame=tabla_ramiro, names='categorias', values='frecuencia', title=f'Frecuencia de {Variable_Cat}',
                          color_discrete_sequence=px.colors.sequential.Blues)
        fig4.update_layout(height=300)
        st.plotly_chart(fig4, use_container_width=True)

import numpy as np
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

if View == 'Regresión Lineal':
    df_numeric = load_numeric_data()

    
    # Filtramos solo columnas numéricas (evita errores con fechas, strings, etc.)
    numeric_df = df_numeric.select_dtypes(include=['float64', 'int64'])  
    Lista_num = numeric_df.columns.tolist()

    # Select boxes para elegir variables
    variable_y = st.sidebar.selectbox('Variable objetivo (y)', options=Lista_num)
    variable_x = st.sidebar.selectbox('Variable independiente (x)', options=Lista_num)



    tab1, tab2, tab3 = st.tabs(['Correlaciones', 'Análisis de modelos', 'Observaciones'])

    with tab1:
        st.title('Análisis de Regresiones')

        # Mapa de calor al principio
        st.subheader('Mapa de calor de correlaciones')
        fig_heatmap, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='Blues', ax=ax)
        st.pyplot(fig_heatmap)

    with tab2:
        # Contenedores de las dos regresiones
        Contenedor_A, Contenedor_B = st.columns(2)

        # ------------------------- REGRESIÓN LINEAL SIMPLE -------------------------
        with Contenedor_A:
            st.subheader('Regresión Lineal Simple')

            model = LinearRegression()
            model.fit(X=numeric_df[[variable_x]], y=numeric_df[variable_y])

            y_pred = model.predict(X=numeric_df[[variable_x]])

            coef_deter_simple = model.score(X=numeric_df[[variable_x]], y=numeric_df[variable_y])
            coef_correl_simple = np.sqrt(coef_deter_simple)

            st.write(f"Coeficiente de correlación lineal simple: {coef_correl_simple:.2f}")

            # Gráfico de dispersión
            fig_simple = px.scatter(x=y_pred, y=numeric_df[variable_y], 
                                    labels={'x': 'Predicciones', 'y': variable_y},
                                    title='Modelo Lineal Simple')
            st.plotly_chart(fig_simple)

    # ------------------------- REGRESIÓN LINEAL MÚLTIPLE -------------------------
        with Contenedor_B:
            st.subheader("Regresión Lineal Múltiple")

            variables_x = st.sidebar.multiselect(
                label="Variables independientes del modelo múltiple (X)", 
                options=[var for var in Lista_num if var != variable_y]
            )

            if variables_x:
                model_M = LinearRegression()
                model_M.fit(X=numeric_df[variables_x], y=numeric_df[variable_y])

                y_pred_M = model_M.predict(X=numeric_df[variables_x])
                coef_Deter_multiple = model_M.score(X=numeric_df[variables_x], y=numeric_df[variable_y])
                coef_Correl_multiple = np.sqrt(coef_Deter_multiple)

                st.write(f"Coeficiente de correlación múltiple: {coef_Correl_multiple:.2f}")

                fig_multiple = px.scatter(x=y_pred_M, y=numeric_df[variable_y],
                                        labels={'x': 'Predicciones', 'y': variable_y},
                                        title='Modelo Lineal Múltiple')
                st.plotly_chart(fig_multiple)
            else:
                st.warning("Selecciona al menos una variable independiente para el modelo múltiple.")

    import streamlit as st
    from fpdf import FPDF
    import io

    # 1. Mantener las observaciones acumuladas en el estado
    if 'observaciones_acumuladas' not in st.session_state:
        st.session_state['observaciones_acumuladas'] = ""

    # 2. Función para crear el PDF
    def crear_pdf(texto):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, texto)
        
        # Obtener contenido como string y codificarlo como bytes
        pdf_output = pdf.output(dest='S').encode('latin1')
        return io.BytesIO(pdf_output)


    # Crear un área de texto donde el usuario pueda escribir observaciones
    with tab3:
        st.header('Observaciones')
        nueva_obs = st.text_area('Escribe tus observaciones aquí', '')

# 4. Botón para agregar nueva observación al historial
        if st.button('Agregar observación'):
            if nueva_obs.strip() != "":
                st.session_state['observaciones_acumuladas'] += nueva_obs + "\n\n"
                st.success("Observación agregada.")
            else:
                st.warning("No escribiste nada.")

        # 5. Botón para generar PDF y descargarlo
        if st.button("Generar PDF"):
            if st.session_state['observaciones_acumuladas'].strip() != "":
                pdf_bytes = crear_pdf(st.session_state['observaciones_acumuladas'])
                st.download_button(
                    label="Descargar PDF",
                    data=pdf_bytes,
                    file_name="observaciones_cliente.pdf",
                    mime="application/pdf"
                )
            else:
                st.warning("No hay observaciones para guardar.")