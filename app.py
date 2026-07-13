###############################################################
# AI STUDENT IMPACT ANALYSIS
# Streamlit Dashboard
#
# Autor: Gustavo Fernando Torres Leyva
# Maestría en Transformación Digital e Innovación
###############################################################

import warnings
warnings.filterwarnings("ignore")

import os
import io
# import joblib
import numpy as np
import pandas as pd

import streamlit as st

# import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

###############################################################
# CONFIGURACIÓN GENERAL
###############################################################

st.set_page_config(

    page_title="AI Student Impact Dashboard",

    page_icon="🎓",

    layout="wide",

    initial_sidebar_state="expanded"

)

###############################################################
# CSS PERSONALIZADO
###############################################################

st.markdown("""

<style>

.main{

    background-color:#F7F9FC;

}

h1{

    color:#003366;

    font-weight:bold;

}

h2{

    color:#003366;

}

.metric-card{

    background:white;

    padding:15px;

    border-radius:12px;

    box-shadow:0px 2px 8px rgba(0,0,0,.15);

}

.sidebar .sidebar-content{

    background:#002244;

}

</style>

""", unsafe_allow_html=True)

###############################################################
# TÍTULO
###############################################################

st.title("🎓 AI Student Impact Analysis")

st.markdown("""

### Exploratory Data Analysis & Machine Learning Dashboard

Este dashboard permite analizar el impacto del uso de Inteligencia Artificial Generativa sobre el rendimiento académico universitario.

""")

###############################################################
# CONFIGURACIÓN
###############################################################

RANDOM_STATE = 42

TARGET = "Post_Semester_GPA"

###############################################################
# CARGA DEL DATASET
###############################################################

@st.cache_data
def load_data():

    df = pd.read_csv(
        "data/ai_student_impact_dataset.csv"
    )

    return df


try:

    df = load_data()

except Exception:

    st.error("No fue posible cargar el dataset.")

    st.stop()

###############################################################
# SIDEBAR
###############################################################

st.sidebar.image(
    "https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png",
    width=180
)

st.sidebar.title("Navegación")

menu = st.sidebar.radio(

    "Seleccione una sección",

    [

        "🏠 Inicio",

        "📊 Dataset",

        "📈 Visualizaciones",

        "🤖 Machine Learning",

        "📋 Resultados",

        "ℹ️ Acerca"

    ]

)

###############################################################
# FILTROS
###############################################################

st.sidebar.markdown("---")

st.sidebar.header("Filtros")

gender = st.sidebar.multiselect(

    "Género",

    options=sorted(df["Gender"].unique()),

    default=sorted(df["Gender"].unique())

)

career = st.sidebar.multiselect(

    "Carrera",

    options=sorted(df["Major_Category"].unique()),

    default=sorted(df["Major_Category"].unique())

)

filtered = df[

    (df["Gender"].isin(gender)) &

    (df["Major_Category"].isin(career))

]

###############################################################
# KPIs
###############################################################

st.markdown("---")

col1,col2,col3,col4 = st.columns(4)

with col1:

    st.metric(

        "Estudiantes",

        f"{len(filtered):,}"

    )

with col2:

    st.metric(

        "Variables",

        filtered.shape[1]

    )

with col3:

    st.metric(

        "GPA Promedio",

        round(filtered[TARGET].mean(),2)

    )

with col4:

    st.metric(

        "Horas IA",

        round(filtered["Weekly_GenAI_Usage_Hours"].mean(),2)

    )

###############################################################
# INICIO
###############################################################

if menu=="🏠 Inicio":

    st.header("Descripción del Proyecto")

    st.write("""

Este proyecto analiza la relación entre el uso de herramientas de Inteligencia Artificial Generativa y el rendimiento académico de estudiantes universitarios.

Se aplican técnicas de:

- Exploratory Data Analysis (EDA)
- Ciencia de Datos
- Machine Learning Supervisado
- Visualización Interactiva

""")

    st.markdown("---")

    st.subheader("Objetivo General")

    st.info("""

Analizar el impacto del uso de herramientas de Inteligencia Artificial Generativa sobre el rendimiento académico utilizando técnicas de análisis exploratorio y aprendizaje automático.

""")

    st.subheader("Objetivos Específicos")

    st.markdown("""

- Comprender el comportamiento del dataset.

- Identificar patrones.

- Analizar correlaciones.

- Comparar distintos modelos predictivos.

- Identificar las variables más importantes.

""")

    st.markdown("---")

    st.subheader("Vista previa del dataset")

    st.dataframe(filtered.head(10), use_container_width=True)

    st.markdown("---")

    st.success(

        f"""

Dataset cargado correctamente

Registros: {filtered.shape[0]:,}

Columnas: {filtered.shape[1]}

"""

    )

###############################################################
# DATASET
###############################################################

if menu == "📊 Dataset":

    st.header("📊 Exploración del Dataset")

    st.markdown("""
    En esta sección se presenta un análisis inicial del conjunto de datos,
    evaluando su estructura, calidad y principales características.
    """)

    ###########################################################
    # Información general
    ###########################################################

    st.subheader("Información General")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Registros", f"{filtered.shape[0]:,}")

    with c2:
        st.metric("Variables", filtered.shape[1])

    with c3:
        memoria = filtered.memory_usage(deep=True).sum() / (1024**2)
        st.metric("Memoria (MB)", round(memoria,2))

    st.markdown("---")

    ###########################################################
    # Vista previa
    ###########################################################

    st.subheader("Vista previa")

    filas = st.slider(
        "Número de filas",
        5,
        50,
        10
    )

    st.dataframe(
        filtered.head(filas),
        use_container_width=True
    )

    ###########################################################
    # Tipos de datos
    ###########################################################

    st.markdown("---")

    st.subheader("Tipos de datos")

    tipos = pd.DataFrame({

        "Variable": filtered.columns,

        "Tipo": filtered.dtypes.astype(str)

    })

    st.dataframe(
        tipos,
        use_container_width=True
    )

    ###########################################################
    # Valores nulos
    ###########################################################

    st.markdown("---")

    st.subheader("Valores nulos")

    nulls = pd.DataFrame({

        "Valores Nulos": filtered.isnull().sum(),

        "Porcentaje (%)": (
            filtered.isnull().mean()*100
        ).round(2)

    })

    st.dataframe(
        nulls,
        use_container_width=True
    )

    fig = px.bar(

        nulls.reset_index(),

        x="index",

        y="Valores Nulos",

        color="Valores Nulos",

        title="Valores nulos por variable"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    ###########################################################
    # Duplicados
    ###########################################################

    st.markdown("---")

    st.subheader("Registros duplicados")

    duplicados = filtered.duplicated().sum()

    if duplicados == 0:

        st.success("No existen registros duplicados.")

    else:

        st.warning(
            f"Se encontraron {duplicados} registros duplicados."
        )

    ###########################################################
    # Estadística descriptiva
    ###########################################################

    st.markdown("---")

    st.subheader("Estadísticas descriptivas")

    st.dataframe(

        filtered.describe(),

        use_container_width=True

    )

    ###########################################################
    # Variables categóricas
    ###########################################################

    st.markdown("---")

    st.subheader("Variables categóricas")

    cat_cols = filtered.select_dtypes(
        include="object"
    ).columns.tolist()

    variable = st.selectbox(

        "Seleccione una variable",

        cat_cols

    )

    frecuencia = (

        filtered[variable]

        .value_counts()

        .reset_index()

    )

    frecuencia.columns = [

        variable,

        "Frecuencia"

    ]

    fig = px.bar(

        frecuencia,

        x=variable,

        y="Frecuencia",

        color="Frecuencia",

        text="Frecuencia",

        title=f"Distribución de {variable}"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    ###########################################################
    # Variables numéricas
    ###########################################################

    st.markdown("---")

    st.subheader("Variables numéricas")

    num_cols = filtered.select_dtypes(
        include=np.number
    ).columns.tolist()

    variable_num = st.selectbox(

        "Seleccione una variable numérica",

        num_cols

    )

    ###########################################################
    # Histograma
    ###########################################################

    fig = px.histogram(

        filtered,

        x=variable_num,

        nbins=30,

        marginal="box",

        color_discrete_sequence=["#3366CC"]

    )

    fig.update_layout(

        title=f"Distribución de {variable_num}"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    ###########################################################
    # Boxplot
    ###########################################################

    fig = px.box(

        filtered,

        y=variable_num,

        color_discrete_sequence=["#DC3912"]

    )

    fig.update_layout(

        title=f"Boxplot de {variable_num}"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    ###########################################################
    # Tabla dinámica
    ###########################################################

    st.markdown("---")

    st.subheader("Tabla dinámica")

    pivot = pd.pivot_table(

        filtered,

        values="Post_Semester_GPA",

        index="Major_Category",

        columns="Gender",

        aggfunc="mean"

    )

    st.dataframe(

        pivot,

        use_container_width=True

    )

    ###########################################################
    # Correlación
    ###########################################################

    st.markdown("---")

    st.subheader("Matriz de correlación")

    corr = filtered.corr(
        numeric_only=True
    )

    fig = px.imshow(

        corr,

        text_auto=".2f",

        color_continuous_scale="RdBu_r",

        aspect="auto"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    ###########################################################
    # Descarga CSV
    ###########################################################

    st.markdown("---")

    st.subheader("Exportar información")

    csv = filtered.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(

        label="📥 Descargar Dataset Filtrado",

        data=csv,

        file_name="dataset_filtrado.csv",

        mime="text/csv"

    )

###############################################################
# VISUALIZACIONES
###############################################################

if menu == "📈 Visualizaciones":

    st.header("📈 Visualizaciones Interactivas")

    st.markdown("""
    Esta sección permite explorar visualmente el comportamiento de las
    variables del dataset mediante gráficos interactivos.
    """)

    # =========================================================
    # VARIABLES NUMÉRICAS
    # =========================================================

    num_cols = filtered.select_dtypes(include=np.number).columns.tolist()

    # ---------------------------------------------------------
    # Scatter Plot
    # ---------------------------------------------------------

    st.subheader("Relación entre dos variables")

    col1, col2 = st.columns(2)

    with col1:
        x_axis = st.selectbox(
            "Variable X",
            num_cols,
            index=num_cols.index("Weekly_GenAI_Usage_Hours")
        )

    with col2:
        y_axis = st.selectbox(
            "Variable Y",
            num_cols,
            index=num_cols.index("Post_Semester_GPA")
        )

    fig = px.scatter(
        filtered,
        x=x_axis,
        y=y_axis,
        color="Major_Category",
        hover_data=["Gender"],
        opacity=0.75,
        trendline="ols",
        title=f"{y_axis} vs {x_axis}"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------------
    # Bubble Chart
    # ---------------------------------------------------------

    st.markdown("---")

    st.subheader("Gráfico de burbujas")

    fig = px.scatter(
        filtered,
        x="Weekly_GenAI_Usage_Hours",
        y="Post_Semester_GPA",
        size="Traditional_Study_Hours",
        color="Major_Category",
        hover_name="Gender",
        opacity=0.7,
        title="Uso de IA vs GPA"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------------
    # GPA por Carrera
    # ---------------------------------------------------------

    st.markdown("---")

    st.subheader("GPA promedio por carrera")

    promedio = (
        filtered
        .groupby("Major_Category")["Post_Semester_GPA"]
        .mean()
        .reset_index()
        .sort_values("Post_Semester_GPA", ascending=False)
    )

    fig = px.bar(
        promedio,
        x="Major_Category",
        y="Post_Semester_GPA",
        color="Post_Semester_GPA",
        text_auto=".2f",
        title="Promedio de GPA por Carrera"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------------
    # GPA por Género
    # ---------------------------------------------------------

    st.markdown("---")

    st.subheader("Distribución del GPA por género")

    fig = px.box(
        filtered,
        x="Gender",
        y="Post_Semester_GPA",
        color="Gender",
        points="outliers"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------------
    # Horas IA por Carrera
    # ---------------------------------------------------------

    st.markdown("---")

    st.subheader("Uso semanal de IA por carrera")

    uso = (
        filtered
        .groupby("Major_Category")["Weekly_GenAI_Usage_Hours"]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        uso,
        x="Major_Category",
        y="Weekly_GenAI_Usage_Hours",
        color="Weekly_GenAI_Usage_Hours",
        text_auto=".2f",
        title="Horas promedio de IA"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------------
    # Violin Plot
    # ---------------------------------------------------------

    st.markdown("---")

    st.subheader("Distribución del GPA por carrera")

    fig = px.violin(
        filtered,
        x="Major_Category",
        y="Post_Semester_GPA",
        color="Major_Category",
        box=True,
        points="all"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------------
    # Burnout
    # ---------------------------------------------------------

    st.markdown("---")

    st.subheader("Burnout y rendimiento académico")

    fig = px.box(
        filtered,
        x="Burnout_Risk_Level",
        y="Post_Semester_GPA",
        color="Burnout_Risk_Level"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------------
    # Skill Retention
    # ---------------------------------------------------------

    st.markdown("---")

    st.subheader("Retención del aprendizaje")

    fig = px.scatter(
        filtered,
        x="Skill_Retention_Score",
        y="Post_Semester_GPA",
        color="Gender",
        trendline="ols",
        opacity=0.70
    )

    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------------
    # Prompt Engineering
    # ---------------------------------------------------------

    st.markdown("---")

    st.subheader("Prompt Engineering")

    prompt = (
        filtered
        .groupby("Prompt_Engineering_Skill")
        ["Post_Semester_GPA"]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        prompt,
        x="Prompt_Engineering_Skill",
        y="Post_Semester_GPA",
        color="Post_Semester_GPA",
        text_auto=".2f"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------------
    # GPA Inicial vs Final
    # ---------------------------------------------------------

    st.markdown("---")

    st.subheader("Comparación GPA Inicial vs GPA Final")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=filtered["Pre_Semester_GPA"],
        y=filtered["Post_Semester_GPA"],
        mode="markers",
        marker=dict(
            color="royalblue",
            opacity=0.6
        )
    ))

    fig.update_layout(
        title="GPA Inicial vs GPA Final",
        xaxis_title="Pre Semester GPA",
        yaxis_title="Post Semester GPA"
    )

    st.plotly_chart(fig, use_container_width=True)
    ###########################################################
    # MATRIZ DE DISPERSIÓN
    ###########################################################

    st.markdown("---")
    st.subheader("📊 Matriz de Dispersión")

    scatter_cols = [
        "Pre_Semester_GPA",
        "Post_Semester_GPA",
        "Weekly_GenAI_Usage_Hours",
        "Traditional_Study_Hours",
        "Skill_Retention_Score"
    ]

    fig = px.scatter_matrix(
        filtered,
        dimensions=scatter_cols,
        color="Gender",
        opacity=0.6,
        title="Relaciones entre Variables Numéricas"
    )

    fig.update_traces(diagonal_visible=False)

    st.plotly_chart(fig, use_container_width=True)

    ###########################################################
    # HEATMAP INTERACTIVO
    ###########################################################

    st.markdown("---")
    st.subheader("🔥 Correlaciones")

    corr = filtered.select_dtypes(include=np.number).corr()

    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        origin="lower"
    )

    st.plotly_chart(fig, use_container_width=True)

    ###########################################################
    # TOP CORRELACIONES
    ###########################################################

    st.markdown("---")
    st.subheader("📈 Variables más relacionadas con el GPA")

    target_corr = (
        corr["Post_Semester_GPA"]
        .drop("Post_Semester_GPA")
        .sort_values(ascending=False)
        .reset_index()
    )

    target_corr.columns = [
        "Variable",
        "Correlación"
    ]

    fig = px.bar(
        target_corr,
        x="Correlación",
        y="Variable",
        orientation="h",
        color="Correlación",
        text_auto=".2f"
    )

    st.plotly_chart(fig, use_container_width=True)

    ###########################################################
    # GPA POR NIVEL DE BURNOUT
    ###########################################################

    st.markdown("---")
    st.subheader("😵 Burnout y GPA")

    burnout = (
        filtered
        .groupby("Burnout_Risk_Level")
        ["Post_Semester_GPA"]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        burnout,
        x="Burnout_Risk_Level",
        y="Post_Semester_GPA",
        color="Burnout_Risk_Level",
        text_auto=".2f"
    )

    st.plotly_chart(fig, use_container_width=True)

    ###########################################################
    # HORAS DE IA VS HORAS DE ESTUDIO
    ###########################################################

    st.markdown("---")
    st.subheader("⏳ IA vs Estudio Tradicional")

    fig = px.scatter(
        filtered,
        x="Traditional_Study_Hours",
        y="Weekly_GenAI_Usage_Hours",
        color="Post_Semester_GPA",
        trendline="ols",
        color_continuous_scale="Viridis"
    )

    st.plotly_chart(fig, use_container_width=True)

    ###########################################################
    # DENSIDAD
    ###########################################################

    st.markdown("---")
    st.subheader("📉 Densidad del GPA")

    fig = px.density_contour(
        filtered,
        x="Weekly_GenAI_Usage_Hours",
        y="Post_Semester_GPA",
        color="Gender"
    )

    st.plotly_chart(fig, use_container_width=True)

    ###########################################################
    # SUNBURST
    ###########################################################

    st.markdown("---")
    st.subheader("🌞 Distribución Jerárquica")

    fig = px.sunburst(
        filtered,
        path=[
            "Major_Category",
            "Gender",
            "Burnout_Risk_Level"
        ]
    )

    st.plotly_chart(fig, use_container_width=True)

    ###########################################################
    # TREEMAP
    ###########################################################

    st.markdown("---")
    st.subheader("🌳 Treemap")

    fig = px.treemap(
        filtered,
        path=[
            "Major_Category",
            "Gender"
        ],
        values="Post_Semester_GPA",
        color="Post_Semester_GPA",
        color_continuous_scale="Blues"
    )

    st.plotly_chart(fig, use_container_width=True)

    ###########################################################
    # PIE CHART
    ###########################################################

    st.markdown("---")
    st.subheader("🥧 Distribución por Carrera")

    carrera = (
        filtered["Major_Category"]
        .value_counts()
        .reset_index()
    )

    carrera.columns = [
        "Carrera",
        "Cantidad"
    ]

    fig = px.pie(
        carrera,
        names="Carrera",
        values="Cantidad",
        hole=0.45
    )

    st.plotly_chart(fig, use_container_width=True)

    ###########################################################
    # HISTOGRAMA INTERACTIVO
    ###########################################################

    st.markdown("---")
    st.subheader("📊 Histograma Interactivo")

    variable = st.selectbox(
        "Seleccione una variable",
        scatter_cols,
        key="histograma"
    )

    fig = px.histogram(
        filtered,
        x=variable,
        nbins=30,
        marginal="rug",
        color_discrete_sequence=["royalblue"]
    )

    st.plotly_chart(fig, use_container_width=True)

    ###########################################################
    # EXPORTAR FIGURAS
    ###########################################################

    st.markdown("---")
    st.subheader("💾 Exportar Resultados")

    os.makedirs("outputs/figures", exist_ok=True)

    if st.button("Guardar Matriz de Correlación"):

        plt.figure(figsize=(12,8))

        sns.heatmap(
            corr,
            annot=True,
            cmap="RdBu_r"
        )

        plt.tight_layout()

        plt.savefig(
            "outputs/figures/correlacion.png",
            dpi=300
        )

        plt.close()

        st.success("Figura guardada en outputs/figures")

    ###########################################################
    # KPIs VISUALES
    ###########################################################

    st.markdown("---")
    st.subheader("📌 Indicadores del Dataset")

    k1, k2, k3 = st.columns(3)

    with k1:

        st.metric(
            "Promedio GPA",
            round(filtered["Post_Semester_GPA"].mean(),2)
        )

    with k2:

        st.metric(
            "Promedio Horas IA",
            round(filtered["Weekly_GenAI_Usage_Hours"].mean(),2)
        )

    with k3:

        st.metric(
            "Promedio Retención",
            round(filtered["Skill_Retention_Score"].mean(),2)
        )

    st.success("Visualizaciones completadas correctamente.")
# ============================================================
# PARTE 4 - MODELO PREDICTIVO MACHINE LEARNING
# ============================================================

elif menu == "🤖 Modelo Predictivo":

    st.title("🤖 Modelo Predictivo del Impacto de la IA en el Rendimiento Académico")

    st.write("""
    En esta sección se entrena un modelo de Machine Learning para predecir 
    el rendimiento académico de los estudiantes utilizando variables 
    relacionadas con el uso de inteligencia artificial.
    """)

    # Librerías ML
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import numpy as np


    df_model = df.copy()


    # ------------------------------------------------------------
    # Codificación variables categóricas
    # ------------------------------------------------------------

    encoder = LabelEncoder()

    categorical_columns = df_model.select_dtypes(
        include=["object"]
    ).columns


    for col in categorical_columns:
        df_model[col] = encoder.fit_transform(
            df_model[col].astype(str)
        )


    # ------------------------------------------------------------
    # Selección variable objetivo
    # ------------------------------------------------------------

    possible_targets = [
        col for col in df_model.columns
        if "gpa" in col.lower()
        or "score" in col.lower()
        or "grade" in col.lower()
    ]


    if len(possible_targets) > 0:

        target = st.selectbox(
            "Selecciona la variable objetivo:",
            possible_targets
        )


        X = df_model.drop(columns=[target])
        y = df_model[target]


        # --------------------------------------------------------
        # División datos
        # --------------------------------------------------------

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )


        # --------------------------------------------------------
        # Entrenamiento modelo
        # --------------------------------------------------------

        model = RandomForestRegressor(
            n_estimators=200,
            random_state=42
        )


        with st.spinner("Entrenando modelo..."):

            model.fit(
                X_train,
                y_train
            )


        # --------------------------------------------------------
        # Predicción
        # --------------------------------------------------------

        predictions = model.predict(
            X_test
        )


        # --------------------------------------------------------
        # Métricas
        # --------------------------------------------------------

        st.subheader("📊 Evaluación del Modelo")


        col1, col2, col3 = st.columns(3)


        with col1:

            mae = mean_absolute_error(
                y_test,
                predictions
            )

            st.metric(
                "Error Absoluto Medio",
                round(mae,3)
            )


        with col2:

            rmse = np.sqrt(
                mean_squared_error(
                    y_test,
                    predictions
                )
            )

            st.metric(
                "RMSE",
                round(rmse,3)
            )


        with col3:

            r2 = r2_score(
                y_test,
                predictions
            )

            st.metric(
                "R² Score",
                round(r2,3)
            )


        # --------------------------------------------------------
        # Resultados predicción
        # --------------------------------------------------------

        resultados = pd.DataFrame({

            "Valor Real": y_test.values,
            "Predicción": predictions

        })


        st.subheader(
            "🔎 Comparación valores reales vs predichos"
        )


        st.dataframe(
            resultados.head(20)
        )


        # --------------------------------------------------------
        # Importancia variables
        # --------------------------------------------------------

        st.subheader(
            "⭐ Importancia de variables"
        )


        importancia = pd.DataFrame({

            "Variable": X.columns,

            "Importancia":
                model.feature_importances_

        })


        importancia = importancia.sort_values(
            by="Importancia",
            ascending=False
        )


        fig_importancia = px.bar(
            importancia.head(15),
            x="Importancia",
            y="Variable",
            orientation="h",
            title="Variables más influyentes en la predicción"
        )


        st.plotly_chart(
            fig_importancia,
            use_container_width=True
        )


    else:

        st.warning(
            "No se encontró una variable adecuada como objetivo predictivo."
        )
# ============================================================
# PARTE 5 - EXPORTACIÓN DE RESULTADOS Y REPORTE FINAL
# ============================================================

elif menu == "📥 Reportes y Exportación":

    st.title("📥 Reportes y Exportación de Resultados")

    st.write("""
    En esta sección puedes descargar los resultados del análisis,
    datos procesados y reportes generados por la aplicación.
    """)


    # ------------------------------------------------------------
    # Exportar Dataset Procesado
    # ------------------------------------------------------------

    st.subheader("📂 Descargar Dataset Procesado")


    csv_data = df.to_csv(
        index=False
    ).encode("utf-8")


    st.download_button(
        label="⬇️ Descargar CSV Procesado",
        data=csv_data,
        file_name="ai_student_analysis_processed.csv",
        mime="text/csv"
    )



    # ------------------------------------------------------------
    # Resumen estadístico
    # ------------------------------------------------------------

    st.subheader(
        "📊 Resumen Estadístico"
    )


    resumen = df.describe()


    st.dataframe(
        resumen
    )


    resumen_csv = resumen.to_csv().encode(
        "utf-8"
    )


    st.download_button(

        label="⬇️ Descargar Resumen Estadístico",

        data=resumen_csv,

        file_name="resumen_estadistico.csv",

        mime="text/csv"

    )



    # ------------------------------------------------------------
    # Reporte automático
    # ------------------------------------------------------------

    st.subheader(
        "📄 Generar Reporte Automático"
    )


    total_registros = df.shape[0]

    total_variables = df.shape[1]


    reporte = f"""

    REPORTE DE ANÁLISIS
    ====================

    Proyecto:
    Impacto de la Inteligencia Artificial
    en el Rendimiento Académico


    Fecha de generación:
    {pd.Timestamp.today()}


    RESUMEN DATASET

    Número de estudiantes:
    {total_registros}


    Número de variables:
    {total_variables}


    VARIABLES DISPONIBLES:

    {list(df.columns)}


    RESUMEN ESTADÍSTICO:

    {resumen}

    """


    st.text_area(
        "Vista previa del reporte",
        reporte,
        height=400
    )


    st.download_button(

        label="⬇️ Descargar Reporte TXT",

        data=reporte,

        file_name="reporte_impacto_ia.txt",

        mime="text/plain"

    )



    # ------------------------------------------------------------
    # Exportar gráficos principales
    # ------------------------------------------------------------

    st.subheader(
        "📈 Exportación de Visualizaciones"
    )


    st.info(
        """
        Los gráficos generados en las secciones anteriores 
        pueden exportarse utilizando la opción de descarga 
        disponible en cada visualización.
        """
    )



    # ------------------------------------------------------------
    # Información despliegue GitHub
    # ------------------------------------------------------------

    st.subheader(
        "🚀 Preparación para GitHub y Streamlit Cloud"
    )


    estructura = """

    Proyecto/
    │
    ├── app.py
    ├── requirements.txt
    ├── README.md
    ├── data/
    │   └── ai_student_impact_dataset.csv
    │
    └── notebooks/
        └── analisis_exploratorio.ipynb


    """

    st.code(
        estructura,
        language="text"
    )


    st.success(
        """
        La aplicación está lista para ser publicada.
        
        Próximos pasos:
        
        1. Crear repositorio en GitHub.
        2. Subir app.py.
        3. Crear requirements.txt.
        4. Conectar repositorio con Streamlit Cloud.
        5. Ejecutar despliegue.
        """
    )

menu = st.sidebar.selectbox(
    "Menú",
    [
        "📊 Análisis Exploratorio",
        "📈 Visualizaciones",
        "🤖 Modelo Predictivo"
    ]
)
