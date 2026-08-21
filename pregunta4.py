import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


st.set_page_config(
    page_title="Banco Perú - Riesgo Crediticio",
    page_icon="🏦",
    layout="wide"
)


df = pd.read_excel("banco_peru_scoring.xlsx")

np.random.seed(42)

df["edad_cliente"] = np.random.randint(
    21,
    71,
    len(df)
)

df["antiguedad_bancaria_meses"] = np.random.randint(
    6,
    241,
    len(df)
)

df["uso_canal_digital"] = np.random.choice(
    ["Bajo", "Medio", "Alto"],
    size=len(df),
    p=[0.25, 0.50, 0.25]
)


mediana_departamento = (
    df.groupby("departamento")["ingreso_mensual"]
      .transform("median")
)

df["ingreso_mensual"] = (
    df["ingreso_mensual"]
      .fillna(mediana_departamento)
)


Q1 = df["saldo_tarjeta"].quantile(0.25)
Q3 = df["saldo_tarjeta"].quantile(0.75)

IQR = Q3 - Q1

limite_superior = Q3 + 1.5 * IQR

df["saldo_tarjeta"] = (
    df["saldo_tarjeta"]
      .clip(upper=limite_superior)
)


y = df["target_default"]

X = df.drop(
    columns=[
        "target_default",
        "id_cliente"
    ]
)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)


columnas_numericas = [
    "ingreso_mensual",
    "saldo_tarjeta",
    "score_infocorp",
    "edad_cliente",
    "antiguedad_bancaria_meses"
]

columnas_categoricas = [
    "departamento",
    "uso_canal_digital"
]


preprocesamiento = ColumnTransformer(
    transformers=[
        (
            "numericas",
            StandardScaler(),
            columnas_numericas
        ),
        (
            "categoricas",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            columnas_categoricas
        )
    ]
)


pipeline_logistico = Pipeline(
    steps=[
        (
            "preprocesamiento",
            preprocesamiento
        ),
        (
            "modelo",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]
)


parametros_logistic = {
    "modelo__C": [
        0.01,
        0.1,
        1,
        10,
        100
    ]
}


grid_logistic = GridSearchCV(
    pipeline_logistico,
    parametros_logistic,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)

grid_logistic.fit(
    X_train,
    y_train
)

mejor_logistic = (
    grid_logistic.best_estimator_
)

prob_logistic = (
    mejor_logistic
    .predict_proba(X_test)[:, 1]
)

auc_logistic = roc_auc_score(
    y_test,
    prob_logistic
)


pipeline_rf = Pipeline(
    steps=[
        (
            "preprocesamiento",
            preprocesamiento
        ),
        (
            "modelo",
            RandomForestClassifier(
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


parametros_rf = {
    "modelo__n_estimators": [
        100,
        200
    ],
    "modelo__max_depth": [
        None,
        10,
        20
    ],
    "modelo__min_samples_split": [
        2,
        5
    ]
}


grid_rf = GridSearchCV(
    pipeline_rf,
    parametros_rf,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)

grid_rf.fit(
    X_train,
    y_train
)

mejor_rf = (
    grid_rf.best_estimator_
)

prob_rf = (
    mejor_rf
    .predict_proba(X_test)[:, 1]
)

auc_rf = roc_auc_score(
    y_test,
    prob_rf
)


variables_cluster = [
    "ingreso_mensual",
    "saldo_tarjeta",
    "score_infocorp",
    "edad_cliente",
    "antiguedad_bancaria_meses"
]


X_cluster = df[
    variables_cluster
]


scaler_cluster = StandardScaler()

X_cluster_scaled = (
    scaler_cluster
    .fit_transform(X_cluster)
)


siluetas = {}

for k in range(2, 8):

    modelo = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    etiquetas = (
        modelo
        .fit_predict(X_cluster_scaled)
    )

    siluetas[k] = silhouette_score(
        X_cluster_scaled,
        etiquetas
    )


mejor_k = max(
    siluetas,
    key=siluetas.get
)


kmeans = KMeans(
    n_clusters=mejor_k,
    random_state=42,
    n_init=10
)


df["cluster"] = (
    kmeans
    .fit_predict(X_cluster_scaled)
)


st.title(
    "🏦 Banco Perú - Dashboard de Riesgo Crediticio"
)

st.write(
    "Análisis de riesgo crediticio, desempeño de modelos "
    "predictivos y segmentación de clientes."
)


st.sidebar.header("Filtros")


departamentos = [
    "Todos"
] + sorted(
    df["departamento"].unique().tolist()
)


departamento = st.sidebar.selectbox(
    "Departamento",
    departamentos
)


if departamento == "Todos":

    datos = df.copy()

else:

    datos = df[
        df["departamento"] == departamento
    ]


clientes = len(datos)

defaults = int(
    datos["target_default"].sum()
)

tasa_default = (
    datos["target_default"].mean()
    * 100
)

ingreso_promedio = (
    datos["ingreso_mensual"].mean()
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Clientes",
        f"{clientes:,}"
    )


with col2:

    st.metric(
        "Defaults",
        f"{defaults:,}"
    )


with col3:

    st.metric(
        "Tasa de Default",
        f"{tasa_default:.2f}%"
    )


with col4:

    st.metric(
        "Ingreso Promedio",
        f"S/ {ingreso_promedio:,.0f}"
    )


st.divider()


st.subheader(
    "Tendencia del riesgo crediticio"
)


riesgo_departamento = (
    df.groupby("departamento")
      ["target_default"]
      .mean()
      .reset_index()
)


riesgo_departamento["tasa_default"] = (
    riesgo_departamento["target_default"]
    * 100
)


fig_riesgo = px.line(
    riesgo_departamento,
    x="departamento",
    y="tasa_default",
    markers=True,
    title="Tasa de Default por Departamento",
    labels={
        "departamento": "Departamento",
        "tasa_default": "Tasa de Default (%)"
    }
)


st.plotly_chart(
    fig_riesgo,
    use_container_width=True
)


st.subheader(
    "Desempeño de modelos predictivos"
)


datos_modelos = pd.DataFrame({

    "Modelo": [
        "Regresión Logística",
        "Random Forest"
    ],

    "ROC-AUC": [
        auc_logistic,
        auc_rf
    ]

})


fig_modelos = px.bar(
    datos_modelos,
    x="Modelo",
    y="ROC-AUC",
    text="ROC-AUC",
    title="Comparación del desempeño predictivo",
    labels={
        "ROC-AUC": "ROC-AUC"
    }
)


fig_modelos.update_traces(
    texttemplate="%{text:.4f}",
    textposition="outside"
)


st.plotly_chart(
    fig_modelos,
    use_container_width=True
)


st.subheader(
    "Segmentación de clientes"
)


conteo_clusters = (
    datos["cluster"]
    .value_counts()
    .reset_index()
)


conteo_clusters.columns = [
    "cluster",
    "clientes"
]


fig_clusters = px.bar(
    conteo_clusters,
    x="cluster",
    y="clientes",
    text="clientes",
    title="Volumen de clientes por Cluster",
    labels={
        "cluster": "Cluster",
        "clientes": "Cantidad de clientes"
    }
)


st.plotly_chart(
    fig_clusters,
    use_container_width=True
)


st.subheader(
    "Relación entre ingreso y saldo de tarjeta"
)


fig_dispersion = px.scatter(
    datos,
    x="ingreso_mensual",
    y="saldo_tarjeta",
    color="cluster",
    hover_data=[
        "score_infocorp",
        "edad_cliente",
        "antiguedad_bancaria_meses",
        "uso_canal_digital"
    ],
    title="Ingreso Mensual vs. Saldo de Tarjeta",
    labels={
        "ingreso_mensual": "Ingreso mensual (S/)",
        "saldo_tarjeta": "Saldo de tarjeta (S/)",
        "cluster": "Cluster"
    }
)


st.plotly_chart(
    fig_dispersion,
    use_container_width=True
)


st.subheader(
    "Perfil financiero de los segmentos"
)


perfil_clusters = (
    datos.groupby("cluster")
         .agg({
             "ingreso_mensual": "mean",
             "saldo_tarjeta": "mean",
             "score_infocorp": "mean",
             "edad_cliente": "mean",
             "antiguedad_bancaria_meses": "mean"
         })
         .round(2)
)


st.dataframe(
    perfil_clusters,
    use_container_width=True
)


st.subheader(
    "Uso del canal digital por Cluster"
)


canal_cluster = (
    pd.crosstab(
        datos["cluster"],
        datos["uso_canal_digital"],
        normalize="index"
    )
    * 100
)


st.dataframe(
    canal_cluster.round(2),
    use_container_width=True
)


st.subheader(
    "Indicadores de desempeño de los modelos"
)


col1, col2 = st.columns(2)


with col1:

    st.metric(
        "ROC-AUC Regresión Logística",
        f"{auc_logistic:.4f}"
    )


with col2:

    st.metric(
        "ROC-AUC Random Forest",
        f"{auc_rf:.4f}"
    )


st.subheader(
    "Configuración de segmentación"
)


st.write(
    f"Número óptimo de clusters: **{mejor_k}**"
)

st.write(
    f"Coeficiente de Silhouette: "
    f"**{siluetas[mejor_k]:.4f}**"
)


st.divider()


st.caption(
    "Banco Perú | Minería de Datos | "
    "Riesgo Crediticio y Segmentación"
)
