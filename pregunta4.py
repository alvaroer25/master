import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans
df = pd.read_excel(r"banco_peru_scoring.xlsx")
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
    transformers=[("numericas",
            StandardScaler(),
            columnas_numericas
        ),
        ("categoricas",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            columnas_categoricas)])

pipeline_logistico = Pipeline(
    steps=[("preprocesamiento",
            preprocesamiento
        ),("modelo",
            LogisticRegression(
                penalty="l2",
                max_iter=1000,
                random_state=42
            ))])

parametros_logistic = {"modelo__C": [
        0.01,
        0.1,
        1,
        10,
        100
    ]}

grid_logistic = GridSearchCV(pipeline_logistico,
    parametros_logistic,
    cv=5,
    scoring="roc_auc",n_jobs=-1)
grid_logistic.fit(X_train,y_train)


mejor_logistic = (grid_logistic.best_estimator_)
prob_logistic = (mejor_logistic.predict_proba(X_test)[:, 1])
auc_logistic = roc_auc_score(y_test,prob_logistic)


# ============================================================
# 4.2.2 RANDOM FOREST
# ============================================================

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
total_clientes = len(df)

total_default = int(
    df["target_default"].sum()
)

tasa_default = (
    df["target_default"].mean() * 100
)


promedio_ingreso = (
    df["ingreso_mensual"].mean()
)


promedio_saldo = (
    df["saldo_tarjeta"].mean()
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
from sklearn.metrics import silhouette_score
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
app = dash.Dash(__name__)
app.title = "Banco Perú - Riesgo Crediticio"
app.layout = html.Div(
    style={
        "fontFamily": "Arial",
        "padding": "20px"
    },
    children=[
        html.H1(
            "Banco Perú - Dashboard de Riesgo Crediticio",
            style={
                "textAlign": "center"
            }
        ),
        html.P(
            "Análisis de riesgo, desempeño predictivo y "
            "segmentación de clientes.",
            style={
                "textAlign": "center"
            }
        ),
        html.Div(
            [
                html.Label(
                    "Departamento:"
                ),
                dcc.Dropdown(
                    id="filtro-departamento",
                    options=[
                        {
                            "label": "Todos",
                            "value": "Todos"
                        }
                    ]
                    +
                    [
                        {
                            "label": departamento,
                            "value": departamento
                        }
                        for departamento
                        in sorted(
                            df["departamento"]
                            .unique()
                        )
                    ],
                    value="Todos",
                    clearable=False
                )

            ],
            style={
                "width": "300px",
                "marginBottom": "25px"
            }),
        html.Div(

            id="indicadores",

            style={
                "display": "flex",
                "gap": "15px",
                "marginBottom": "30px"
            }

        ),
        html.H2(
            "Tendencia del riesgo crediticio"
        ),
        dcc.Graph(
            id="grafico-riesgo"
        ),
        html.H2(
            "Desempeño de modelos predictivos"
        ),
        dcc.Graph(
            id="grafico-modelos"
        ),
        html.H2(
            "Segmentación de clientes"
        ),
        dcc.Graph(
            id="grafico-clusters"
        ),
        html.H2(
            "Relación entre ingreso y saldo de tarjeta"
        ),
        dcc.Graph(
            id="grafico-dispersion"
        )])

@app.callback(
    Output(
        "indicadores",
        "children"
    ),
    Output(
        "grafico-riesgo",
        "figure"
    ),
    Output(
        "grafico-modelos",
        "figure"
    ),
    Output(
        "grafico-clusters",
        "figure"
    ),
    Output(
        "grafico-dispersion",
        "figure"
    ),
    Input(
        "filtro-departamento",
        "value"
    ))

def actualizar_dashboard(departamento):
    if departamento == "Todos":
        datos = df.copy()
    else:
        datos = df[
            df["departamento"]
            == departamento]
    clientes = len(datos)
    defaults = int(datos["target_default"].sum())
    tasa = (datos["target_default"].mean()* 100)
    ingreso = (datos["ingreso_mensual"].mean())
    cards = [
        html.Div(
            [
                html.H3(
                    "Clientes"
                ),
                html.H2(
                    f"{clientes:,}"
                )
            ],
            style={
                "padding": "15px",
                "flex": "1",
                "background": "#2196f3",
                "color": "white"
            }),
        html.Div([
                html.H3(
                    "Defaults"
                ),
                html.H2(
                    f"{defaults:,}"
                )
            ],
            style={
                "padding": "15px",
                "flex": "1",
                "background": "#f44336",
                "color": "white"
            }),
        html.Div([
                html.H3(
                    "Tasa de Default"
                ),
                html.H2(
                    f"{tasa:.2f}%"
                )],
            style={
                "padding": "15px",
                "flex": "1",
                "background": "#ff9800",
                "color": "white"
            }),
        html.Div([
                html.H3(
                    "Ingreso Promedio"
                ),
                html.H2(
                    f"S/ {ingreso:,.0f}"
                )],
            style={
                "padding": "15px",
                "flex": "1",
                "background": "#4caf50",
                "color": "white"
            })]

    riesgo_departamento = (
        df.groupby("departamento")
          ["target_default"]
          .mean()
          .reset_index()
    )
    riesgo_departamento[
        "tasa_default"
    ] = (riesgo_departamento["target_default"] * 100)


    fig_riesgo = px.line(
        riesgo_departamento,
        x="departamento",
        y="tasa_default",
        markers=True,
        title=(
            "Tasa de Default por Departamento"
        ),
        labels={
            "departamento":
                "Departamento",

            "tasa_default":
                "Tasa de Default (%)"
        })

    datos_modelos = pd.DataFrame({
        "Modelo": [
            "Regresión Logística",
            "Random Forest"
        ],
        "ROC-AUC": [
            auc_logistic,
            auc_rf
        ]})

    fig_modelos = px.bar(
        datos_modelos,
        x="Modelo",
        y="ROC-AUC",
        text="ROC-AUC",
        title=("Comparación del desempeño predictivo"),
        labels={"ROC-AUC":"ROC-AUC"}
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
        title=("Volumen de clientes por cluster"),
        labels={
            "cluster":
                "Cluster",
            "clientes":
                "Cantidad de clientes"
        })

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
        title=(
            "Ingreso mensual vs. saldo de tarjeta"
        ),
        labels={
            "ingreso_mensual":
                "Ingreso mensual (S/)",
            "saldo_tarjeta":
                "Saldo de tarjeta (S/)",
            "cluster":
                "Cluster"
        })

    return (
        cards,
        fig_riesgo,
        fig_modelos,
        fig_clusters,
        fig_dispersion
    )

if __name__ == "__main__":

    app.run(
        debug=True
    )
