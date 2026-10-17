import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


# --------------------------------------------------
# 1. Configuração da página
# --------------------------------------------------
st.set_page_config(
    page_title="Energia Industrial | Previsão",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("⚡ Previsão de Consumo de Energia Industrial")
st.markdown(
    """
    Simule o consumo de energia de uma linha de produção
    com base nas condições operacionais das máquinas.
    """
)

st.caption(
    "Modelo demonstrativo treinado com dados sintéticos. "
    "Os resultados não substituem medições reais da fábrica."
)


# --------------------------------------------------
# 2. Geração de dataset sintético (40 registros)
# --------------------------------------------------
@st.cache_data
def gerar_dataset():
    rng = np.random.default_rng(42)
    n = 40

    velocidade = rng.uniform(10, 80, n).round(1)
    maquinas = rng.integers(1, 11, n)
    temperatura = rng.uniform(15, 40, n).round(1)

    # Relação simulada entre variáveis e consumo.
    # O ruído representa variações não explicadas.
    consumo = (
        12
        + 0.85 * velocidade
        + 18.0 * maquinas
        + 1.2 * temperatura
        + rng.normal(0, 5, n)
    )

    df = pd.DataFrame({
        "Velocidade_Esteira": velocidade,
        "Maquinas_Ativas": maquinas,
        "Temp_Ambiente": temperatura,
        "Consumo_kWh": consumo.round(2),
    })

    return df


df = gerar_dataset()

FEATURES = [
    "Velocidade_Esteira",
    "Maquinas_Ativas",
    "Temp_Ambiente",
]

NOMES_VARIAVEIS = {
    "Velocidade_Esteira": "Velocidade da esteira",
    "Maquinas_Ativas": "Máquinas ativas",
    "Temp_Ambiente": "Temperatura ambiente",
}


# --------------------------------------------------
# 3. Treinamento do modelo de regressão
# --------------------------------------------------
X = df[FEATURES]
y = df["Consumo_kWh"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)

modelo = Pipeline([
    ("escalador", StandardScaler()),
    ("regressor", LinearRegression()),
])

modelo.fit(X_train, y_train)

y_test_pred = modelo.predict(X_test)

mae = mean_absolute_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)


# --------------------------------------------------
# 4. Controles na barra lateral
# --------------------------------------------------
st.sidebar.header("⚙️ Parâmetros operacionais")
st.sidebar.write(
    "Informe as condições atuais da linha de produção."
)

with st.sidebar.form("form_parametros"):
    velocidade_input = st.number_input(
        "Velocidade da esteira (m/min)",
        min_value=10.0,
        max_value=80.0,
        value=40.0,
        step=1.0,
        help="Velocidade de movimentação do material.",
    )

    maquinas_input = st.number_input(
        "Máquinas ativas",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
        help="Quantidade de máquinas em operação.",
    )

    temperatura_input = st.number_input(
        "Temperatura ambiente (°C)",
        min_value=15.0,
        max_value=40.0,
        value=25.0,
        step=0.5,
        help="Temperatura do ambiente industrial.",
    )

    calcular = st.form_submit_button(
        "⚡ Prever consumo",
        use_container_width=True,
        type="primary",
    )


# --------------------------------------------------
# 5. Previsão do consumo
# --------------------------------------------------
entrada = pd.DataFrame(
    [[
        velocidade_input,
        maquinas_input,
        temperatura_input,
    ]],
    columns=FEATURES,
)

consumo_previsto = float(modelo.predict(entrada)[0])

# A previsão pode ser calculada novamente a cada
# interação; o botão do formulário organiza a entrada.
st.subheader("Resultado da simulação")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Consumo estimado",
        value=f"{consumo_previsto:,.2f} kWh",
    )

with col2:
    st.metric(
        label="Erro absoluto médio (MAE)",
        value=f"{mae:.2f} kWh",
        help="Erro médio absoluto no conjunto de teste.",
    )

with col3:
    st.metric(
        label="R² no teste",
        value=f"{r2:.3f}",
        help=(
            "Indica quanto da variação do consumo "
            "é explicada pelo modelo no conjunto de teste."
        ),
    )

if calcular:
    st.success("Previsão atualizada com os parâmetros informados.")

st.caption(
    "O consumo previsto é uma estimativa do modelo, "
    "não uma medição em tempo real."
)


# --------------------------------------------------
# 6. Gráfico de contribuição das variáveis
# --------------------------------------------------
st.divider()
st.subheader("📊 Influência relativa das variáveis")

regressor = modelo.named_steps["regressor"]

# Os coeficientes estão na escala padronizada:
# permitem comparar a influência relativa das entradas.
coeficientes = pd.Series(
    regressor.coef_,
    index=FEATURES,
)

coeficientes.index = [
    NOMES_VARIAVEIS[nome] for nome in coeficientes.index
]

coeficientes = coeficientes.sort_values()

cores = [
    "#D9534F" if valor < 0 else "#168AAD"
    for valor in coeficientes.values
]

fig, ax = plt.subplots(figsize=(9, 4.5))

ax.barh(
    coeficientes.index,
    coeficientes.values,
    color=cores,
)

ax.axvline(0, color="gray", linewidth=0.8)
ax.set_title("Coeficientes padronizados da regressão")
ax.set_xlabel("Impacto estimado por desvio-padrão da variável")
ax.set_ylabel("")
ax.grid(axis="x", linestyle="--", alpha=0.3)
ax.set_axisbelow(True)

fig.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close(fig)

st.info(
    """
    **Como interpretar:** coeficientes positivos indicam associação
    com aumento do consumo previsto; coeficientes negativos indicam
    associação com redução. A magnitude permite comparar as variáveis
    nesta regressão, mas não demonstra causalidade.
    """
)


# --------------------------------------------------
# 7. Dados do treinamento
# --------------------------------------------------
with st.expander("🔎 Consultar dataset e dados do modelo"):
    st.write("Dataset sintético utilizado no treinamento:")
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.write("Resumo estatístico:")
    st.dataframe(
        df.describe().round(2),
        use_container_width=True,
    )

    st.write(
        f"Registros: {len(df)} | "
        f"Treinamento: {len(X_train)} | "
        f"Teste: {len(X_test)}"
    )

st.divider()
st.caption(
    "Projeto demonstrativo de Machine Learning | "
    "Python, Pandas, Scikit-learn, Matplotlib e Streamlit"
)
