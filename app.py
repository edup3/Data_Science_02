"""
Dashboard de Monitoreo Ambiental - Ciudades de Colombia
----------------------------------------------------------
- Análisis Exploratorio de Datos (EDA)
- Dashboard con storytelling sobre calidad del aire
- Asistente conversacional con Groq para preguntar sobre los datos

Ejecutar con:
    streamlit run app.py

Requiere una clave de API de Groq (https://console.groq.com/keys) para el
asistente conversacional. Se puede configurar como variable de entorno
GROQ_API_KEY, en st.secrets, o pegarla directamente en la barra lateral.
"""

import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Configuración general de la página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Monitoreo Ambiental - Colombia",
    page_icon="🌎",
    layout="wide",
    initial_sidebar_state="expanded",
)

RUTA_DATOS_LOCAL = os.path.join(os.path.dirname(__file__), "monitoreo_ambiental.csv")

ORDEN_ICA = [
    "Buena",
    "Moderada",
    "Dañina para grupos sensibles",
    "Dañina",
    "Muy Dañina",
    "Peligrosa",
]
COLORES_ICA = {
    "Buena": "#2ecc71",
    "Moderada": "#f1c40f",
    "Dañina para grupos sensibles": "#e67e22",
    "Dañina": "#e74c3c",
    "Muy Dañina": "#8e44ad",
    "Peligrosa": "#6e2c00",
}


# ---------------------------------------------------------------------------
# Carga y preparación de datos
# ---------------------------------------------------------------------------
@st.cache_data
def cargar_datos(archivo) -> pd.DataFrame:
    df = pd.read_csv(archivo)

    # Normalizar tipos
    if df["Presencia_Lluvia"].dtype == object:
        df["Presencia_Lluvia"] = df["Presencia_Lluvia"].astype(str).str.strip().isin(
            ["True", "1", "true", "Sí", "Si"]
        )

    # Hora -> hora numérica y franja del día
    horas = pd.to_datetime(df["Hora_Lectura"], format="%H:%M", errors="coerce")
    df["Hora_Num"] = horas.dt.hour + horas.dt.minute / 60

    def franja(h):
        if pd.isna(h):
            return "Sin dato"
        if 0 <= h < 6:
            return "Madrugada (00-06h)"
        if 6 <= h < 12:
            return "Mañana (06-12h)"
        if 12 <= h < 18:
            return "Tarde (12-18h)"
        return "Noche (18-24h)"

    df["Franja_Horaria"] = df["Hora_Num"].apply(franja)

    # Categoría ICA como ordinal
    categorias_presentes = [c for c in ORDEN_ICA if c in df["Indice_Calidad_Aire_ICA"].unique()]
    df["Indice_Calidad_Aire_ICA"] = pd.Categorical(
        df["Indice_Calidad_Aire_ICA"], categories=categorias_presentes, ordered=True
    )

    return df


st.sidebar.title("🌎 Monitoreo Ambiental")

if os.path.exists(RUTA_DATOS_LOCAL):
    df_raw = cargar_datos(RUTA_DATOS_LOCAL)
else:
    st.sidebar.warning("No se encontró 'monitoreo_ambiental.csv' junto a la app.")
    archivo_subido = st.sidebar.file_uploader("Sube el archivo CSV", type=["csv"])
    if archivo_subido is None:
        st.title("🌎 Dashboard de Monitoreo Ambiental")
        st.info(
            "Sube el archivo `monitoreo_ambiental.csv` desde la barra lateral "
            "para comenzar el análisis."
        )
        st.stop()
    df_raw = cargar_datos(archivo_subido)

# ---------------------------------------------------------------------------
# Sidebar - Filtros
# ---------------------------------------------------------------------------
st.sidebar.markdown("### Filtros")

ciudades_sel = st.sidebar.multiselect(
    "Ciudad", sorted(df_raw["Ciudad"].unique()), default=sorted(df_raw["Ciudad"].unique())
)
zonas_sel = st.sidebar.multiselect(
    "Tipo de zona",
    sorted(df_raw["Tipo_Zona"].unique()),
    default=sorted(df_raw["Tipo_Zona"].unique()),
)
lluvia_sel = st.sidebar.multiselect(
    "Presencia de lluvia",
    options=[False, True],
    default=[False, True],
    format_func=lambda x: "Con lluvia" if x else "Sin lluvia",
)

df = df_raw[
    df_raw["Ciudad"].isin(ciudades_sel)
    & df_raw["Tipo_Zona"].isin(zonas_sel)
    & df_raw["Presencia_Lluvia"].isin(lluvia_sel)
].copy()

st.sidebar.caption(f"Registros filtrados: **{len(df)}** de {len(df_raw)}")

st.sidebar.markdown("---")

# ---------------------------------------------------------------------------
# Sidebar - Configuración del asistente Groq
# ---------------------------------------------------------------------------
st.sidebar.markdown("### 🤖 Asistente Groq")
clave_por_defecto = os.environ.get("GROQ_API_KEY", "")
try:
    clave_por_defecto = st.secrets.get("GROQ_API_KEY", clave_por_defecto)
except Exception:
    pass

groq_api_key = st.sidebar.text_input(
    "Clave de API de Groq",
    value="",
    type="password",
    placeholder="gsk_..." if not clave_por_defecto else "Usando clave configurada",
    help="Obtén tu clave gratuita en https://console.groq.com/keys",
)
if not groq_api_key:
    groq_api_key = clave_por_defecto

modelo_groq = st.sidebar.selectbox(
    "Modelo",
    ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it"],
    index=0,
)

if df.empty:
    st.warning("No hay datos para los filtros seleccionados. Ajusta los filtros.")
    st.stop()

# ---------------------------------------------------------------------------
# Encabezado y KPIs
# ---------------------------------------------------------------------------
st.title("🌎 Dashboard de Monitoreo Ambiental - Colombia")
st.caption(
    "Datos de sensores de calidad del aire, ruido y clima en cinco ciudades "
    "colombianas. Explora el EDA, la narrativa de datos y consulta al "
    "asistente de IA."
)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Sensores", f"{len(df):,}")
k2.metric("PM2.5 promedio", f"{df['PM2_5_Ug_m3'].mean():.1f} µg/m³")
k3.metric("Temperatura prom.", f"{df['Temperatura_C'].mean():.1f} °C")
k4.metric("Ruido promedio", f"{df['Nivel_Ruido_dB'].mean():.1f} dB")
pct_dañino = df["Indice_Calidad_Aire_ICA"].isin(
    ["Dañina para grupos sensibles", "Dañina", "Muy Dañina", "Peligrosa"]
).mean() * 100
k5.metric("% Lecturas dañinas o peores", f"{pct_dañino:.1f}%")

st.markdown("---")

tab_eda, tab_story, tab_ia = st.tabs(
    ["📊 EDA - Análisis Exploratorio", "📖 Storytelling", "🤖 Asistente IA (Groq)"]
)

# ===========================================================================
# TAB 1: EDA
# ===========================================================================
with tab_eda:
    st.subheader("Vista general del conjunto de datos")

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("**Dimensiones y tipos de datos**")
        info_df = pd.DataFrame(
            {
                "Columna": df_raw.columns,
                "Tipo de dato": df_raw.dtypes.astype(str).values,
                "Valores únicos": [df_raw[c].nunique() for c in df_raw.columns],
                "Nulos": df_raw.isnull().sum().values,
            }
        )
        st.dataframe(info_df, use_container_width=True, hide_index=True)
        st.caption(f"Total de registros: {df_raw.shape[0]} · Columnas: {df_raw.shape[1]}")

    with c2:
        st.markdown("**Estadísticas descriptivas (numéricas, datos filtrados)**")
        st.dataframe(
            df[["PM2_5_Ug_m3", "Temperatura_C", "Humedad_Relativa_Pct", "Nivel_Ruido_dB"]]
            .describe()
            .round(2),
            use_container_width=True,
        )

    with st.expander("Ver muestra de los datos crudos"):
        st.dataframe(df.head(50), use_container_width=True)

    st.markdown("---")
    st.subheader("Distribuciones de variables numéricas")

    variables_num = {
        "PM2_5_Ug_m3": "PM2.5 (µg/m³)",
        "Temperatura_C": "Temperatura (°C)",
        "Humedad_Relativa_Pct": "Humedad relativa (%)",
        "Nivel_Ruido_dB": "Nivel de ruido (dB)",
    }
    cols_hist = st.columns(2)
    for i, (col, label) in enumerate(variables_num.items()):
        with cols_hist[i % 2]:
            fig = px.histogram(
                df, x=col, nbins=25, marginal="box",
                color_discrete_sequence=["#3498db"],
            )
            fig.update_layout(title=label, xaxis_title=label, yaxis_title="Frecuencia")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Variables categóricas")

    cols_cat = st.columns(3)
    with cols_cat[0]:
        conteo_ciudad = df["Ciudad"].value_counts().reset_index()
        conteo_ciudad.columns = ["Ciudad", "Conteo"]
        fig_c = px.bar(conteo_ciudad, x="Ciudad", y="Conteo", color="Ciudad")
        fig_c.update_layout(title="Sensores por ciudad", showlegend=False)
        st.plotly_chart(fig_c, use_container_width=True)

    with cols_cat[1]:
        conteo_zona = df["Tipo_Zona"].value_counts().reset_index()
        conteo_zona.columns = ["Tipo_Zona", "Conteo"]
        fig_z = px.bar(conteo_zona, x="Tipo_Zona", y="Conteo", color="Tipo_Zona")
        fig_z.update_layout(title="Sensores por tipo de zona", showlegend=False)
        st.plotly_chart(fig_z, use_container_width=True)

    with cols_cat[2]:
        conteo_ica = df["Indice_Calidad_Aire_ICA"].value_counts().reindex(ORDEN_ICA).dropna().reset_index()
        conteo_ica.columns = ["ICA", "Conteo"]
        fig_i = px.bar(
            conteo_ica, x="ICA", y="Conteo", color="ICA",
            color_discrete_map=COLORES_ICA, category_orders={"ICA": ORDEN_ICA},
        )
        fig_i.update_layout(title="Lecturas por índice de calidad del aire", showlegend=False)
        fig_i.update_xaxes(tickangle=-25)
        st.plotly_chart(fig_i, use_container_width=True)

    st.markdown("---")
    st.subheader("Correlación entre variables numéricas")
    num_cols = ["PM2_5_Ug_m3", "Temperatura_C", "Humedad_Relativa_Pct", "Nivel_Ruido_dB", "Hora_Num"]
    corr = df[num_cols].corr()
    fig_corr = go.Figure(
        data=go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            colorscale="RdBu", zmid=0,
            text=corr.round(2).values, texttemplate="%{text}",
        )
    )
    fig_corr.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=420)
    st.plotly_chart(fig_corr, use_container_width=True)
    st.caption(
        "Nota: coeficientes cercanos a 0 indican ausencia de relación lineal "
        "entre las variables analizadas."
    )

# ===========================================================================
# TAB 2: STORYTELLING
# ===========================================================================
with tab_story:
    # ---- Cálculo dinámico de insights ----
    pm_por_ciudad = df.groupby("Ciudad")["PM2_5_Ug_m3"].mean().sort_values(ascending=False)
    pm_por_zona = df.groupby("Tipo_Zona")["PM2_5_Ug_m3"].mean().sort_values(ascending=False)
    ciudad_top = pm_por_ciudad.index[0]
    zona_top = pm_por_zona.index[0]
    ciudad_mejor = pm_por_ciudad.index[-1]

    pm_lluvia = df.groupby("Presencia_Lluvia")["PM2_5_Ug_m3"].mean()
    dif_lluvia = pm_lluvia.get(True, np.nan) - pm_lluvia.get(False, np.nan)

    ruido_por_zona = df.groupby("Tipo_Zona")["Nivel_Ruido_dB"].mean().sort_values(ascending=False)
    zona_ruidosa = ruido_por_zona.index[0]

    corr_pm_ruido = df["PM2_5_Ug_m3"].corr(df["Nivel_Ruido_dB"])
    corr_pm_humedad = df["PM2_5_Ug_m3"].corr(df["Humedad_Relativa_Pct"])

    franja_top = df.groupby("Franja_Horaria")["PM2_5_Ug_m3"].mean().sort_values(ascending=False).index[0]

    st.subheader("La historia detrás de los datos")
    st.markdown(
        f"""
Este panel narra lo que revelan **{len(df)} lecturas** de sensores ambientales
distribuidos en cinco ciudades colombianas. El objetivo es entender **dónde,
cuándo y bajo qué condiciones** se concentra la contaminación por partículas
finas (PM2.5), y cómo se relaciona con el ruido, el clima y el tipo de zona.
"""
    )

    st.markdown("#### 1️⃣ ¿Qué ciudades respiran peor?")
    fig_ciudad = px.bar(
        pm_por_ciudad.reset_index(), x="Ciudad", y="PM2_5_Ug_m3",
        color="PM2_5_Ug_m3", color_continuous_scale="OrRd", text_auto=".1f",
    )
    fig_ciudad.update_layout(
        yaxis_title="PM2.5 promedio (µg/m³)", coloraxis_showscale=False
    )
    st.plotly_chart(fig_ciudad, use_container_width=True)
    st.markdown(
        f"**{ciudad_top}** registra el nivel promedio de PM2.5 más alto del "
        f"conjunto filtrado ({pm_por_ciudad.iloc[0]:.1f} µg/m³), mientras que "
        f"**{ciudad_mejor}** presenta el valor más bajo "
        f"({pm_por_ciudad.iloc[-1]:.1f} µg/m³)."
    )

    st.markdown("#### 2️⃣ ¿Importa el tipo de zona?")
    c1, c2 = st.columns([1.3, 1])
    with c1:
        fig_zona = px.box(
            df, x="Tipo_Zona", y="PM2_5_Ug_m3", color="Tipo_Zona",
            color_discrete_sequence=px.colors.qualitative.Set3,
        )
        fig_zona.update_layout(showlegend=False, yaxis_title="PM2.5 (µg/m³)")
        st.plotly_chart(fig_zona, use_container_width=True)
    with c2:
        st.markdown(
            f"""
La zona **{zona_top}** muestra el promedio de PM2.5 más elevado
({pm_por_zona.iloc[0]:.1f} µg/m³). Comparar la dispersión (cajas) entre tipos
de zona ayuda a identificar si la contaminación es un problema generalizado
o si está concentrada en zonas específicas de la ciudad.

Adicionalmente, la zona **{zona_ruidosa}** es la más ruidosa en promedio
({ruido_por_zona.iloc[0]:.1f} dB), lo cual suele coincidir con áreas de alto
tráfico o actividad industrial.
"""
        )

    st.markdown("#### 3️⃣ ¿Ayuda la lluvia a limpiar el aire?")
    c3, c4 = st.columns([1, 1.3])
    with c3:
        direccion = "más alto" if dif_lluvia > 0 else "más bajo"
        st.markdown(
            f"""
En este conjunto de datos, el PM2.5 promedio en horas **con lluvia** es
**{abs(dif_lluvia):.1f} µg/m³ {direccion}** que en horas sin lluvia.

- Sin lluvia: {pm_lluvia.get(False, float('nan')):.1f} µg/m³
- Con lluvia: {pm_lluvia.get(True, float('nan')):.1f} µg/m³

La correlación entre PM2.5 y humedad relativa es de **{corr_pm_humedad:.2f}**,
lo que sugiere una relación {"débil" if abs(corr_pm_humedad) < 0.3 else "moderada" if abs(corr_pm_humedad) < 0.6 else "fuerte"}
entre ambas variables en esta muestra.
"""
        )
    with c4:
        fig_lluvia = px.violin(
            df, x="Presencia_Lluvia", y="PM2_5_Ug_m3", color="Presencia_Lluvia",
            box=True, points="outliers",
            color_discrete_sequence=["#e74c3c", "#3498db"],
        )
        fig_lluvia.update_layout(
            xaxis_title="Presencia de lluvia", yaxis_title="PM2.5 (µg/m³)",
            showlegend=False,
        )
        fig_lluvia.update_xaxes(ticktext=["Sin lluvia", "Con lluvia"], tickvals=[False, True])
        st.plotly_chart(fig_lluvia, use_container_width=True)

    st.markdown("#### 4️⃣ ¿Hay un patrón a lo largo del día?")
    orden_franjas = ["Madrugada (00-06h)", "Mañana (06-12h)", "Tarde (12-18h)", "Noche (18-24h)"]
    pm_franja = (
        df.groupby("Franja_Horaria")["PM2_5_Ug_m3"].mean().reindex(orden_franjas).reset_index()
    )
    fig_franja = px.line(
        pm_franja, x="Franja_Horaria", y="PM2_5_Ug_m3", markers=True,
    )
    fig_franja.update_traces(line_color="#c0392b", marker=dict(size=10))
    fig_franja.update_layout(xaxis_title="Franja horaria", yaxis_title="PM2.5 promedio (µg/m³)")
    st.plotly_chart(fig_franja, use_container_width=True)
    st.markdown(
        f"La franja **{franja_top}** concentra los niveles promedio de PM2.5 "
        "más altos, lo que puede orientar el momento del día en que se "
        "recomienda evitar exposición prolongada en zonas críticas."
    )

    st.markdown("#### 5️⃣ Mapa de calor: Ciudad × Zona")
    pivot = df.pivot_table(
        index="Ciudad", columns="Tipo_Zona", values="PM2_5_Ug_m3", aggfunc="mean"
    )
    fig_pivot = go.Figure(
        data=go.Heatmap(
            z=pivot.values, x=pivot.columns, y=pivot.index,
            colorscale="OrRd", text=pivot.round(1).values, texttemplate="%{text}",
        )
    )
    fig_pivot.update_layout(
        xaxis_title="Tipo de zona", yaxis_title="Ciudad", height=380,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig_pivot, use_container_width=True)

    st.markdown("#### 6️⃣ Distribución del índice de calidad del aire (ICA)")
    conteo_ica_story = (
        df["Indice_Calidad_Aire_ICA"].value_counts().reindex(ORDEN_ICA).dropna().reset_index()
    )
    conteo_ica_story.columns = ["ICA", "Conteo"]
    fig_ica_story = px.pie(
        conteo_ica_story, names="ICA", values="Conteo", hole=0.45,
        color="ICA", color_discrete_map=COLORES_ICA,
        category_orders={"ICA": ORDEN_ICA},
    )
    st.plotly_chart(fig_ica_story, use_container_width=True)

    st.info(
        f"**Conclusión:** con los filtros actuales, el **{pct_dañino:.1f}%** de "
        "las lecturas se clasifican como dañinas o peores para la salud. "
        f"La combinación de mayor riesgo se concentra en **{ciudad_top}**, en "
        f"zonas de tipo **{zona_top}**, principalmente durante la "
        f"**{franja_top.lower()}**."
    )

# ===========================================================================
# TAB 3: ASISTENTE IA (GROQ)
# ===========================================================================
with tab_ia:
    st.subheader("Pregúntale a la IA sobre estos datos")
    st.caption(
        "El asistente responde con base en un resumen estadístico del "
        "conjunto de datos filtrado (no envía el archivo completo, solo "
        "agregados y estadísticas)."
    )

    if not groq_api_key:
        st.warning(
            "⚠️ No se ha configurado una clave de API de Groq. Ingrésala en la "
            "barra lateral para activar el asistente. Puedes obtener una "
            "clave gratuita en https://console.groq.com/keys"
        )
    else:
        try:
            from groq import Groq
        except ImportError:
            st.error(
                "La librería `groq` no está instalada. Agrega `groq` a tu "
                "requirements.txt e instala las dependencias."
            )
            st.stop()

        # -------------------------------------------------------------
        # Construcción del contexto de datos para el modelo
        # -------------------------------------------------------------
        def construir_contexto(df: pd.DataFrame) -> str:
            desc = df[
                ["PM2_5_Ug_m3", "Temperatura_C", "Humedad_Relativa_Pct", "Nivel_Ruido_dB"]
            ].describe().round(2).to_string()

            pm_ciudad = df.groupby("Ciudad")["PM2_5_Ug_m3"].mean().round(1).to_string()
            pm_zona = df.groupby("Tipo_Zona")["PM2_5_Ug_m3"].mean().round(1).to_string()
            ica_counts = (
                df["Indice_Calidad_Aire_ICA"].value_counts().reindex(ORDEN_ICA).dropna().to_string()
            )
            pm_lluvia = df.groupby("Presencia_Lluvia")["PM2_5_Ug_m3"].mean().round(1).to_string()
            pm_franja = df.groupby("Franja_Horaria")["PM2_5_Ug_m3"].mean().round(1).to_string()
            corr = df[
                ["PM2_5_Ug_m3", "Temperatura_C", "Humedad_Relativa_Pct", "Nivel_Ruido_dB"]
            ].corr().round(2).to_string()

            return f"""
Eres un analista de datos ambientales. A continuación tienes un resumen
estadístico de un conjunto de datos de monitoreo ambiental en ciudades de
Colombia (sensores de PM2.5, temperatura, humedad, ruido y calidad del aire).
Responde SIEMPRE con base en esta información. Si la pregunta no se puede
responder con estos datos, dilo explícitamente. Sé claro, conciso y usa
cifras concretas cuando sea posible. Responde en español.

Número de registros filtrados: {len(df)}
Ciudades incluidas: {', '.join(sorted(df['Ciudad'].unique()))}
Tipos de zona incluidos: {', '.join(sorted(df['Tipo_Zona'].unique()))}

--- Estadísticas descriptivas (numéricas) ---
{desc}

--- PM2.5 promedio por ciudad (µg/m³) ---
{pm_ciudad}

--- PM2.5 promedio por tipo de zona (µg/m³) ---
{pm_zona}

--- Conteo de lecturas por índice de calidad del aire (ICA) ---
{ica_counts}

--- PM2.5 promedio según presencia de lluvia (µg/m³) ---
{pm_lluvia}

--- PM2.5 promedio por franja horaria (µg/m³) ---
{pm_franja}

--- Matriz de correlación entre variables numéricas ---
{corr}
"""

        contexto = construir_contexto(df)

        if "mensajes_chat" not in st.session_state:
            st.session_state.mensajes_chat = []

        preguntas_sugeridas = [
            "¿Cuál ciudad tiene la peor calidad del aire?",
            "¿La lluvia reduce el PM2.5 en estos datos?",
            "¿Qué zona debería priorizarse para intervención?",
            "Resume los hallazgos principales en 3 puntos",
        ]
        st.markdown("**Preguntas sugeridas:**")
        cols_sugeridas = st.columns(len(preguntas_sugeridas))
        pregunta_click = None
        for i, p in enumerate(preguntas_sugeridas):
            if cols_sugeridas[i].button(p, use_container_width=True):
                pregunta_click = p

        for msg in st.session_state.mensajes_chat:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        pregunta_usuario = st.chat_input("Escribe tu pregunta sobre los datos...")
        pregunta_final = pregunta_click or pregunta_usuario

        if pregunta_final:
            st.session_state.mensajes_chat.append({"role": "user", "content": pregunta_final})
            with st.chat_message("user"):
                st.markdown(pregunta_final)

            with st.chat_message("assistant"):
                marcador = st.empty()
                marcador.markdown("Analizando datos... ⏳")
                try:
                    cliente = Groq(api_key=groq_api_key)

                    mensajes_api = [{"role": "system", "content": contexto}]
                    # Incluir historial reciente (últimos 6 mensajes) para contexto conversacional
                    for m in st.session_state.mensajes_chat[-6:]:
                        mensajes_api.append({"role": m["role"], "content": m["content"]})

                    respuesta = cliente.chat.completions.create(
                        model=modelo_groq,
                        messages=mensajes_api,
                        temperature=0.3,
                        max_tokens=700,
                    )
                    texto_respuesta = respuesta.choices[0].message.content
                    marcador.markdown(texto_respuesta)
                    st.session_state.mensajes_chat.append(
                        {"role": "assistant", "content": texto_respuesta}
                    )
                except Exception as e:
                    marcador.error(f"Ocurrió un error al consultar Groq: {e}")

        if st.session_state.mensajes_chat:
            if st.button("🗑️ Limpiar conversación"):
                st.session_state.mensajes_chat = []
                st.rerun()

st.markdown("---")
st.caption(
    "Dashboard construido con Streamlit, Plotly y Groq · Análisis de "
    "monitoreo ambiental en ciudades de Colombia."
)
