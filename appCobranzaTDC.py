# app.py - Streamlit template para Smart Debt Recovery Assistant
# Autor: generado por asistente - template modular y listo para adaptar
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import joblib
import os
import plotly.graph_objects as go


sns.set_style("whitegrid")

# ---------------------------
# CONFIGURACIÓN / RUTAS
# ---------------------------
st.set_page_config(
    page_title="RecuperaIA",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ruta al PDF de referencia (archivo que subiste)
PDF_PATH = "/mnt/data/harvard-business-review-analytic-services-modernizing-debt-collection-through-ai-and-ei.pdf"

# Rutas por defecto (ajusta si es necesario)
DATA_PATH = "data/df_result_nuevos.csv"               # tu dataset final

# ---------------------------
# UTILIDADES
# ---------------------------
@st.cache_data
def cargar_datos(path=DATA_PATH, nrows=None):
    if not os.path.exists(path):
        st.warning(f"No se encontró {path}. Sube tu CSV a la carpeta del proyecto con ese nombre.")
        return pd.DataFrame()
    return pd.read_csv(path, nrows=nrows)


def estimar_tokens(texto):
    palabras = len(str(texto).split())
    return int(palabras / 0.75)

# ---------------------------
# SIDEBAR - NAVEGACIÓN Y FILTROS GLOBALES
# ---------------------------
with st.sidebar:
    st.title("RecuperaIA")
    st.markdown("**Menu**")
    page = st.radio("Ir a:", ["Inicio", "Dashboard", "Modelo (Propensión)", 
                              "Recomendador (Soluciones)", "Mensajería IA"])

    st.markdown("---")
    st.subheader("Filtros globales")
    # Filtros globales: si existen las columnas, se muestran
    df_preview = cargar_datos(nrows=50)
    if not df_preview.empty:
        if "region" in df_preview.columns:
            regiones = ["Todas"] + sorted(df_preview["region"].dropna().unique().tolist())
            region_sel = st.selectbox("Región", regiones)
        else:
            region_sel = "Todas"
        max_rows = 4000
        seg_sel = "Todos"
    else:
        region_sel = "Todas"
        seg_sel = "Todos"
        max_rows = 4000

    st.markdown("---")

# ---------------------------
# CARGAR DATOS (una vez centralizado)
# ---------------------------
nrows = None if max_rows == 0 else max_rows
df = cargar_datos(nrows=nrows)

# Aplicar filtros globales si corresponde
if not df.empty:
    if "region" in df.columns and region_sel != "Todas":
        df = df[df["region"] == region_sel]
    if "segmento_propension" in df.columns and seg_sel != "Todos":
        df = df[df["segmento_propension"] == seg_sel]

# ---------------------------
# PÁGINAS
# ---------------------------
def page_inicio():
    st.title("Sistema Inteligente de Recuperación de Cartera de TDC")
    st.markdown("**Resolviendo:** Predecir propensión de pago, priorizar gestiones y generar mensajes hiperpersonalizados.")
    st.markdown("---")
    col1, col2 = st.columns([3,1])
    with col1:
        st.header("Resumen ejecutivo")
        st.markdown("""
        - Modelo principal: **Propensión / Capacidad de pago** (LightGBM)
        - Regla de soluciones: Reglas de negocio para asignar prórroga, reestructura, planes, etc.
        - Mensajes: Generación por IA (Gemini) .
        """)
        
    with col2:
        st.image("https://images.unsplash.com/photo-1556740749-887f6717d7e4?w=800", use_column_width=True)
    st.markdown("---")
    st.subheader("Dataset & metadatos")
    if df.empty:
        st.info("Dataset no cargado. Sube 'dataset_limpio.csv' en el folder del proyecto.")
    else:
        st.write("Registros cargados:", len(df))
        st.dataframe(df.head(5))

def page_dashboard():
    st.title("📊 Dashboard Ejecutivo de Cobranza Inteligente")
    st.caption(
        "Visión integral del estado del portafolio, riesgo de clientes y efectividad de recuperación "
        "basada en modelos de propensión y soluciones personalizadas."
    )

    st.divider()

    if df.empty:
        st.info("Dataset vacío: cargar CSV en la ruta indicada.")
        return
    
    st.subheader("📌 Vista general del portafolio")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="👥 Clientes en gestión",
            value=f"{len(df):,}"
        )

    with col2:
        if "probabilidad_pago_30d" in df.columns:
            st.metric(
                label="✅ Prob. promedio de pago",
                value=f"{df['probabilidad_pago_30d'].mean()*100:.1f}%",
                help="Probabilidad estimada de que el cliente pague en los próximos 30 días."
            )
        else:
            st.metric("✅ Prob. promedio de pago", "N/D")

    with col3:
        if "vulnerabilidad_detectada" in df.columns:
            st.metric(
                label="⚠️ Clientes vulnerables",
                value=f"{df['vulnerabilidad_detectada'].mean()*100:.0f}%",
                help="Clientes con señales financieras o emocionales de vulnerabilidad."
            )
        else:
            st.metric("⚠️ Clientes vulnerables", "N/D")

    with col4:
        if "porcentaje_utilizacion" in df.columns:
            st.metric(
                label="💳 Uso promedio del crédito",
                value=f"{df['porcentaje_utilizacion'].mean():.1f}%",
                help="Porcentaje promedio de uso sobre el límite de crédito."
            )
        else:
            st.metric("💳 Uso promedio del crédito", "N/D")

    st.subheader("💰 Riesgo y recuperación del portafolio")

    recuperacion_efectiva = (
        df['monto_recuperado'].sum() / df['tdc_saldo_actual'].sum() * 100
        if df['tdc_saldo_actual'].sum() > 0 else 0
    )

    col5, col6, col7, col8 = st.columns(4)

    with col5:
        if "dias_atraso_actual" in df.columns:
            st.metric(
                label="⏳ Días de atraso (prom.)",
                value=f"{df['dias_atraso_actual'].mean():.1f}",
                help="Promedio de días de atraso en el portafolio."
            )
        else:
            st.metric("⏳ Días de atraso (prom.)", "N/D")

    with col6:
        if "tdc_saldo_actual" in df.columns:
            st.metric(
                label="📉 Deuda total",
                value=f"${df['tdc_saldo_actual'].sum():,.0f}",
                help="Saldo total adeudado por los clientes."
            )
        else:
            st.metric("📉 Deuda total", "N/D")

    with col7:
        if "monto_recuperado" in df.columns:
            st.metric(
                label="💵 Monto recuperado",
                value=f"${df['monto_recuperado'].sum():,.0f}",
                help="Monto total recuperado mediante gestiones previas."
            )
        else:
            st.metric("💵 Monto recuperado", "N/D")

    with col8:
        st.metric(
            label="📈 Recuperación efectiva",
            value=f"{recuperacion_efectiva:.0f}%",
            help="Monto recuperado sobre el total de deuda del portafolio."
        )

        
    st.markdown("---")
    # Visualización 1: Distribución de propensión
    st.subheader("Distribución de propensión de pago")
    if "segmento_propension" in df.columns:
        fig = px.histogram(df, x="segmento_propension", nbins=30, title="Histograma: probabilidad de pago")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("La columna 'segmento_propension' no existe en el dataset.")

    # Visualización 2: Heatmap Atraso vs Utilización vs Probabilidad (mapa de calor)
    st.subheader("Heatmap: Días de atraso × % Utilización → Prob. de pago")
    st.text("Clientes con pocos días de atraso y baja utilización: alta probabilidad de pago → se puede usar recordatorio simple")
    st.text("Clientes con muchos días de atraso y alta utilización: baja probabilidad de pago → se requiere acción directa, plan flexible o incentivo")
    if set(["dias_atraso_actual","porcentaje_utilizacion","probabilidad_pago_30d"]).issubset(df.columns):
        # Agrupación y pivot
        df_plot = df.copy()
        # Bin days and utilization
        df_plot["dias_bin"] = pd.cut(df_plot["dias_atraso_actual"], bins=10)
        df_plot["util_bin"] = pd.cut(df_plot["porcentaje_utilizacion"], bins=10)
        heat = df_plot.groupby(["dias_bin","util_bin"])["probabilidad_pago_30d"].mean().reset_index()
        heat_pivot = heat.pivot(index="util_bin", columns="dias_bin", values="probabilidad_pago_30d")
        fig2, ax = plt.subplots(figsize=(10,6))
        sns.heatmap(heat_pivot, ax=ax, cmap="RdYlGn_r", cbar_kws={'label':'Prob pago'})
        ax.set_xlabel("Días atraso (bins)")
        ax.set_ylabel("Utilización (bins)")
        st.pyplot(fig2)
    else:
        st.info("Columnas necesarias para heatmap faltantes.")

    # Visualización 3: Embudo de contacto (si hay datos)
    st.subheader("Embudo de contacto: Envíos → Aperturas → Respuestas → Pagos")
    if set(["historial_llamadas_realizadas","tasa_respuesta_llamadas"]).issubset(df.columns):
       # Datos del embudo
        etapas = ["Envíos", "Aperturas", "Respuestas", "Pagos"]
        valores = [len(df), int(df[["tasa_lectura_sms","tasa_apertura_email","tasa_respuesta_whatsapp"]].mean().mean()*len(df)),
                    int(df[["tasa_respuesta_whatsapp","tasa_respuesta_llamadas"]].mean().mean()*len(df)),
                    int(df["prediccion_pago_30d"].sum() if "prediccion_pago_30d" in df.columns else 0)]  # Ejemplo de cantidad de clientes por etapa    
        fig = go.Figure(go.Funnel(
        y = etapas,
        x = valores,
        textinfo = "value+percent initial"
        ))

        fig.update_layout(title="Embudo de Contacto: Envíos → Aperturas → Respuestas → Pagos")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Datos de interacción incompletos para embudo.")

    st.markdown("### Boxplot: monto_recuperado por segmento_propension (si existe)")
    if "monto_recuperado" in df.columns and "segmento_propension" in df.columns:
        fig = px.box(df, x="segmento_propension", y="monto_recuperado", title="Monto recuperado por segmento")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Columnas necesarias no disponibles para boxplot.")
    
    #visualización 4
    # Ejemplo de datos
    df_eje = pd.DataFrame({
        'segmento_propension': ['Alta', 'Media', 'Baja'],
        'tasa_recuperacion': [0.75, 0.50, 0.20]
    })

    plt.figure(figsize=(8,5))
    sns.barplot(data=df_eje, x='segmento_propension', y='tasa_recuperacion', palette="Blues_d")
    plt.ylim(0,1)
    plt.ylabel("Tasa de Recuperación")
    plt.xlabel("Segmento de Propensión")
    plt.title("Tasas de Recuperación por Segmento de Propensión")
    plt.show()
   

def page_modelo():
    st.title("Modelo: Propensión / Capacidad de pago")

    if df.empty:
        st.info("Dataset vacío.")
        return

    st.markdown("""
    ¿A quién debo priorizar, con qué urgencia y por qué?
    """)
    
    st.caption(
    "Distribución de clientes y saldo total por nivel de propensión de pago. "
    "Este indicador guía la priorización de estrategias de comunicación y recuperación."
    )
    # Total de clientes
    total_clientes = len(df)

    # KPIs por segmento
    kpi_seg = (
        df
        .groupby("segmento_propension")
        .agg(
            clientes=("segmento_propension", "count"),
            saldo_total=("tdc_saldo_actual", "sum")
        )
        .reset_index()
    )

    # Porcentaje de clientes
    kpi_seg["pct_clientes"] = (kpi_seg["clientes"] / total_clientes) * 100

    col1, col2, col3 = st.columns(3)

    for col, segmento, emoji in zip(
        [col1, col2, col3],
        ["Alto", "Medio", "Bajo"],
        ["🟢", "🟡", "🔴"]
    ):
        row = kpi_seg[kpi_seg["segmento_propension"] == segmento].iloc[0]

        with col:
            st.metric(
                label=f"{emoji} Propensión {segmento}",
                value=f"{row['pct_clientes']:.1f}%",
                delta=f"Saldo: ${row['saldo_total']/1_000_000:.2f} M",
                delta_color="off"
            )
   
    st.markdown("---")

    segmento = st.selectbox(
            "Segmento",
            ["Todos"] + sorted(df["segmento_propension"].dropna().unique().tolist())
            if "segmento_propension" in df.columns else ["Todos"]
        )
    
    # Aplicar filtros
    df_filt = df.copy()

    if segmento != "Todos":
        df_filt = df_filt[df_filt["segmento_propension"] == segmento]
    
    st.markdown(f"**Registros filtrados:** {len(df_filt)}")
    st.markdown("---")

    # Seleccionar cliente
    cliente = st.selectbox("Selecciona el cliente (index)", df_filt.index.tolist())

    # Selección del cliente
    #cliente = st.selectbox("Selecciona un cliente (index)", df.index.tolist())

    row = df.loc[cliente]

    st.subheader("📌 Resultados del modelo para este cliente")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Probabilidad de pago", f"{row.get('probabilidad_pago_30d', float('nan')):.2f}")

    with col2:
        st.metric("Segmento", row.get("segmento_propension", "N/D"))

    with col3:
        st.metric("Solución recomendada", row.get("solucion_recomendada", "N/D"))

    st.markdown("---")

    st.subheader("Detalle del cliente")
    mostrar = ["ingresos_mensuales","porcentaje_utilizacion",
            "dias_atraso_actual","vulnerabilidad_detectada",
            "sentimiento_ultima_interaccion","canal_preferido_cliente"]

    mostrar = [c for c in mostrar if c in df.columns]

    st.write(row[mostrar])

    st.markdown("---")

    st.subheader("Justificación del resultado")
    st.write(row.get("justificacion_corta", "Sin justificación en dataset."))



def page_recomendador():
    st.title("Recomendador de Soluciones (Reglas)")
    # ============================
    # SECCIÓN: SOLUCIONES RECOMENDADAS
    # ============================

    st.subheader("💡 Soluciones de pago recomendadas")

    st.markdown(
        "Esta sección muestra las **soluciones de pago sugeridas** por el motor de decisión, "
        "agrupadas por **nivel de propensión de pago**, junto con la **regla aplicada** y su "
        "**justificación de negocio**."
    )

    st.divider()

    # ============================
    # FILTRO PRINCIPAL
    # ============================

    segmentos = sorted(df["segmento_propension"].dropna().unique())
    segmento_sel = st.selectbox(
        "Selecciona el segmento de propensión:",
        segmentos
    )

    df_seg = df[df["segmento_propension"] == segmento_sel]

    # ============================
    # KPI SUPERIOR
    # ============================

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Clientes en el segmento", len(df_seg))

    with col2:
        st.metric(
            "Soluciones distintas recomendadas",
            df_seg["solucion_recomendada"].nunique()
        )

    st.divider()

    # ============================
    # VISUALIZACIÓN 1: BARRAS
    # ============================

    st.markdown("### 📊 Distribución de soluciones recomendadas")

    soluciones_count = (
        df_seg["solucion_recomendada"]
        .value_counts()
        .reset_index()
    )
    soluciones_count.columns = ["Solución recomendada", "Clientes"]

    st.bar_chart(
        soluciones_count.set_index("Solución recomendada"),
        height=300
    )

    st.divider()

    # ============================
    # VISUALIZACIÓN 2: TABLA DE DECISIÓN
    # ============================

    st.markdown("### 📋 Reglas aplicadas y justificación")

    tabla_reglas = (
        df_seg[
            [
                "solucion_recomendada",
                "regla_aplicada",
                "justificacion_corta"
            ]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    st.dataframe(
        tabla_reglas,
        use_container_width=True
    )

    st.subheader("📈 Recuperación efectiva por solución de pago")

    st.markdown(
        "La gráfica muestra qué **soluciones de pago** generan mayor **recuperación relativa del saldo** "
        "dentro del **segmento de propensión seleccionado**."
    )

    st.divider()

    # ============================
    # FILTRO GLOBAL
    # ============================

    df_seg = df[df["segmento_propension"] == segmento_sel].copy()

    # ============================
    # CÁLCULO DE MÉTRICA
    # ============================

    df_seg["ratio_recuperacion"] = (
        df_seg["monto_recuperado"] /
        df_seg["tdc_saldo_actual"]
    )

    df_seg = df_seg.replace([np.inf, -np.inf], np.nan)
    df_seg = df_seg.dropna(subset=["ratio_recuperacion"])

    # ============================
    # AGREGACIÓN POR SOLUCIÓN
    # ============================

    recuperacion_solucion = (
        df_seg
        .groupby("solucion_recomendada")
        .agg(
            ratio_promedio=("ratio_recuperacion", "mean")
        )
        .sort_values("ratio_promedio", ascending=False)
        .reset_index()
    )

    # ============================
    # VISUALIZACIÓN ÚNICA
    # ============================

    st.bar_chart(
        recuperacion_solucion.set_index("solucion_recomendada"),
        height=400
    )

    
  
def page_mensajeria():
    st.title("📨 Mensajes Personalizados por Canal")

    if df.empty:
        st.info("Dataset vacío.")
        return

    st.markdown("""
    Visualiza los mensajes personalizados generados previamente para cada cliente,
    mostrando un estilo visual distinto según el **canal óptimo** seleccionado.
    """)

    st.markdown("---")

    # filtros
    colf1, colf2, colf3 = st.columns(3)

    with colf1:
        segmento = st.selectbox(
            "Segmento",
            ["Todos"] + sorted(df["segmento_propension"].dropna().unique().tolist())
            if "segmento_propension" in df.columns else ["Todos"]
        )

    with colf2:
        canal = st.selectbox(
            "Canal óptimo",
            ["Todos"] + sorted(df["canal_optimo"].dropna().unique().tolist())
            if "canal_optimo" in df.columns else ["Todos"]
        )

    with colf3:
        solucion = st.selectbox(
            "Solución recomendada",
            ["Todos"] + sorted(df["solucion_recomendada"].dropna().unique().tolist())
            if "solucion_recomendada" in df.columns else ["Todos"]
        )

    # Aplicar filtros
    df_filt = df.copy()

    if segmento != "Todos":
        df_filt = df_filt[df_filt["segmento_propension"] == segmento]

    if canal != "Todos":
        df_filt = df_filt[df_filt["canal_optimo"] == canal]

    if solucion != "Todos":
        df_filt = df_filt[df_filt["solucion_recomendada"] == solucion]

    st.markdown(f"**Registros filtrados:** {len(df_filt)}")
    st.markdown("---")

    # Seleccionar cliente
    cliente = st.selectbox("Selecciona el cliente (index)", df_filt.index.tolist())

    row = df_filt.loc[cliente]

    st.subheader("📌 Información del cliente")
    cola, colb, colc = st.columns(3)

    with cola:
        st.metric("Prob. de pago", f"{row.get('probabilidad_pago_30d', float('nan')):.2f}")

    with colb:
        st.metric("Segmento", row.get("segmento_propension", "N/D"))

    with colc:
        st.metric("Solución", row.get("solucion_recomendada", "N/D"))

    st.markdown("---")
    st.subheader("📡 Información de comunicación")

    colm1, colm2, colm3 = st.columns(3)

    with colm1:
        st.write("**Canal óptimo:**", row.get("canal_optimo", "N/D"))

    with colm2:
        st.write("**Canal de respaldo:**", row.get("canal_respaldo", "N/D"))

    with colm3:
        st.write("**Tono del cliente:**", row.get("tono_cliente", "N/D"))

    st.markdown("---")

    # Mostrar mensaje según estilo por canal
    mensaje_raw = row.get("mensaje_generado")

    if pd.isna(mensaje_raw) or not str(mensaje_raw).strip():
        mensaje = "Mensaje no disponible para este cliente."
    else:
        mensaje = str(mensaje_raw).strip()
    #mensaje = row.get("mensaje_generado", "").strip()
    canal = row.get("canal_optimo", "").lower()

    st.subheader("📝 Mensaje personalizado")

    if mensaje == "":
        st.warning("El cliente no tiene mensaje generado.")
    else:
        # Estilo WhatsApp
        if canal == "whatsapp":
            st.markdown(f"""
            <div style="background:#ECE5DD;padding:15px;border-radius:15px;
                        width:70%; margin-bottom:10px;
                        border:1px solid #c1c1c1;">
                <div style="background:#DCF8C6;padding:15px;border-radius:15px;
                            font-size:17px;">
                    {mensaje}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Estilo Email
        elif canal == "email":
            st.markdown(f"""
            <div style="padding:20px;border-radius:8px;background:white;
                        border:1px solid #D0D0D0;">
                <p style="font-size:14px;color:#555;">
                    <b>De:</b> Santander Cobranza<br>
                    <b>Asunto:</b> Información importante sobre tu cuenta<br><br>
                </p>
                <p style="font-size:16px;">{mensaje}</p>
                <br>
                <hr>
                <p style="font-size:13px;color:#777;">
                    Este es un mensaje generado automáticamente.  
                    Si ya realizaste tu pago, por favor ignora esta notificación.
                </p>
            </div>
            """, unsafe_allow_html=True)

        # Estilo SMS
        elif canal == "sms":
            st.markdown(f"""
            <div style="background:#F4F4F4;padding:15px;border-radius:10px;
                        width:60%; font-size:16px;border:1px solid #DDD;">
                {mensaje}
            </div>
            """, unsafe_allow_html=True)

        # Estilo Llamada
        elif canal == "llamada":
            st.markdown(f"""
            <div style="padding:20px;border-radius:10px;background:#FAFAFA;
                        border-left:5px solid #6A1B9A;">
                <h4>📞 Guion de llamada sugerido</h4>
                <p style="font-size:17px;">
                {mensaje}
                </p>
            </div>
            """, unsafe_allow_html=True)

        # Default (por si canal no está bien definido)
        else:
            st.info("Canal no reconocido. Mostrando mensaje estándar.")
            st.markdown(f"""
            <div style="padding:18px;border-radius:10px;background:#F8F9FA;
                        border-left:5px solid #1A73E8;">
                <p style="font-size:17px;">{mensaje}</p>
            </div>
            """, unsafe_allow_html=True)

        st.caption(f"Tokens aproximados: {row.get('tokens', 'N/D')}")

    st.markdown("---")

    with st.expander("Ver todos los datos del cliente"):
        st.dataframe(pd.DataFrame([row]))

def page_slide():
    # ============================
    # SLIDE PREMIUM EXECUTIVE
    # ============================

    st.subheader("🔍 Problemas críticos en los procesos de cobranza")

    st.markdown("### La conversación mas dificil con un cliente es la deuda")
    st.markdown(
        "La cobranza tradicional sigue siendo reactiva, genérica y poco humana, "
        "estos son los problemas que más impactan."
    )

    # ======== PREMIUM LAYOUT: CARDS EN 2 COLUMNAS =========

    col1, col2 = st.columns(2, gap="large")

    with col1:
        with st.container(border=True):
            st.markdown("### 📨 Mensajes genéricos")
            st.markdown(
                "- No consideran la **situación emocional** del cliente.\n"
                "- No adaptan el mensaje a su **capacidad de pago**.\n"
                "- Causan baja conexión y mayor fricción."
            )

        with st.container(border=True):
            st.markdown("### 🤖 Comunicación impersonal")
            st.markdown(
                "- Automatización rígida.\n"
                "- Tono inadecuado.\n"
                "- Percepción negativa de ‘mensaje masivo’."
            )

        with st.container(border=True):
            st.markdown("### 🕒 Intervención tardía")
            st.markdown(
                "- No se personaliza en periodo **preventivo**.\n"
                "- Se actúa solo cuando el cliente ya está estresado.\n"
                "- Se pierden las mejores ventanas de recuperación."
            )

    with col2:
        with st.container(border=True):
            st.markdown("### 😣 Mala experiencia del cliente")
            st.markdown(
                "- Mensajes que generan presión o ansiedad.\n"
                "- Canal incorrecto aumenta el rechazo.\n"
                "- Reduce la disposición a pagar."
            )


        with st.container(border=True):
            st.markdown("### 🎯 Falta de personalización inteligente")
            st.markdown(             
                "- No se adapta el tono, canal o mensaje.\n"
                "- No hay priorización basada en riesgo."
            )

        with st.container(border=True):
            st.markdown("### 📉 Baja tasa de recuperación")
            st.markdown(
                "- Provisiones que afectan la rentabilidad.\n"
                "- Segmentación insuficiente.\n"
                "- Incapacidad de priorizar esfuerzos."
            )

    st.divider()


    # ============================
    # SLIDE KEYNOTE – VALOR EN 3 PREGUNTAS
    # ============================

    st.markdown("### 🚀 3 principales preguntas de negocio")

    with st.container(border=True):
        st.markdown(
            """
    ### 🧠 1. ¿Cómo podemos predecir la probabilidad y capacidad de pago de un cliente?   
    Con la finalidad de detonar acciones tempranas y optimizar la asignación de recursos.

    ### 📈 2. ¿Cómo ofrecer soluciones de pago personalizadas en etapas tempranas de la deuda? 
    Se desea considerar la situación financiera, emocional y el riesgo del cliente, para aumentar el cumplimiento y prevenir la entrada a mora. 

    ### ⚙️ 3. ¿Cómo podemos mejorar la comunicación con el cliente sin ser invasivos y genéricos?   
    Se busca adaptar los mensajes de acuerdo a su situación actual, por el canal y momento mas adecuado.
    """
        )

    st.divider()

     # ============================
    # DISEÑO TIPO KEYNOTE – 3 PILARES
    # ============================

    st.markdown("### 🔑 Tres Pilares de la Solución Inteligente")

    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        with st.container(border=True):
            st.markdown("## 🎯")
            st.markdown("### Modelo Propensión de pago")
            st.markdown(
                "- Se consideran datos de **riesgo**, **uso de la TDC**, **capacidad de pago**.\n"
                "- Modelo de clasificación Gradient Boosting.\n"
                "- LightGBM, ¿Pagar? y probabilidad de pago"
            )

    with col2:
        with st.container(border=True):
            st.markdown("## 🔌")
            st.markdown("### Soluciones de pago Personalizados")
            st.markdown(
                "- Propensión de pago.\n"
                "- Reglas de negocio personalizadas.\n"
                "- Considera comportamientos de pago, compra, etc."
            )

    with col3:
        with st.container(border=True):
            st.markdown("## 🤖")
            st.markdown("### IA generativa empática")
            st.markdown(
                "- Tono amable, humano y adecuado al contexto.\n"
                "- Comunicación que reduce ansiedad.\n"
                "- Se determina el canal adecuado."
            )

   

    # =======================================
    # SLIDE KEYNOTE PREMIUM – SOLUCIÓN
    # =======================================
    st.divider()
    st.markdown("## 🌟 Transformación de la Cobranza: De Genérica a Inteligente y Empática")

    st.markdown(
        "En un entorno donde los clientes viven bajo presión financiera y emocional, "
        "la cobranza tradicional ya no funciona. La siguiente es la visión moderna "
        "basada en IA generativa, analítica avanzada y comunicación humana."
    )
 
    st.divider()
    # ============================
    # CIERRE KEYNOTE
    # ============================

    st.markdown("### 🌈 Resultado Final")
    st.markdown(
        "**Una estrategia de cobranza moderna, empática y accionable, que eleva el desempeño operativo, mejora la experiencia del cliente y maximiza la recuperación de deuda.**"
    )
 



# ---------------------------
# RENDER PAGES
# ---------------------------
pages = {
    "Inicio": page_slide,
    "Dashboard": page_dashboard,
    "Modelo (Propensión)": page_modelo,
    "Recomendador (Soluciones)": page_recomendador,
    "Mensajería IA": page_mensajeria
}

pages_map = {
    "Inicio":"Inicio","Dashboard":"Dashboard","Modelo (Propensión)":"Modelo (Propensión)",
    "Recomendador (Soluciones)":"Recomendador (Soluciones)","Mensajería IA":"Mensajería IA"
}

# Ejecutar página seleccionada
if page in pages_map:
    pages[pages_map[page]]()
else:
    st.error("Página no encontrada.")

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
