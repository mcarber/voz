from voz_utils import *

import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# ============================================
# CONFIGURACIÓN
# ============================================

st.set_page_config(
    page_title="Plataforma de Análisis de Voz",
    page_icon="🎙️",
    layout="wide"
)

# ============================================
# ESTILO CSS
# ============================================

st.markdown("""
<style>

/* Fondo general */
.stApp {
    background-color: #0f1117;
    color: white;
}

/* Títulos */
h1, h2, h3 {
    color: #00d4ff;
}

/* Texto */
p {
    font-size: 18px;
}

/* Botones */
.stButton>button {
    background-color: #00d4ff;
    color: black;
    border-radius: 12px;
    border: none;
    padding: 10px 20px;
    font-weight: bold;
}

/* Upload */
[data-testid="stFileUploader"] {
    border: 2px dashed #00d4ff;
    border-radius: 15px;
    padding: 20px;
    background-color: #161b22;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 20px;
}

.stTabs [data-baseweb="tab"] {
    background-color: #1b1f2a;
    border-radius: 10px;
    padding: 10px 20px;
}

/* Cards */
.card {
    background-color: #1b1f2a;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 0px 15px rgba(0,212,255,0.2);
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

# ============================================
# HEADER
# ============================================

st.title("🎙️ Detector de Voz IA")

st.markdown("""
### Aplicación interactiva para el análisis temporal, espectral y acústico de señales de voz mediante técnicas de procesamiento digital.
""")
st.markdown("""
## Sube una voz y analiza sus características acústicas
""")

# ============================================
# CARDS INFORMATIVAS
# ============================================

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="card">
        <h3>Pitch</h3>
        <p>Frecuencia fundamental de la voz</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
        <h3>MFCC</h3>
        <p>Características espectrales de la señal</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="card">
        <h3>LPC</h3>
        <p>Análisis de formantes y envolvente espectral</p>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ============================================
# SUBIR AUDIO
# ============================================

audio = st.file_uploader(
    "Sube un archivo de audio",
    type=["wav", "mp3"]
)

# ============================================
# SI HAY AUDIO
# ============================================

if audio is not None:

    st.success("✅ Audio cargado correctamente")

    # ============================================
    # REPRODUCTOR
    # ============================================

    st.audio(audio)

    # ============================================
    # CARGAR AUDIO
    # ============================================

    y, fs = librosa.load(audio, sr=22050)





    # ============================================
    # TABS
    # ============================================

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([

        "Forma temporal",
        "Energía",
        "Pitch",
        "Espectros",
        "MFCC",
        "LPC"

    ])

    # ============================================
    # TAB PITCH
    # ============================================
    
    with tab1:

        st.subheader("Forma temporal")

        st.write("""
        La representación temporal permite observar cómo evoluciona la amplitud de la señal a lo largo del tiempo.
        """)

        st.audio(audio)

        # Métricas básicas
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Frecuencia de muestreo",
                f"{fs} Hz"
            )

        with col2:
            st.metric(
                "Duración",
                f"{len(y)/fs:.2f} s"
            )

        with col3:
            st.metric(
                "Número de muestras",
                len(y)
            )

        # ============================================
        # FORMA DE ONDA
        # ============================================

        fig, ax = plt.subplots(figsize=(12,4))

        fig.patch.set_facecolor("#0f1117")
        ax.set_facecolor("#161b22")

        tiempos = np.arange(len(y)) / fs

        ax.plot(
            tiempos,
            y,
            color="#00d4ff",
            linewidth=1
        )

        ax.set_title(
            "Representación temporal del audio",
            color="white"
        )

        ax.set_xlabel(
            "Tiempo (s)",
            color="white"
        )

        ax.set_ylabel(
            "Amplitud",
            color="white"
        )

        ax.tick_params(colors='white')

        for spine in ax.spines.values():
            spine.set_color("white")

        ax.grid(
            color='white',
            alpha=0.1
        )

        st.pyplot(fig)

    with tab2:
        st.subheader("Análisis de energía")

        st.write("""
        La energía RMS permite estudiar cómo varía la intensidad de la señal de voz a lo largo del tiempo.
        """)

        # ============================================
        # ENERGÍA
        # ============================================

        rms, times, media, std, max_val = analizar_energia(
            y,
            fs
        )

        # ============================================
        # MÉTRICAS
        # ============================================

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Energía media",
                f"{media:.4f}"
            )

        with col2:
            st.metric(
                "Variabilidad",
                f"{std:.4f}"
            )

        with col3:
            st.metric(
                "Máximo RMS",
                f"{max_val:.4f}"
            )

        # ============================================
        # GRÁFICA
        # ============================================

        fig, ax = plt.subplots(figsize=(12,4))

        fig.patch.set_facecolor("#0f1117")
        ax.set_facecolor("#161b22")

        ax.plot(
            times,
            rms,
            color="#00d4ff",
            linewidth=2
        )

        ax.fill_between(
            times,
            rms,
            alpha=0.3,
            color="#00d4ff"
        )

        ax.set_title(
            "Evolución temporal de la energía",
            color="white"
        )

        ax.set_xlabel(
            "Tiempo (s)",
            color="white"
        )

        ax.set_ylabel(
            "RMS",
            color="white"
        )

        ax.tick_params(colors='white')

        for spine in ax.spines.values():
            spine.set_color("white")

        ax.grid(
            color='white',
            alpha=0.1
        )

        st.pyplot(fig)

        # ============================================
        # INTERPRETACIÓN
        # ============================================

        st.subheader("Interpretación")

        if std < 0.03:

            st.write("""
            La señal presenta una energía relativamente estable, con pocas variaciones de intensidad a lo largo del tiempo.
            """)

        elif std < 0.08:

            st.write("""
            La señal presenta variaciones moderadas de energía, compatibles con cambios naturales de intensidad vocal.
            """)

        else:

            st.write("""
            La señal presenta cambios pronunciados de energía y regiones de alta actividad sonora.
            """)


    with tab3:

        st.subheader("Análisis de Pitch")

        st.write("""
        El pitch representa la frecuencia fundamental de la voz y permite analizar la variabilidad tonal de la señal.
        """)

        # ============================================
        # ANÁLISIS DE PITCH
        # ============================================

        stats_pitch = analizar_pitch_voz(y, fs)

        f0 = stats_pitch['f0_serie']

        # Tiempo
        tiempos = 512 * np.arange(len(f0)) / fs

        # Métricas
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Pitch medio",
                f"{stats_pitch['media']:.2f} Hz"
            )

        with col2:
            st.metric(
                "Pitch mínimo",
                f"{stats_pitch['min']:.2f} Hz"
            )

        with col3:
            st.metric(
                "Pitch máximo",
                f"{stats_pitch['max']:.2f} Hz"
            )

        # ============================================
        # GRÁFICA
        # ============================================

        fig, ax = plt.subplots(figsize=(12,4))

        fig.patch.set_facecolor("#0f1117")
        ax.set_facecolor("#161b22")

        ax.plot(
            tiempos,
            f0,
            color="#00d4ff",
            linewidth=1.5
        )

        ax.set_title(
            "Trayectoria temporal del pitch",
            color="white"
        )

        ax.set_xlabel(
            "Tiempo (s)",
            color="white"
        )

        ax.set_ylabel(
            "Frecuencia (Hz)",
            color="white"
        )

        ax.tick_params(colors='white')

        for spine in ax.spines.values():
            spine.set_color("white")

        ax.grid(
            color='white',
            alpha=0.15,
            linestyle='--'
        )

        st.pyplot(fig)

    # ============================================
    # TAB MFCC
    # ============================================
    with tab4:

        st.subheader("Representaciones espectrales")

        st.write("""
        Las distintas representaciones espectrales permiten analizar cómo evoluciona el contenido frecuencial de la señal a lo largo del tiempo.
        """)

        # ============================================
        # SELECTOR
        # ============================================

        tipo = st.selectbox(

            "Selecciona una representación espectral",

            [
                "Lineal",
                "Log",
                "MEL",
                "CQT",
                "Cromagrama"
            ]

        )

        # ============================================
        # EXPLICACIONES
        # ============================================

        explicaciones = {

            "Lineal":
            """
            El espectrograma lineal representa directamente las frecuencias reales presentes en la señal.
            """,

            "Log":
            """
            La representación logarítmica permite visualizar mejor las bajas frecuencias y se aproxima más a la percepción humana.
            """,

            "MEL":
            """
            La escala MEL adapta las frecuencias según el comportamiento perceptual del oído humano y se utiliza ampliamente en reconocimiento de voz.
            """,

            "CQT":
            """
            La CQT organiza las frecuencias siguiendo una distribución logarítmica basada en notas musicales, siendo especialmente útil para señales musicales.
            """,

            "Cromagrama":
            """
            El cromagrama agrupa la energía espectral según clases tonales independientemente de la octava.
            """
        }

        st.write(explicaciones[tipo])

        # ============================================
        # OBTENER ESPECTRO
        # ============================================

        S_db, modo = obtener_espectrograma(
            y,
            fs,
            tipo
        )

        # ============================================
        # FIGURA
        # ============================================

        fig, ax = plt.subplots(figsize=(12,5))

        fig.patch.set_facecolor("#0f1117")
        ax.set_facecolor("#161b22")

        # ============================================
        # MOSTRAR SEGÚN TIPO
        # ============================================

        if modo == "linear":

            img = librosa.display.specshow(
                S_db,
                sr=fs,
                x_axis='time',
                y_axis='linear',
                cmap='magma',
                ax=ax
            )

        elif modo == "log":

            img = librosa.display.specshow(
                S_db,
                sr=fs,
                x_axis='time',
                y_axis='log',
                cmap='magma',
                ax=ax
            )

        elif modo == "mel":

            img = librosa.display.specshow(
                S_db,
                sr=fs,
                x_axis='time',
                y_axis='mel',
                cmap='magma',
                ax=ax
            )

        elif modo == "cqt":

            img = librosa.display.specshow(
                S_db,
                sr=fs,
                x_axis='time',
                y_axis='cqt_note',
                cmap='magma',
                ax=ax
            )

        elif modo == "chroma":

            img = librosa.display.specshow(
                S_db,
                y_axis='chroma',
                x_axis='time',
                cmap='magma',
                ax=ax
            )

        # ============================================
        # ESTILO
        # ============================================

        ax.set_title(
            f"Representación {tipo}",
            color="white",
            fontsize=16
        )

        ax.tick_params(colors='white')

        for spine in ax.spines.values():
            spine.set_color("white")

        cbar = fig.colorbar(img)

        cbar.ax.yaxis.set_tick_params(color='white')

        plt.setp(
            plt.getp(cbar.ax.axes, 'yticklabels'),
            color='white'
        )

        st.pyplot(fig)

    with tab5:

        st.subheader("Coeficientes MFCC")

        st.write("""
        Los coeficientes MFCC permiten representar características espectrales relacionadas con el timbre y contenido acústico de la voz.
        """)

        # ============================================
        # CÁLCULO MFCC
        # ============================================

        mfcc = calcular_mfcc(y, fs)

        # ============================================
        # MÉTRICAS
        # ============================================

        # ============================================
        # INFORMACIÓN MFCC
        # ============================================

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("""
            <div class="card">
            <h3>Coeficientes MFCC</h3>
            <p>Dimensionalidad del análisis acústico</p>
            </div>
            """, unsafe_allow_html=True)

            st.metric(
                "Número de coeficientes",
                mfcc.shape[0]
            )

        with col2:

            st.markdown("""
            <div class="card">
            <h3>Frames temporales</h3>
            <p>Resolución temporal del análisis</p>
            </div>
            """, unsafe_allow_html=True)

            st.metric(
                "Número de frames",
                mfcc.shape[1]
            )

        # ============================================
        # GRÁFICA MFCC
        # ============================================

        fig, ax = plt.subplots(figsize=(12,5))

        fig.patch.set_facecolor("#0f1117")
        ax.set_facecolor("#161b22")

        img = librosa.display.specshow(
            mfcc,
            x_axis='time',
            sr=fs,
            cmap='magma',
            ax=ax
        )

        ax.set_title(
            "Representación MFCC",
            color="white",
            fontsize=16
        )

        ax.tick_params(colors='white')

        for spine in ax.spines.values():
            spine.set_color("white")

        cbar = fig.colorbar(img)
        cbar.ax.yaxis.set_tick_params(color='white')

        plt.setp(
            plt.getp(cbar.ax.axes, 'yticklabels'),
            color='white'
        )

        st.pyplot(fig)

        st.write("")

        st.subheader("Interpretación")

        st.write("""
        Los coeficientes MFCC permiten representar características espectrales relacionadas con el timbre y la envolvente frecuencial de la voz.

        - Un mayor número de frames temporales proporciona una representación temporal más detallada.
        - Los coeficientes MFCC condensan información espectral relevante utilizada frecuentemente en análisis y reconocimiento de voz.
        """)

    # ============================================
    # TAB LPC
    # ============================================

    with tab6:

        st.subheader("Análisis LPC")

        st.write("""
        El análisis LPC permite modelar la envolvente espectral de la señal y estimar los principales formantes asociados al tracto vocal.
        """)

        # ============================================
        # SELECCIÓN DE SEGMENTO
        # ============================================

        duracion = len(y) / fs

        inicio_seg = st.slider(

            "Inicio del segmento (s)",

            min_value=0.0,

            max_value=max(0.1, duracion - 0.1),

            value=min(1.0, duracion/2),

            step=0.05
        )

        # Conversión a muestras
        inicio_muestra = int(inicio_seg * fs)

        # Segmento corto
        segmento = y[
            inicio_muestra:inicio_muestra + 2048
        ]

        # Ventana Hamming
        segmento = segmento * np.hamming(len(segmento))

        # Preénfasis
        segmento = librosa.effects.preemphasis(segmento)

        # ============================================
        # LPC
        # ============================================

        freqs_lpc, lpc_env, formantes = formantes_lpc(
            segmento,
            fs,
            P=10
        )

        # ============================================
        # FFT REAL
        # ============================================

        X = np.abs(
            np.fft.rfft(
                segmento,
                n=4096
            )
        )

        freqs_fft = np.linspace(
            0,
            fs/2,
            len(X)
        )

        # ============================================
        # MÉTRICAS
        # ============================================

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "F1",
                f"{formantes[0]:.0f} Hz"
                if len(formantes) > 0 else "-"
            )

        with col2:
            st.metric(
                "F2",
                f"{formantes[1]:.0f} Hz"
                if len(formantes) > 1 else "-"
            )

        with col3:
            st.metric(
                "F3",
                f"{formantes[2]:.0f} Hz"
                if len(formantes) > 2 else "-"
            )

        st.write("")

        # ============================================
        # GRÁFICA LPC
        # ============================================

        fig, ax = plt.subplots(figsize=(12,5))

        fig.patch.set_facecolor("#0f1117")
        ax.set_facecolor("#161b22")

        # Espectro real
        ax.plot(
            freqs_fft,
            20*np.log10(X + 1e-10),
            alpha=0.4,
            color="white",
            linewidth=1,
            label="Espectro"
        )

        # LPC
        ax.plot(
            freqs_lpc,
            lpc_env,
            color="#00d4ff",
            linewidth=2,
            label="Envolvente LPC"
        )

        # Formantes
        for f in formantes:

            ax.axvline(
                f,
                color="red",
                linestyle="--",
                alpha=0.7
            )

        ax.set_xlim(0, 5000)

        ax.set_title(
            "Envolvente espectral LPC",
            color="white",
            fontsize=16
        )

        ax.set_xlabel(
            "Frecuencia (Hz)",
            color="white"
        )

        ax.set_ylabel(
            "Magnitud (dB)",
            color="white"
        )

        ax.tick_params(colors='white')

        for spine in ax.spines.values():
            spine.set_color("white")

        ax.grid(
            color='white',
            alpha=0.1,
            linestyle='--'
        )

        ax.legend()

        st.pyplot(fig)

    
