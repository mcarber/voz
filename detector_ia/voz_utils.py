from vozyaudio import *
import scipy.signal

def analizar_espectrogramas(y, fs, hop_length=512, n_fft=2048):
    
    # 1. STFT 
    S_l = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(np.abs(S_l), ref=np.max)

    # 2. Espectrograma lineal
    fig, ax = plt.subplots()
    img = librosa.display.specshow(S_db, sr=fs, hop_length=hop_length,
                                  x_axis='time', y_axis='linear', ax=ax)
    ax.set(title='Espectrograma (frecuencia lineal)')
    fig.colorbar(img, ax=ax, format="%+2.f dB")

    # 3. Espectrograma log
    fig, ax = plt.subplots()
    img = librosa.display.specshow(S_db, sr=fs, hop_length=hop_length,
                                  x_axis='time', y_axis='log', ax=ax)
    ax.set(title='Espectrograma (frecuencia logarítmica)')
    fig.colorbar(img, ax=ax, format="%+2.f dB")

    # 4. Espectrograma MEL
    S_mel = librosa.feature.melspectrogram(y=y.astype(float), sr=fs, hop_length=hop_length, n_fft=n_fft)
    S_mel_db = librosa.amplitude_to_db(S_mel, ref=np.max)

    fig, ax = plt.subplots()
    img = librosa.display.specshow(S_mel_db, sr=fs, hop_length=hop_length,
                                  x_axis='time', y_axis='mel', ax=ax)
    ax.set(title='Espectrograma en escala MEL')
    fig.colorbar(img, ax=ax, format="%+2.f dB")

    # 5. CQT 
    S_cqt = librosa.cqt(y=y.astype(float), sr=fs)
    S_cqt_db = librosa.amplitude_to_db(np.abs(S_cqt), ref=np.max)

    fig, ax = plt.subplots()
    img = librosa.display.specshow(S_cqt_db, sr=fs,
                                  x_axis='time', y_axis='cqt_note', ax=ax)
    ax.set(title='Energía por nota musical (CQT)')
    fig.colorbar(img, ax=ax, format="%+2.f dB")

    # 6. Cromagrama
    S_chroma = librosa.feature.chroma_cqt(y=y.astype(float), sr=fs)

    fig, ax = plt.subplots()
    img = librosa.display.specshow(S_chroma,
                                  y_axis='chroma', x_axis='time', ax=ax)
    ax.set(title='Cromagrama')
    fig.colorbar(img, ax=ax)

# ============================================
# ESPECTRO INDIVIDUAL
# ============================================

def obtener_espectrograma(y, fs, tipo="Lineal",
                           hop_length=512,
                           n_fft=2048):

    # ============================================
    # STFT BASE
    # ============================================

    S_l = librosa.stft(
        y,
        n_fft=n_fft,
        hop_length=hop_length
    )

    S_db = librosa.amplitude_to_db(
        np.abs(S_l),
        ref=np.max
    )

    # ============================================
    # LINEAL
    # ============================================

    if tipo == "Lineal":

        return S_db, "linear"

    # ============================================
    # LOG
    # ============================================

    elif tipo == "Log":

        return S_db, "log"

    # ============================================
    # MEL
    # ============================================

    elif tipo == "MEL":

        S_mel = librosa.feature.melspectrogram(
            y=y.astype(float),
            sr=fs,
            hop_length=hop_length,
            n_fft=n_fft
        )

        S_mel_db = librosa.power_to_db(
            S_mel,
            ref=np.max
        )

        return S_mel_db, "mel"

    # ============================================
    # CQT
    # ============================================

    elif tipo == "CQT":

        S_cqt = librosa.cqt(
            y=y.astype(float),
            sr=fs
        )

        S_cqt_db = librosa.amplitude_to_db(
            np.abs(S_cqt),
            ref=np.max
        )

        return S_cqt_db, "cqt"

    # ============================================
    # CROMAGRAMA
    # ============================================

    elif tipo == "Cromagrama":

        S_chroma = librosa.feature.chroma_cqt(
            y=y.astype(float),
            sr=fs
        )

        return S_chroma, "chroma"
    
    
def analizar_energia(y, fs, frame_length=1024, hop_length=512):
    """
    Calcula la energía de la señal mediante RMS.
    """
    
    # Energía RMS por frame
    rms = librosa.feature.rms(
        y=y,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]
    
    # Eje temporal
    times = librosa.frames_to_time(
        range(len(rms)),
        sr=fs,
        hop_length=hop_length
    )
    
    # Estadísticas
    energia_media = np.mean(rms)
    energia_std = np.std(rms)
    energia_max = np.max(rms)
    
    return rms, times, energia_media, energia_std, energia_max




def envolvente_amplitud(y, fs, alfa=0.99):
    
    # Rectificación de la señal
    d = np.abs(y)
    
    # Filtro exponencial (suavizado de la envolvente)
    b = [1 - alfa]
    a = [1, -alfa]
    
    # Cálculo de la envolvente
    env = signal.lfilter(b, a, d)
    
    # Eje temporal
    t = np.arange(len(y)) / fs
    
    return env, t


# ============================================
# LPC Y FORMANTES
# ============================================

def formantes_lpc(y, fs, P=10):

    # LPC
    a = librosa.lpc(y, order=P)

    # Respuesta frecuencia
    w, h = scipy.signal.freqz(
        1,
        a,
        worN=1024
    )

    # Raíces
    roots = np.roots(a)

    # Solo parte positiva
    roots = roots[np.imag(roots) >= 0]

    # Ángulos
    angz = np.arctan2(
        np.imag(roots),
        np.real(roots)
    )

    # Frecuencias
    freqs = angz * (fs / (2 * np.pi))

    # Bandwidths
    bandwidths = -0.5 * (
        fs / (2 * np.pi)
    ) * np.log(np.abs(roots))

    # Filtrar formantes válidos
    formantes = []

    for f, bw in zip(freqs, bandwidths):

        if f > 90 and f < 5000 and bw < 400:
            formantes.append(f)

    formantes = np.sort(formantes)

    return (
        w * fs / (2 * np.pi),
        20 * np.log10(np.abs(h) + 1e-10),
        formantes[:3]
    )



def analizar_pitch_voz(y, fs, fmin=75, fmax=450):
    """
    Estima la frecuencia fundamental (F0) usando pYIN.
    """
    
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y.astype(float),
        fmin=fmin,
        fmax=fmax,
        sr=fs
    )
    
    # Estadísticas básicas ignorando NaN
    stats = {
        'f0_serie': f0,
        'media': np.nanmean(f0),
        'mediana': np.nanmedian(f0),
        'std': np.nanstd(f0),
        'min': np.nanmin(f0),
        'max': np.nanmax(f0)
    }
    
    return stats

def extraer_caracteristicas_vocales(y, fs):

    # 1. Espectrograma MEL
    S_mel = librosa.feature.melspectrogram(
        y=y,
        sr=fs,
        n_mels=40,
        hop_length=512
    )

    S_mel_db = librosa.power_to_db(S_mel, ref=np.max)

    # 2. MFCCs
    mfccs = librosa.feature.mfcc(
        S=librosa.power_to_db(S_mel),
        n_mfcc=13
    )

    # 3. Delta MFCCs
    mfcc_delta = librosa.feature.delta(mfccs)

    # 4. Centroide espectral
    centroid = librosa.feature.spectral_centroid(y=y, sr=fs)[0]

    # 5. Roll-off
    rolloff = librosa.feature.spectral_rolloff(
        y=y,
        sr=fs,
        roll_percent=0.85
    )[0]

    return {
        "mfccs_mean": np.mean(mfccs, axis=1),
        "mfcc_delta_mean": np.mean(mfcc_delta, axis=1),
        "centroid_mean": np.mean(centroid),
        "rolloff_mean": np.mean(rolloff),
        "mel_spectrogram": S_mel_db
    }

from scipy.spatial.distance import euclidean

def comparar_caracteristicas(h1, h2):

    dist_mfcc = euclidean(
        h1["mfccs_mean"],
        h2["mfccs_mean"]
    )

    dist_delta = euclidean(
        h1["mfcc_delta_mean"],
        h2["mfcc_delta_mean"]
    )

    dist_centroid = abs(
        h1["centroid_mean"] - h2["centroid_mean"]
    )

    dist_rolloff = abs(
        h1["rolloff_mean"] - h2["rolloff_mean"]
    )

    distancia_total = (
        dist_mfcc
        + dist_delta
        + dist_centroid * 0.001
        + dist_rolloff * 0.001
    )

    return distancia_total

def analizar_bandas_fft(y, fs, num_bandas=8):
    
    # FFT
    X = np.fft.rfft(y)
    freqs_fft = np.fft.rfftfreq(len(y), 1/fs)
    power = np.abs(X)**2
    
    # bandas
    limites = np.linspace(0, fs/2, num_bandas + 1)
    
    energia_bandas = []
    
    for i in range(num_bandas):
        f1, f2 = limites[i], limites[i+1]
        
        mask = (freqs_fft >= f1) & (freqs_fft < f2)
        
        energia = np.sum(power[mask])
        energia_bandas.append(energia)
    
    return np.array(energia_bandas), limites


# ============================================
# MFCC en 3D
# Voz real vs voz IA
# ============================================

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


# ============================================
# FUNCIÓN 3D
# ============================================

def plot_mfcc_3d(mfcc, titulo):

    fig = plt.figure(figsize=(12,7))
    ax = fig.add_subplot(111, projection='3d')

    x = np.arange(mfcc.shape[1])
    y_mfcc = np.arange(mfcc.shape[0])

    X, Y = np.meshgrid(x, y_mfcc)
    Z = mfcc

    surf = ax.plot_surface(
        X,
        Y,
        Z,
        cmap='magma',
        linewidth=0,
        antialiased=True
    )

    ax.set_title(titulo, fontsize=15)
    ax.set_xlabel('Tiempo')
    ax.set_ylabel('Coeficiente MFCC')
    ax.set_zlabel('Amplitud')

    fig.colorbar(surf, shrink=0.6)

    return fig

def calcular_mfcc(y, fs):

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=fs,
        n_mfcc=13
    )

    return mfcc

# ============================================
# RESUMEN ACÚSTICO
# ============================================

def resumen_acustico(y, fs):

    # Pitch
    pitch = analizar_pitch_voz(y, fs)

    # Características espectrales
    features = extraer_caracteristicas_vocales(y, fs)

    return {

        "pitch_medio": pitch["media"],

        "variabilidad_pitch": pitch["std"],

        "centroid": features["centroid_mean"],

        "rolloff": features["rolloff_mean"]

    }