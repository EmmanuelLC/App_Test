import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import pi
from scipy import special
import json
import os

PARAMS_FILE = "params.json"

# Fonctions de traitement
def SignalFunction(T, nbpt, t0, tf, td, deltaeps, A, sigma, D):
    Deltat = (tf - t0) / nbpt
    t = np.arange(t0 + deltaeps, tf + deltaeps - Deltat, Deltat)
    f = 1 / T
    w = 2 * pi * f
    s = np.sin(w * (t - td))
    Env = A * np.exp(-(-w * (t - td))**2 / (2 * sigma**2)) + D
    signaltemp = Env * s
    return t, signaltemp, Env, Deltat

def SpectreFunction(signale, deltat, decalage):
    nfft = 2 ** 18
    spectre = np.fft.fft(signale, nfft)
    freq = np.fft.fftfreq(nfft, d=deltat)
    intervallatt = np.arange(0, len(freq) // 2)
    intervalerf = -(intervallatt / intervallatt.max() - decalage)
    intervalerf /= intervalerf.max()
    facteur_dilatation = 2
    intervalerf *= facteur_dilatation
    att_curve = (special.erf(intervalerf) + 1) / np.max(special.erf(intervalerf) + 1)
    attenuation = np.ones(len(freq))
    attenuation[0:len(freq)//2] *= att_curve
    attenuation[len(freq):len(freq)//2 -1:-1] *= att_curve
    spectre_att = spectre * attenuation
    signal_reconstruit = np.fft.ifft(spectre_att, nfft)
    time = np.arange(0, len(signal_reconstruit)*deltat, deltat)
    return spectre, freq, att_curve, spectre_att, signal_reconstruit, time

# Chargement des paramètres
def load_params():
    if os.path.exists(PARAMS_FILE):
        with open(PARAMS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_params(params):
    with open(PARAMS_FILE, "w") as f:
        json.dump(params, f)

# Interface Streamlit
st.set_page_config(layout="wide", page_title="Analyse du Signal")
st.title("Analyse de Signal Synthétique + CSV")

params = load_params()

with st.sidebar:
    st.header("Paramètres")
    
    uploaded_file = st.file_uploader("Fichier CSV", type=["csv"])
    
    st.subheader("Signal de base")
    A = st.number_input("Amplitude (volts)", value=float(params.get("Amplitude (volts)", 1.0)))
    freq = st.number_input("Fréquence (MHz)", value=float(params.get("Fréquence (MHz)", 1.0)))
    sigma = st.number_input("Sigma", value=float(params.get("Sigma", 1.0)))
    td = st.number_input("Temps amplitude maximale (µs)", value=float(params.get("Temps amplitude maximale (µs)", 5.0)))
    D = st.number_input("D (volts)", value=float(params.get("D (volts)", 0.0)))

    st.subheader("Échos")
    nbsignaux = st.number_input("Nombre de signaux", min_value=1, value=int(params.get("Nombre de signaux", 1)))
    tmpsvol = st.number_input("Temps de vol (µs)", value=float(params.get("Temps de vol (µs)", 5.0)))
    facd = st.number_input("Diminution de l amplitude", value=float(params.get("Diminution de l amplitude", 0.5)))

    st.subheader("Temps")
    nbpt = st.number_input("Nombre de points", value=int(params.get("Nombre de points", 1024)))
    t0 = st.number_input("Temps initial (µs)", value=float(params.get("Temps initial (µs)", 0.0)))
    tf = st.number_input("Temps final (µs)", value=float(params.get("Temps final (µs)", 20.0)))
    deltaeps = st.number_input("Deltaeps (µs)", value=float(params.get("Deltaeps (µs)", 0.0)))

    st.subheader("Spectre")
    decalage = st.number_input("Décalage", value=float(params.get("Décalage", 0.0)))
    fminaff = st.number_input("Fréquence minimum (MHz)", value=float(params.get("Fréquence minimum (MHz)", 0.0)))
    fmaxaff = st.number_input("Fréquence maximum (MHz)", value=float(params.get("Fréquence maximum (MHz)", 5.0)))
    normalize = st.checkbox("Normaliser", value=True)

    run_button = st.button("Générer les signaux")

if run_button:
    try:
        T = 1 / (freq * 1e6)
        td *= 1e-6
        t0 *= 1e-6
        tf *= 1e-6
        deltaeps *= 1e-6
        tmpsvol *= 1e-6

        # Création des signaux
        signals = []
        for i in range(nbsignaux):
            t, s, env, deltat = SignalFunction(T, nbpt, t0, tf, td + i * tmpsvol, deltaeps, A * (facd ** i), sigma, D)
            signals.append((t, s, env))

        # Spectre du premier signal
        spectre, freq_arr, att_curve, spectre_att, signal_reconstruit, time = SpectreFunction(signals[0][1], deltat, decalage)

        # Tracé du signal
        fig1, ax1 = plt.subplots()
        for t, s, env in signals:
            ax1.plot(t * 1e6, s, '--', label="Signal créé")
            ax1.plot(t * 1e6, env, ':', label="Enveloppe")
        if normalize:
            ax1.plot(time[:len(t)] * 1e6, signal_reconstruit.real[:len(t)] / np.max(np.abs(signals[0][1])), '--', label="Signal reconstruit (IFFT)", color="magenta")
        else:
            ax1.plot(time * 1e6, signal_reconstruit.real, '--', label="Signal reconstruit (IFFT)", color="magenta")
        ax1.set_xlabel("Temps (µs)")
        ax1.set_ylabel("Amplitude (V)")
        ax1.grid(True)
        ax1.set_title("Signaux simulés")
        ax1.legend()
        st.pyplot(fig1)

        # Tracé du spectre
        fig2, ax2 = plt.subplots()
        ax2.plot(freq_arr / 1e6, np.abs(spectre) / (np.max(np.abs(spectre)) if normalize else 1), label="Spectre")
        ax2.plot(freq_arr / 1e6, np.abs(spectre_att) / (np.max(np.abs(spectre_att)) if normalize else 1), label="Spectre atténué")
        ax2.plot(freq_arr[:len(att_curve)] / 1e6, att_curve / (np.max(att_curve) if normalize else 1), label="Atténuation")
        ax2.set_xlim(fminaff, fmaxaff)
        ax2.set_xlabel("Fréquence (MHz)")
        ax2.set_ylabel("Amplitude (A.U.)")
        ax2.set_title("Spectre")
        ax2.grid(True)
        ax2.legend()
        st.pyplot(fig2)

        # Affichage du CSV si fourni
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file, sep=';')
            fig3, ax3 = plt.subplots()
            ax3.plot(df['Time (seconds)'] * 1e6, df['Voltage (volts)'], label="Données CSV", color='red')
            ax3.set_title("Données du fichier CSV")
            ax3.set_xlabel("Temps (µs)")
            ax3.set_ylabel("Tension (V)")
            ax3.grid(True)
            ax3.legend()
            st.pyplot(fig3)

        # Sauvegarde des paramètres
        save_params({k: str(v) for k, v in {
            'Amplitude (volts)': A,
            'Fréquence (MHz)': freq,
            'Sigma': sigma,
            'Temps amplitude maximale (µs)': td * 1e6,
            'D (volts)': D,
            'Nombre de signaux': nbsignaux,
            'Temps de vol (µs)': tmpsvol * 1e6,
            'Diminution de l amplitude': facd,
            'Nombre de points': nbpt,
            'Temps initial (µs)': t0 * 1e6,
            'Temps final (µs)': tf * 1e6,
            'Deltaeps (µs)': deltaeps * 1e6,
            'Décalage': decalage,
            'Fréquence minimum (MHz)': fminaff,
            'Fréquence maximum (MHz)': fmaxaff,
        }.items()})

    except Exception as e:
        st.error(f"Erreur : {e}")