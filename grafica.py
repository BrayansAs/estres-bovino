import pandas as pd
import numpy as np
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt

# Función para calcular el ciclo circadiano
def extract_circadian_cycle(data, sample_rate=24, z=1):
    fft_result = fft(data)
    n = len(data)
    freq = np.fft.fftfreq(n, d=1 / sample_rate)

    fft_filtered = np.zeros_like(fft_result, dtype=complex)
    for k in range(-z, z + 1):
        idx = np.argmin(np.abs(freq - k))
        fft_filtered[idx] = fft_result[idx]

    return np.real(ifft(fft_filtered))


# Función para procesar y graficar por bloques
def process_activity_by_blocks(df, activity_column, block_size=500, nombre_csv=""):
    total_points = len(df)
    blocks = range(0, total_points, block_size)  # Crear bloques de tamaño fijo
    fig, ax = plt.subplots(figsize=(18, 6))

    for i, start_idx in enumerate(blocks):
        end_idx = min(start_idx + block_size, total_points)
        block_data = df.iloc[start_idx:end_idx]

        # Crear ventanas dentro del bloque
        window_size = 36
        for start_window in range(0, len(block_data) - window_size + 1, 1):
            window_data = block_data.iloc[start_window:start_window + window_size]
            subseries_A = window_data.iloc[:24]
            subseries_B = window_data.iloc[12:36]

            cycle_A = extract_circadian_cycle(subseries_A[activity_column].values)
            cycle_B = extract_circadian_cycle(subseries_B[activity_column].values)

            hours = range(start_idx + start_window, start_idx + start_window + window_size)
            ax.plot(hours, window_data[activity_column].values, label=f"Nivel de actividad ({nombre_csv})", alpha=0.5)
            ax.plot(range(start_idx + start_window, start_idx + start_window + 24), cycle_A, 'k--', alpha=0.5)
            ax.plot(range(start_idx + start_window + 12, start_idx + start_window + 36), cycle_B, 'r--', alpha=0.5)

        # Configuración del gráfico
        ax.set_title(f"Actividad por bloques (Bloque {i + 1} de {nombre_csv})")
        ax.set_xlabel("Horas")
        ax.set_ylabel("Nivel de actividad")
        ax.legend()
        ax.grid(True)

        # Mostrar el bloque actual
        plt.show()
        ax.clear()


# Configuración de datos de entrada
nombres_csv = {1}  # IDs de los CSV
for nombre_csv in nombres_csv:
    df = pd.read_csv(f'Datos_actividad/{nombre_csv}_act.csv')
    df['datetime'] = pd.to_datetime(df['Fecha'] + ' ' + df['Hora'].astype(str) + ':00:00')
    df = df.set_index('datetime')
    df = df.sort_index()

    # Procesar y graficar por bloques
    process_activity_by_blocks(df, 'nivel_actividad', block_size=500, nombre_csv=nombre_csv)
