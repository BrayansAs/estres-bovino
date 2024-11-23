import pandas as pd
import numpy as np
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt

# Función para calcular el ciclo circadiano utilizando la Transformada de Fourier
def extract_circadian_cycle(data, sample_rate=24, z=1):
    fft_result = fft(data)
    n = len(data)
    freq = np.fft.fftfreq(n, d=1 / sample_rate)

    # Mantener solo los armónicos en el rango [-z, z]
    fft_filtered = np.zeros_like(fft_result, dtype=complex)
    for k in range(-z, z + 1):
        idx = np.argmin(np.abs(freq - k))
        fft_filtered[idx] = fft_result[idx]

    # Reconstrucción de la señal
    return np.real(ifft(fft_filtered))


# Función para procesar una columna de actividad y calcular niveles de estrés
def process_activity(df, activity_column, ax, nombre_csv):
    # Crear ventanas móviles de 36 horas con paso de 1 hora
    window_size = 36
    step_size = 1

    for start_idx in range(0, len(df) - window_size + 1, step_size):
        window_data = df.iloc[start_idx:start_idx + window_size]
        if len(window_data) == window_size:
            # Subseries A y B
            subseries_A = window_data.iloc[:24]
            subseries_B = window_data.iloc[12:36]

            # Extraer ciclos circadianos
            cycle_A = extract_circadian_cycle(subseries_A[activity_column].values)
            cycle_B = extract_circadian_cycle(subseries_B[activity_column].values)

            # Graficar en el mismo eje
            hours = range(start_idx, start_idx + window_size)
            ax.plot(hours, window_data[activity_column].values, label=f"Nivel de actividad ({nombre_csv})", linestyle='-', alpha=0.6)
            ax.plot(range(start_idx, start_idx + 24), cycle_A, 'k--', label=f"Ciclo A ({nombre_csv})")
            ax.plot(range(start_idx + 12, start_idx + 36), cycle_B, 'r--', label=f"Ciclo B ({nombre_csv})")


# Configuración de datos de entrada
nombres_csv = {1,}  # IDs de los CSV
fig, ax = plt.subplots(figsize=(18, 6))  # Crear figura y eje principal

for nombre_csv in nombres_csv:
    df = pd.read_csv(f'Datos_actividad/{nombre_csv}_act.csv')
    df['datetime'] = pd.to_datetime(df['Fecha'] + ' ' + df['Hora'].astype(str) + ':00:00')
    df = df.set_index('datetime')
    df = df.sort_index()

    # Procesar datos y graficar
    process_activity(df, 'nivel_actividad', ax, nombre_csv)

# Configuración final del gráfico
ax.set_title("Serie de actividad con ciclos circadianos acumulados")
ax.set_xlabel("Horas")
ax.set_ylabel("Nivel de actividad")
ax.legend()
ax.grid(True)
plt.show()
