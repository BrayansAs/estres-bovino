import pandas as pd
import numpy as np
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button
import tkinter as tk
from tkinter import simpledialog, messagebox

# Función para calcular el ciclo circadiano utilizando la Transformada de Fourier
def extract_circadian_cycle(data, sample_rate=24, z=1):
    fft_result = fft(data)
    n = len(data)
    freq = np.fft.fftfreq(n, d=1/sample_rate)
    
    # Mantener solo los armónicos en el rango [-z, z]
    fft_filtered = np.zeros_like(fft_result, dtype=complex)
    for k in range(-z, z + 1):
        idx = np.argmin(np.abs(freq - k))
        fft_filtered[idx] = fft_result[idx]
    
    # Reconstrucción de la señal para obtener el ciclo circadiano
    return np.real(ifft(fft_filtered))

# Función para filtrar datos por vaca, rango de fechas y rango de horas
def filter_data(df, cow_id, start_date, end_date, start_hour, end_hour):
    filtered_df = df[(df['id_vaca'] == cow_id) & 
                     (df.index.date >= pd.to_datetime(start_date).date()) & 
                     (df.index.date <= pd.to_datetime(end_date).date()) &
                     (df.index.hour >= start_hour) & 
                     (df.index.hour <= end_hour)]
    return filtered_df

# Función para procesar una columna de actividad y calcular niveles de estrés
def process_activity(df, activity_column):
    window_size = 36
    step_size = 1

    series_results = []
    for start_idx in range(0, len(df) - window_size + 1, step_size):
        window_data = df.iloc[start_idx:start_idx + window_size]
        if len(window_data) == window_size:
            subseries_A = window_data.iloc[:24]
            subseries_B = window_data.iloc[12:36]
            
            cycle_A = extract_circadian_cycle(subseries_A[activity_column].values)
            cycle_B = extract_circadian_cycle(subseries_B[activity_column].values)
            
            differences = [(cycle_A[i] - cycle_B[i - 12])**2 for i in range(12, 24)]
            dist_euclidiana = np.sqrt(sum(differences))
            
            series_results.append({
                'Fecha': window_data.index[12].date(),
                'Hora': window_data.index[12].hour,
                'distancia': dist_euclidiana
            })

            plt.figure(figsize=(18, 6))
            hours = range(36)
            plt.plot(hours, window_data[activity_column].values, label="Nivel de actividad (Original)", linestyle='-', alpha=0.8)
            plt.plot(range(24), cycle_A, 'k--', label="Ciclo A")
            plt.plot(range(12, 36), cycle_B, 'k--', label="Ciclo B")
            plt.title("Serie de 36 horas con ciclos circadianos A y B")
            plt.xlabel("Horas")
            plt.ylabel("Nivel de actividad")
            plt.legend()
            plt.grid(True)
            plt.show()

    return pd.DataFrame(series_results)

# Interfaz gráfica para aplicar filtros
def launch_filter_interface():
    root = tk.Tk()
    root.withdraw()  # Oculta la ventana principal

    # Ventana emergente para recolectar parámetros
    try:
        cow_id = int(simpledialog.askstring("Filtro", "Ingrese el ID de la vaca:"))
        start_date = simpledialog.askstring("Filtro", "Ingrese la fecha de inicio (YYYY-MM-DD):")
        end_date = simpledialog.askstring("Filtro", "Ingrese la fecha de fin (YYYY-MM-DD):")
        start_hour = int(simpledialog.askstring("Filtro", "Ingrese la hora de inicio (0-23):"))
        end_hour = int(simpledialog.askstring("Filtro", "Ingrese la hora de fin (0-23):"))

        filtered_df = filter_data(df, cow_id, start_date, end_date, start_hour, end_hour)
        if not filtered_df.empty:
            stress_levels = process_activity(filtered_df, 'nivel_actividad')
            print(stress_levels)
            messagebox.showinfo("Resultados", f"Datos procesados para la vaca {cow_id}.")
        else:
            messagebox.showwarning("Sin Datos", f"No se encontraron datos para la vaca {cow_id} entre {start_date} y {end_date} en las horas {start_hour}-{end_hour}.")
    except Exception as e:
        messagebox.showerror("Error", f"Se produjo un error: {e}")

# Carga de datos
df = pd.read_csv('Datos_actividad/datos_vacas.csv')
df['datetime'] = pd.to_datetime(df['Fecha'] + ' ' + df['Hora'].astype(str) + ':00:00')
df = df.set_index('datetime')
df = df.sort_index()

# Lanzar la interfaz de usuario
launch_filter_interface()
