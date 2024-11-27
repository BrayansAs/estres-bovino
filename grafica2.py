import pandas as pd
import numpy as np
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox

class StressBovineAnalysis:
    def __init__(self):
        self.cow_id = None
        self.root = tk.Tk()

    def llamar_datos(self, cow_id):
        data= pd.read_csv(f'Datos_actividad/{cow_id}_act.csv')
        data['datetime'] = pd.to_datetime(data['Fecha'] + ' ' + data['Hora'].astype(str) + ':00:00')
        data = data.set_index('datetime')
        data = data.sort_index()
        return data

    def extract_circadian_cycle(self, data, sample_rate=24, z=1):
        fft_result = fft(data)
        n = len(data)
        freq = np.fft.fftfreq(n, d=1/sample_rate)
        
        fft_filtered = np.zeros_like(fft_result, dtype=complex)
        for k in range(-z, z + 1):
            idx = np.argmin(np.abs(freq - k))
            fft_filtered[idx] = fft_result[idx]
        
        return np.real(ifft(fft_filtered))

    def filter_data(self, df, start_date, start_hour):
        end_date = pd.to_datetime(start_date) + pd.Timedelta(hours=36)

        filtered_df = df[(df.index >= pd.to_datetime(start_date).replace(hour=start_hour)) &
                        (df.index < end_date)]

        print(filtered_df)
        return filtered_df

    def process_activity(self, df, activity_column):
        window_size = 36
        step_size = 1

        series_results = []
        for start_idx in range(0, len(df) - window_size + 1, step_size):
            window_data = df.iloc[start_idx:start_idx + window_size]
            if len(window_data) == window_size:
                subseries_A = window_data.iloc[:24]
                subseries_B = window_data.iloc[12:36]

                cycle_A = self.extract_circadian_cycle(subseries_A[activity_column].values)
                cycle_B = self.extract_circadian_cycle(subseries_B[activity_column].values)

                differences = [(cycle_A[i] - cycle_B[i - 12])**2 for i in range(12, 24)]
                dist_euclidiana = np.sqrt(sum(differences))

                series_results.append({
                    'Fecha': window_data.index[12].date(),
                    'Hora': window_data.index[12].hour,
                    'distancia': dist_euclidiana
                })

                # Create a figure and a self.canvas for the plot
            
                # Create a blank self.canvas
                self.canvas_frame = ttk.Frame(self.root)
                self.canvas_frame.grid(row=4, column=0, columnspan=2)
                self.canvas = FigureCanvasTkAgg(Figure(figsize=(18, 6)), master=self.canvas_frame)
                self.canvas.get_tk_widget().pack()

                # Plot the graph on the self.canvas
                ax = self.canvas.figure.add_subplot(111)
                hours = range(36)
                ax.plot(hours, df['nivel_actividad'].values, label="Nivel de actividad (Original)", linestyle='-', alpha=0.8)
                ax.plot(range(24), self.extract_circadian_cycle(df['nivel_actividad'].iloc[:24].values), 'k--', label="Ciclo A")
                ax.plot(range(12, 36), self.extract_circadian_cycle(df['nivel_actividad'].iloc[12:].values), 'k--', label="Ciclo B")
                ax.set_title(f"Serie de 36 horas con ciclos circadianos A y B (Vaca {self.cow_id})")
                ax.set_xlabel("Horas")
                ax.set_ylabel("Nivel de actividad")
                ax.legend()
                ax.grid(True)

                self.root.update()  # Update the interface to display the self.canvas

               

                return pd.DataFrame(series_results)

    def launch_filter_interface(self):
        
        self.root.title("Análisis de estrés en vacas")

        # Listado desplegable de vacas
        cow_ids = [1, 2, 3, 4, 468, 479, 4003, 4151, 4160, 4173]
        cow_id_var = tk.StringVar()
        cow_id_var.set(cow_ids[0])
        cow_id_label = ttk.Label(self.root, text="Vaca:")
        cow_id_combobox = ttk.Combobox(self.root, textvariable=cow_id_var, values=cow_ids)

        # Cuadro para seleccionar la fecha
        start_date_var = tk.StringVar()
        start_date_label = ttk.Label(self.root, text="Fecha de inicio (YYYY-MM-DD):")
        start_date_entry = ttk.Entry(self.root, textvariable=start_date_var)

        # Cuadro para seleccionar la hora
        start_hour_var = tk.StringVar()
        start_hour_label = ttk.Label(self.root, text="Hora de inicio (0-23):")
        start_hour_entry = ttk.Entry(self.root, textvariable=start_hour_var)
        
        # Botón de reiniciar
        reset_button = ttk.Button(self.root, text="Reiniciar", command=self.reset_filter)

        # Función para procesar los datos y mostrar la gráfica
        def process_and_show_graph():
            self.cow_id = int(cow_id_var.get())
            start_date = start_date_var.get()
            start_hour = int(start_hour_var.get())

            df = self.llamar_datos(self.cow_id)
            filtered_df = self.filter_data(df, start_date, start_hour)

            if not filtered_df.empty:
                self.process_activity(filtered_df, 'nivel_actividad')
                messagebox.showinfo("Resultados", f"Datos procesados para la vaca {self.cow_id}.")
            else:
                messagebox.showwarning("Sin Datos", f"No se encontraron datos para la vaca {self.cow_id} a partir de la fecha y hora {start_date} {start_hour}.")

        # Botón para procesar los datos y mostrar la gráfica
        process_button = ttk.Button(self.root, text="Procesar", command=process_and_show_graph)

        # Colocar los widgets en la interfaz
        cow_id_label.grid(row=0, column=0, sticky="e")
        cow_id_combobox.grid(row=0, column=1, sticky="w")
        start_date_label.grid(row=1, column=0, sticky="e")
        start_date_entry.grid(row=1, column=1, sticky="w")
        start_hour_label.grid(row=2, column=0, sticky="e")
        start_hour_entry.grid(row=2, column=1, sticky="w")
        reset_button.grid(row=3, column=0, sticky="e")
        process_button.grid(row=3, column=1, sticky="w")

        self.root.mainloop()

    def reset_filter(self):
        self.cow_id = None
        self.launch_filter_interface()

# Lanzar la interfaz de usuario
if __name__ == "__main__":
    analysis = StressBovineAnalysis()
    analysis.launch_filter_interface()