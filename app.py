from flask import Flask, request, jsonify
from flask_cors import CORS
from sentinelhub import SHConfig
import base64
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.relativedelta import relativedelta
from PIL import Image
from io import BytesIO
from sentinelhub import (
    CRS,
    BBox,
    DataCollection,
    MimeType,
    SentinelHubRequest,
    bbox_to_dimensions,
)
from threading import Thread
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

config = SHConfig()

app = Flask(__name__)
CORS(app)  # Permitir solicitudes entre dominios (CORS)

@app.route('/receive-data', methods=['POST'])
def receive_data():
# Extrae el valor de 'opc' antes de continuar
  opc = data.get('opc')



def opcion_1():       
            # Define la función get_coordinates que toma un argumento 'data'
            def get_coordinates(data):
                # Devuelve la lista de coordenadas del diccionario 'data', o una lista vacía si no se encuentran
                return data.get('coordenadas', [])

            # Define la función fetch_and_process_image que toma un argumento 'coordenadas'
            def fetch_and_process_image(coordenadas):
                # Crea una tupla de área de interés (AOI) usando las coordenadas proporcionadas
                aoi = (coordenadas[0], coordenadas[1], coordenadas[2], coordenadas[3])
                resolution = 60  # Define la resolución para las imágenes a descargar
                # Crea un objeto BBox que representa el área de interés con las coordenadas y el sistema de referencia CRS WGS84
                aoi_bbox = BBox(bbox=aoi, crs=CRS.WGS84)
                # Calcula el tamaño de la AOI en píxeles dado el BBox y la resolución
                aoi_size = bbox_to_dimensions(aoi_bbox, resolution=resolution)

                # Define las fechas de inicio y fin para la descarga de imágenes
                fecha_inicio = '2024-01-01'  # Cambia estas fechas según sea necesario
                fecha_fin = '2024-12-31'
                # Convierte las fechas de inicio y fin de string a objetos datetime
                fecha_inicio_dt = datetime.strptime(fecha_inicio, '%Y-%m-%d')
                fecha_fin_dt = datetime.strptime(fecha_fin, '%Y-%m-%d')
                fechas = []  # Lista para almacenar las fechas a procesar
                # Genera una lista de fechas mensuales entre fecha_inicio y fecha_fin
                while fecha_inicio_dt <= fecha_fin_dt:
                    fechas.append(fecha_inicio_dt.strftime('%Y-%m-%d'))  # Añade la fecha actual a la lista
                    fecha_inicio_dt += relativedelta(months=1)  # Avanza un mes

                # Inicializa una lista para almacenar las imágenes NDWI descargadas
                images_ndwi = []
                # Itera sobre cada fecha para descargar las imágenes NDWI
                for date in fechas:
                    try:
                        # Crea una solicitud a Sentinel Hub para calcular NDWI
                        request = SentinelHubRequest(
                            evalscript="""  # Script para calcular NDWI
                            // NDWI calculation with cloud masking
                            function setup() {
                                return {
                                    input: ["B03", "B08", "dataMask"],  # Bandas a usar
                                    output: { bands: 3 }  # Número de bandas de salida
                                };
                            }

                            const ramp = [  # Define un mapa de colores para la visualización
                                [-0.8, 0x008000],  // Verde
                                [0, 0xFFFFFF],     // Blanco
                                [0.8, 0x0000CC]    // Azul
                            ];

                            let viz = new ColorRampVisualizer(ramp);  // Crea un visualizador de colores

                            function evaluatePixel(samples) {
                                // Verificar si hay nubes usando la máscara de datos
                                if (samples.dataMask === 0) {
                                    return [0, 0, 0];  // Devuelve negro para píxeles enmascarados (nubes)
                                }
                                // Calcula el NDWI
                                const val = (samples.B03 - samples.B08) / (samples.B03 + samples.B08);
                                let imgVals = viz.process(val);  // Procesa el valor usando el visualizador
                                return imgVals.concat(samples.dataMask);  // Devuelve los valores de imagen y la máscara de datos
                            }
                            """,
                            input_data=[  # Define la entrada de datos para la solicitud
                                SentinelHubRequest.input_data(
                                    data_collection=DataCollection.SENTINEL2_L2A,  # Colección de datos de Sentinel-2 L2A
                                    time_interval=(date, date)  # Intervalo de tiempo para la solicitud
                                )
                            ],
                            responses=[  # Define las respuestas esperadas
                                SentinelHubRequest.output_response('default', MimeType.TIFF)  # Cambiar a TIFF
                            ],
                            bbox=aoi_bbox,  # Establece el área de interés
                            size=aoi_size,  # Establece el tamaño de la imagen
                            config=config  # Configuración de Sentinel Hub
                        )

                        # Realiza la solicitud y obtiene los datos
                        response = request.get_data()
                        if response:
                            print(f"Imagen NDWI descargada para la fecha: {date}")  # Mensaje de éxito
                            images_ndwi.append(response[0])  # Añade la imagen a la lista
                        else:
                            print(f"No se encontraron datos NDWI para la fecha {date}.")  # Mensaje si no se encuentran datos
                    except Exception as e:
                        print(f"Error al procesar la fecha {date}: {e}")  # Manejo de errores

                # Verifica si se han descargado imágenes NDWI
                if images_ndwi:
                    pil_images_ndwi = []  # Lista para almacenar imágenes PIL
                    for img in images_ndwi:
                        # Convierte cada imagen de TIFF a un objeto PIL
                        pil_image = Image.fromarray(img)
                        pil_images_ndwi.append(pil_image)  # Añade la imagen convertida a la lista

                    # Guarda las imágenes como un GIF
                    pil_images_ndwi[0].save('sentinel_ndwi_timelapse.gif', save_all=True, append_images=pil_images_ndwi[1:], duration=500, loop=0)
                    print("GIF NDWI creado con éxito.")  # Mensaje de éxito

                    frames = []  # Lista para almacenar los fotogramas del GIF

                    # Carga el GIF en OpenCV
                    cap = cv2.VideoCapture('sentinel_ndwi_timelapse.gif')

                    while True:
                        # Lee cada fotograma del GIF
                        success, frame = cap.read()
                        if not success:
                            break  # Salir si no hay más fotogramas

                        # Convierte el fotograma de BGR a RGB (OpenCV usa BGR por defecto)
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        # Convierte el fotograma a un objeto PIL Image
                        pil_image = Image.fromarray(frame_rgb)

                        # Añade el fotograma a la lista
                        frames.append(pil_image)

                    cap.release()  # Libera el objeto de captura

                    gif_bytes = BytesIO()  # Crea un objeto BytesIO para guardar el GIF en memoria
                    # Guarda el GIF en el objeto BytesIO
                    frames[0].save(gif_bytes, format='GIF', save_all=True, append_images=frames[1:], duration=100, loop=0)
                    gif_bytes.seek(0)  # Vuelve al inicio del BytesIO

                    # Codifica el GIF en base64 para el envío
                    img_base64 = base64.b64encode(gif_bytes.read()).decode('utf-8')

                    return img_base64  # Devuelve la imagen codificada en base64
                else:
                    print("No se descargaron imágenes NDWI.")  # Mensaje si no hay imágenes descargadas
                    return None  # Devuelve None si no se descargaron imágenes

            # Define la ruta para recibir datos en la aplicación web
            @app.route('/receive-data', methods=['POST'])
            def receive_data():
                # Obtiene los datos JSON enviados en la solicitud
                data = request.get_json()

                # Obtiene las coordenadas del JSON usando la función definida anteriormente
                coordenadas = get_coordinates(data)

                # Procesa la imagen usando las coordenadas y obtiene la imagen en base64
                img_base64 = fetch_and_process_image(coordenadas)

                # Devuelve la imagen procesada en formato JSON, o un mensaje de error si no se procesó ninguna imagen
                return jsonify({"image": img_base64}) if img_base64 else jsonify({"error": "No images processed."})

            # Inicia la aplicación en modo de depuración
            if __name__ == '__main__':
                app.run(debug=True)  # Ejecuta la aplicación Flask
            return "nvdi pruebas pp"  # Devuelve un mensaje (no se ejecuta en el contexto de Flask)
def opcion_2():
        # Función para procesar la imagen de NDVI y detectar áreas en azul
        def procesar_ndvi(ndvi_image, fondo_verde_copy):        
            gray_lower = np.array([0, 0, 50], np.uint8)
            gray_upper = np.array([180, 50, 200], np.uint8)

            hsv_frame = cv2.cvtColor(ndvi_image, cv2.COLOR_BGR2HSV)
            gray_mask = cv2.inRange(hsv_frame, gray_lower, gray_upper)
            contours, _ = cv2.findContours(gray_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 10:  # Ajustar el umbral si es necesario
                    cv2.drawContours(fondo_verde_copy, [contour], -1, (255, 0, 0), 2)  # Azul

        # Función para procesar la imagen de NDWI y detectar áreas en azul
        def procesar_ndwi(ndwi_image, fondo_verde_copy):
            blue_lower = np.array([90, 50, 50], np.uint8)
            blue_upper = np.array([130, 255, 255], np.uint8)

            hsv_frame = cv2.cvtColor(ndwi_image, cv2.COLOR_BGR2HSV)
            blue_mask = cv2.inRange(hsv_frame, blue_lower, blue_upper)
            contours, _ = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 50:  # Ajustar el umbral si es necesario
                    cv2.drawContours(fondo_verde_copy, [contour], -1, (255, 0, 0), 2)  # Azul

        # Función para procesar el GIF de NDVI y detectar áreas en rosa
        def procesar_gif_ndvi(cap, fondo_verde_copy, rosa_mask):
            gray_lower = np.array([0, 0, 50], np.uint8)
            gray_upper = np.array([180, 50, 200], np.uint8)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                gray_mask = cv2.inRange(hsv_frame, gray_lower, gray_upper)
                contours, _ = cv2.findContours(gray_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 50:
                        cv2.drawContours(rosa_mask, [contour], -1, (255, 105, 180), -1)  # Rosa en la máscara

        # Función para procesar el GIF de NDWI y detectar áreas en rosa
        def procesar_gif_ndwi(cap, fondo_verde_copy, rosa_mask):
            blue_lower = np.array([90, 50, 50], np.uint8)
            blue_upper = np.array([130, 255, 255], np.uint8)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                blue_mask = cv2.inRange(hsv_frame, blue_lower, blue_upper)
                contours, _ = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 50:
                        cv2.drawContours(rosa_mask, [contour], -1, (255, 105, 180), -1)  # Rosa en la máscara

        # Cargar la imagen de fondo
        fondo_verde = cv2.imread('./CodigosIndependientes/ensenada/2024-04-01-00_00_2024-04-01-23_59_Sentinel-2_Quarterly_Mosaics_True_Color_Cloudless.jpg')

        # Cargar las imágenes de NDVI y NDWI
        ndvi_image = cv2.imread('./ensenada/Sentinel-2_L1C-917111078020849-timelapse-0042.jpg')
        ndwi_image = cv2.imread("./ensenada/Sentinel-2_L1C-1466683088009321-timelapse-0042.jpg")

        # Obtener dimensiones de las imágenes de NDVI y NDWI
        ndvi_height, ndvi_width = ndvi_image.shape[:2]
        ndwi_height, ndwi_width = ndwi_image.shape[:2]

        # Asegurarse de que las dimensiones coincidan
        if ndvi_height != ndwi_height or ndvi_width != ndwi_width:
            print("Las imágenes de NDVI y NDWI deben tener las mismas dimensiones.")
            exit()

        # Redimensionar la imagen de fondo al tamaño de las imágenes de NDVI y NDWI
        fondo_verde_copy = cv2.resize(fondo_verde, (ndvi_width, ndvi_height))

        # Procesar las imágenes de NDVI y NDWI primero (dibuja en azul)
        procesar_ndvi(ndvi_image, fondo_verde_copy)
        procesar_ndwi(ndwi_image, fondo_verde_copy)

        # Cargar los GIFs
        cap_ndvi = cv2.VideoCapture("./ensenada/Sentinel-2_L1C-917111078020849-timelapse.gif")
        cap_ndwi = cv2.VideoCapture("./ensenada/Sentinel-2_L1C-1557339245669505-timelapse.gif")

        # Crear una máscara rosa vacía
        rosa_mask = np.zeros_like(fondo_verde_copy)

        # Crear threads para procesar los GIFs
        thread_ndvi = Thread(target=procesar_gif_ndvi, args=(cap_ndvi, fondo_verde_copy, rosa_mask))
        thread_ndwi = Thread(target=procesar_gif_ndwi, args=(cap_ndwi, fondo_verde_copy, rosa_mask))

        # Iniciar los threads
        thread_ndvi.start()
        thread_ndwi.start()

        # Mostrar la imagen de fondo con los resultados de NDVI y NDWI
        while True:
            # Hacer una copia de fondo para mostrar
            imagen_a_mostrar = fondo_verde_copy.copy()

            # Aplicar el color rosa solo donde haya contornos detectados
            rosa_transparente = cv2.addWeighted(imagen_a_mostrar, 0.75, rosa_mask, 0.25, 0)

            # Mostrar la imagen con los resultados
            cv2.imshow('Mapa con Resultados de NDVI y NDWI', rosa_transparente)

            # Espera a que el usuario presione 'q' para cerrar
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Detener los threads de procesamiento de imágenes
        thread_ndvi.join()
        thread_ndwi.join()

        # Cerrar todo
        cap_ndvi.release()
        cap_ndwi.release()
        cv2.destroyAllWindows()
def opcion_3():
        # 1. Cargar los archivos CSV
        vh_data = pd.read_csv("ensenada/Sentinel-1 IW VV+VH-IW-DV-VH-LINEAR-GAMMA0-ORTHORECTIFIED-2019-10-12T00_00_00.000Z-2024-10-12T23_59_59.999Z.csv")
        vv_data = pd.read_csv("ensenada/Sentinel-1 IW VV+VH-IW-DV-VV-DECIBEL-GAMMA0-ORTHORECTIFIED-2019-10-12T00_00_00.000Z-2024-10-12T23_59_59.999Z.csv")

        # 2. Convertir la columna de fechas al formato datetime
        vv_data['date'] = pd.to_datetime(vv_data['C0/date'])
        vh_data['date'] = pd.to_datetime(vh_data['C0/date'])

        # 3. Asegurarse de que ambas series temporales están ordenadas por fecha
        vv_data.sort_values('date', inplace=True)
        vh_data.sort_values('date', inplace=True)

        # 4. Establecer la columna de fechas como índice
        vv_data.set_index('date', inplace=True)
        vh_data.set_index('date', inplace=True)

        # 5. Interpolación de datos faltantes
        vv_data['C0/mean'] = vv_data['C0/mean'].interpolate(method='linear')
        vh_data['C0/mean'] = vh_data['C0/mean'].interpolate(method='linear')

        # 6. Normalización de los datos utilizando escaladores separados
        scaler_vv = MinMaxScaler(feature_range=(0, 1))
        scaler_vh = MinMaxScaler(feature_range=(0, 1))

        vv_scaled = scaler_vv.fit_transform(vv_data[['C0/mean']])
        vh_scaled = scaler_vh.fit_transform(vh_data[['C0/mean']])

        # 7. Crear secuencias temporales con mayor longitud
        def create_sequences(data, sequence_length):
            sequences = []
            for i in range(len(data) - sequence_length):
                sequences.append(data[i:i + sequence_length])
            return np.array(sequences)

        sequence_length = 30  # Aumentar la longitud de la secuencia

        # Crear secuencias para los datos VV y VH
        vv_sequences = create_sequences(vv_scaled, sequence_length)
        vh_sequences = create_sequences(vh_scaled, sequence_length)

        # Separar características y etiquetas
        X_vv = vv_sequences
        y_vv = vv_scaled[sequence_length:]  # Etiquetas desde el índice "sequence_length"

        X_vh = vh_sequences
        y_vh = vh_scaled[sequence_length:]

        # 8. Crear un modelo LSTM ajustado
        def create_lstm_model(input_shape):
            model = Sequential()
            model.add(LSTM(100, return_sequences=True, input_shape=input_shape))  # Más unidades LSTM
            model.add(LSTM(50, return_sequences=False))
            model.add(Dense(1))
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
            return model

        # Crear los modelos para VV y VH
        model_vv = create_lstm_model((X_vv.shape[1], X_vv.shape[2]))
        model_vh = create_lstm_model((X_vh.shape[1], X_vh.shape[2]))

        # 9. Entrenar los modelos con más épocas
        epochs = 100
        batch_size = 32
        history_vv = model_vv.fit(X_vv, y_vv, epochs=epochs, batch_size=batch_size, validation_split=0.2)
        history_vh = model_vh.fit(X_vh, y_vh, epochs=epochs, batch_size=batch_size, validation_split=0.2)

        # 10. Función para realizar predicciones futuras con menor cantidad de predicciones
        def predict_future(model, last_sequence, n_predictions):
            future_predictions = []
            current_sequence = last_sequence

            for _ in range(n_predictions):
                prediction = model.predict(current_sequence.reshape(1, sequence_length, 1))
                future_predictions.append(prediction[0, 0])
                # Actualizar la secuencia: eliminar el primer valor y añadir la predicción
                current_sequence = np.append(current_sequence[1:], prediction)
            
            return np.array(future_predictions)

        # 11. Predecir los próximos 6 valores (menos predicciones) para VV y VH
        n_future = 6
        last_sequence_vv = vv_scaled[-sequence_length:]
        last_sequence_vh = vh_scaled[-sequence_length:]

        vv_future_predictions = predict_future(model_vv, last_sequence_vv, n_future)
        vh_future_predictions = predict_future(model_vh, last_sequence_vh, n_future)

        # 12. Desnormalizar correctamente usando los escaladores respectivos
        vv_future_predictions_rescaled = scaler_vv.inverse_transform(vv_future_predictions.reshape(-1, 1))
        vh_future_predictions_rescaled = scaler_vh.inverse_transform(vh_future_predictions.reshape(-1, 1))

        # 13. Graficar los resultados de las predicciones
        plt.figure(figsize=(10, 6))

        # Graficar para VV
        plt.subplot(2, 1, 1)
        plt.plot(vv_data.index, vv_data['C0/mean'], label='VV gamma0 decibel (Histórico)', color='blue')
        future_dates_vv = pd.date_range(start=vv_data.index[-1], periods=n_future + 1, freq='M')[1:]
        plt.plot(future_dates_vv, vv_future_predictions_rescaled, label='Predicciones Futuras VV', color='red')
        plt.title('Predicciones LSTM para VV gamma0 decibel')
        plt.xlabel('Fecha')
        plt.ylabel('Retrodispersión (dB)')
        plt.legend()
        plt.grid(True)

        # Graficar para VH
        plt.subplot(2, 1, 2)
        plt.plot(vh_data.index, vh_data['C0/mean'], label='VH gamma0 decibel (Histórico)', color='green')
        future_dates_vh = pd.date_range(start=vh_data.index[-1], periods=n_future + 1, freq='M')[1:]
        plt.plot(future_dates_vh, vh_future_predictions_rescaled, label='Predicciones Futuras VH', color='red')
        plt.title('Predicciones LSTM para VH gamma0 decibel')
        plt.xlabel('Fecha')
        plt.ylabel('Retrodispersión (dB)')
        plt.legend()

        plt.tight_layout()
        plt.show()

def opcion_4():
   def detectar_calles(imagenn_path):
        # Cargar la imagen
        imagenn = cv2.imread(imagenn_path)
        if imagenn is None:
            print("Error: No se pudo cargar la imagen.")
            return
        
        # Convertir la imagen a escala de grises
        gris = cv2.cvtColor(imagenn, cv2.COLOR_BGR2GRAY)

        # Aplicar un desenfoque gaussiano para reducir el ruido
        gris = cv2.GaussianBlur(gris, (7, 7), 0)

        # Detectar bordes usando el algoritmo Canny
        bordes = cv2.Canny(gris, 50, 150)

        # Usar una operación de dilatación para engrosar los bordes detectados
        kernel = np.ones((5, 5), np.uint8)
        dilatacion = cv2.dilate(bordes, kernel, iterations=1)

        # Crear una máscara de la imagen original donde se dibujarán las calles detectadas
        mascara_calles = np.zeros_like(imagenn)

        # Copiar las áreas detectadas en la máscara
        mascara_calles[dilatacion != 0] = [0, 255, 0]  # Verde para las calles

        # Combinar la imagen original con la máscara de calles
        resultado = cv2.addWeighted(imagenn, 0.8, mascara_calles, 0.2, 0)

        # Mostrar solo la ventana con las calles detectadas
        cv2.imshow('Calles con posible peligro', resultado)

        # Esperar a que el usuario presione una tecla y cerrar las ventanas
        cv2.waitKey(0)
        cv2.destroyAllWindows()

     # Ruta de la imagen que deseas procesar
   ruta_imagen = 'ensenada/2024-04-01-00_00_2024-04-01-23_59_Sentinel-2_Quarterly_Mosaics_True_Color_Cloudless.jpg'
   detectar_calles(ruta_imagen)

    
# Crear un diccionario para simular el switch
opciones = {
    '1': opcion_1,
    '2': opcion_2,
    '3': opcion_3,
    '4': opcion_4
}

#Selección de opcion
opcion = '2'

# Ejecutar la opción seleccionada
if opcion in opciones:
    opciones[opcion]()  # Llamar a la función correspondiente
else:
    print("Opción no válida.")