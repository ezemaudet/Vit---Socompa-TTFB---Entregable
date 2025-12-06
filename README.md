# Segmentación de cuerpos de agua en Laguna Socompa con ViT/U-Net
Este repositorio contiene el código, documentación e insumos para el proyecto de **Visión por Computadora** cuyo objetivo es 
segmentar cuerpos de agua (agua / no agua) en la zona de **Laguna Socompa** utilizando imágenes satelitales **Landsat 5/8/9** y 
un modelo de segmentación tipo **U-Net / ViT** implementado en PyTorch.

## 1. Objetivo del proyecto
- Construir un **pipeline reproducible** para:
  - Descargar y procesar imágenes satelitales multitemporales desde **Google Earth Engine (GEE)**.
  - Sincronizar dichas imágenes con máscaras de referencia de agua (“teacher”).
  - Entrenar un modelo de segmentación (U-Net / ViT) para clasificar **agua vs no-agua**.
  - Evaluar el desempeño con métricas estándar de segmentación.
  - Realizar **inferencias interactivas** sobre cualquier bounding box geográfico usando una interfaz Gradio.

- Mostrar cómo este modelo puede utilizarse para:
  - Medir la **evolución temporal de la superficie de agua**.
  - Apoyar decisiones en contextos reales de **monitoreo ambiental** o **gestión hídrica**.

## 2. Arquitectura general

### 2.1. Flujo de datos

1. **Ingesta de datos (GEE)**  
   - Colecciones Landsat:
     - `LANDSAT/LT05/C02/T1_L2` (Landsat 5)  
     - `LANDSAT/LC08/C02/T1_L2` (Landsat 8)  
     - `LANDSAT/LC09/C02/T1_L2` (Landsat 9)  
   - Unificación de nombres de bandas a un esquema “L8-like”.
   - Recorte a un **ROI** que cubre Laguna Socompa.
   - Selección de bandas **SR_B5, SR_B4, SR_B3** (NIR, Rojo, Verde) para composiciones RGB.

2. **Construcción del cubo satelital (`a_m`)**  
   - Serie de tiempo multibanda en formato `xarray.DataArray`:
     - Dimensiones típicas: `(band, time, y, x)`.
   - Se genera un cubo mensual (por ejemplo, medianas mensuales).

3. **Máscaras de referencia (“teacher”) (`water_bin`)**  
   - Cubo de máscaras binarias agua/no-agua con la misma resolución espacial y temporal.

4. **Sincronización de cubos (`a_m_sync`, `water_sync`)**  
   - Uso de funciones como `sync_and_pick` para:
     - Encontrar períodos comunes entre cubos.
     - Seleccionar una imagen por mes y alinear temporalmente imágenes y máscaras.
   - Resultados:
     - `a_m_sync`: cubo de reflectancia alineado.
     - `water_sync`: cubo de máscaras alineado.
     - Lista de fechas sincronizadas.

5. **Normalización y dataset PyTorch**  
   - `compute_channel_stats_full(a_m_sync)`:
     - Calcula medias y desvíos estándar por banda (B5, B4, B3).
   - `RGBMonthDataset`:
     - Recibe `a_m_sync` y `water_sync`.
     - Extrae bandas 5-4-3, las reordena a formato `(C, H, W)`.
     - Aplica normalización `(x - mean) / std`.
     - Devuelve pares `(imagen, máscara)` listos para entrenamiento y validación.

6. **Modelo de segmentación (U-Net / ViT)**  
   - Implementación de `UNetSmall` / `ViTSeg2D`:
     - Entrada: imagen 3×H×W (NIR, Rojo, Verde).
     - Salida: logits 1×H×W (probabilidad de “agua”).
   - Pérdida:
     - Función `masked_bce_dice_loss` (BCE + Dice).

7. **Entrenamiento y validación**  
   - Loop de entrenamiento:
     - DataLoader de entrenamiento y validación.
     - Cálculo de loss en train y val por época.
     - Cálculo de métricas con `seg_metrics`.
   - Se registra:
     - `metrics_df` con las métricas por época.
     - `best_state` para almacenar el mejor modelo (menor `val_loss`).

8. **Evaluación y visualización**  
   - `seg_metrics(pred, true)`:
     - Calcula IoU, Dice, Precisión y Recall.
   - Funciones de visualización:
     - `infer_one(...)`: muestra RGB + máscara teacher + máscara student para una fecha.
     - `plot_student_panels(...)`: paneles de máscaras a través del tiempo.
     - `plot_rgb_student_pairs(...)`: pares RGB / máscara predicha.

9. **Cálculo de superficie de agua en el tiempo**  
   - Se alinean orientación y grilla de `student_all` (todas las predicciones) con el cubo original.
   - A partir de la resolución espacial se estima la **superficie de agua** por mes (m² o ha).

10. **Inferencia interactiva (Gradio + GEE)**  
    - `descargar_landsat_rgb(lon1, lat1, lon2, lat2)`:
      - Define un rectángulo a partir de coordenadas.
      - Descarga la mejor imagen disponible (menor nubosidad) en SR_B5, SR_B4, SR_B3.
    - `segmentar_area_bbox(coord_str)`:
      - Parsea `"lon1,lat1,lon2,lat2"`.
      - Normaliza la imagen con las mismas `means` y `stds` del entrenamiento.
      - Ejecuta el modelo entrenado y devuelve la máscara segmentada.
    - Interfaz Gradio (`gr.Interface`):
      - Entrada: string con bounding box.
      - Salida: imagem RGB + máscara de agua.

## 3. Estructura del repositorio

> **Nota:** el código se originó en un notebook de Colab y se consolidó en `rev1_vit_socompa_ttfb_entregable.py`.
>En la versión actual también se puede ejecutar directamente el script rev1_vit_socompa_ttfb_entregable.py
en un entorno tipo Jupyter/Colab siguiendo el orden de las celdas trasladadas al script.

> A medida que se modularice, se recomienda la siguiente estructura:

```text
.
├── src/
│   ├── data/
│   │   ├── gee_ingest.py          # Conexión GEE y construcción del cubo a_m / water_bin
│   │   └── sync_utils.py          # sync_and_pick, utilidades para alineación temporal
│   ├── models/
│   │   └── vit_unet.py            # Definición de UNetSmall / ViTSeg2D
│   ├── training/
│   │   ├── datasets.py            # RGBMonthDataset, compute_channel_stats_full
│   │   ├── losses.py              # masked_bce_dice_loss
│   │   └── train_vit_socompa.py   # Loop de entrenamiento/validación + guardado de métricas
│   ├── evaluation/
│   │   ├── metrics.py             # seg_metrics y helpers
│   │   └── visualization.py       # infer_one, plot_student_panels, plot_rgb_student_pairs, etc.
│   └── app/
│       └── gradio_interface.py    # descargar_landsat_rgb, segmentar_area_bbox, interfaz Gradio
│
├── docs/
│   ├── informe_tecnico.pdf        # Informe técnico de la materia
│   └── presentacion_final.pdf     # Slides para la presentación de 15'
│
├── data/                          # (opcional, o manejado vía .gitignore / DVC)
│   └── ...
│
├── rev1_vit_socompa_ttfb_entregable.py  # Script consolidado (versión Colab)
├── requirements.txt
└── README.md

## 4. Requisitos
4.1. Dependencias principales
Archivo sugerido requirements.txt:
torch
torchvision
numpy
pandas
matplotlib
scikit-image
xarray
rioxarray
xee
earthengine-api
geemap
scikit-learn
gradio
Pillow

4.2. Autenticación con Google Earth Engine
Antes de usar cualquier función que invoque a GEE se debe crear una cuenta en GEE:
import ee
ee.Authenticate()
ee.Initialize()

Nota: En entornos locales se requiere tener una cuenta de GEE y las credenciales configuradas.

## 5. Uso
5.1. Entrenamiento
Clonar el repositorio:

git clone <URL_DEL_REPO>
cd <NOMBRE_DEL_REPO>
Crear entorno (opcional) e instalar dependencias:
pip install -r requirements.txt
Ejecutar el script de entrenamiento (una vez modularizado):
python src/training/train_vit_socompa.py

Esto:
Construye/lee los cubos a_m y water_bin.
Genera a_m_sync y water_sync.
Calcula estadísticas de normalización.
Entrena el modelo, guarda:
models/vit_socompa_best.pth (mejor estado).
metrics/metrics_df.csv (métricas por época).



5.2. Evaluación y visualización
Luego de entrenar:
Usar las funciones:
seg_metrics para calcular IoU, Dice, Precisión, Recall.
infer_one para inspeccionar ejemplos individuales.
plot_student_panels / plot_rgb_student_pairs para generar figuras para el informe y la presentación.
Las figuras típicas que se generan son:
Comparaciones RGB / máscara teacher / máscara student.
Paneles de evolución temporal de la superficie de agua.
Curvas de loss y métricas vs época.

5.3. Inferencia interactiva con Gradio
Con el modelo entrenado cargado en memoria (por ejemplo, en gradio_interface.py):
python src/app/gradio_interface.py
En la interfaz, introducir un bounding box de la forma:
lon1,lat1,lon2,lat2

Ejemplo en la zona de Socompa:
-68.226,-24.506,-68.189,-24.539


La app:
Descarga la mejor imagen Landsat 5/8/9 disponible.
Normaliza con las estadísticas del entrenamiento.
Devuelve imagen RGB y máscara de agua predicha.

## 6. Evaluación: métricas de desempeño

Se utilizan métricas estándar de segmentación binaria:
IoU (Intersection over Union)
​Dice / F1 de segmentación
​Precisión (Precision)
Recall (Sensibilidad)

En el informe y la presentación se muestran:
Tablas con métricas finales en validación/test.
Gráficos de evolución de pérdida y métricas por época.
Ejemplos cualitativos donde el modelo acierta y donde falla (nubes, sombras, cambios abruptos del nivel de agua, etc.).

## 7. Resultados y ejemplos
Desempeño cuantitativo:
IoU, Dice, Precisión y Recall en el conjunto de validación.
Comparación entre diferentes configuraciones.
Resultados cualitativos:
Imágenes RGB con superposición de máscara teacher y máscara student.
Paneles temporales que muestran la evolución del cuerpo de agua.
Gráfico de superficie de agua vs tiempo estimada por el modelo.

Discusión:
Casos en que el modelo funciona muy bien.
Casos en que comete errores y posibles causas (nubosidad, saturación, etc.).

## 8. Aplicación en contexto real
Algunos usos potenciales del modelo:
Monitoreo de lagunas y embalses en zonas áridas o de montaña.
Gestión hídrica agrícola: planificación de riego según disponibilidad de agua superficial.
Alertas tempranas por descenso de niveles de agua.
Soporte a políticas ambientales, permitiendo seguimiento histórico de cuerpos de agua.
