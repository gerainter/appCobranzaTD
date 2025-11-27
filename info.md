# Crear ambiente virtual

conda create -n streamlit_pyspark python=3.10 -y

# Activar el ambiente

conda activate streamlit_pyspark

# Si tienes error de java

https://adoptium.net/es/download

# Si no tienes instalado streamlit


# si no tienes instalado wordcloud
conda install wordcloud

# Ejemplo de ejecución 
streamlit run appCobranzaTDC.py

# Instalar dependencias

pip install streamlit pyspark pandas plotly scikit-learn openpyxl

```

### 1.2 Estructura del Proyecto

Vamos a crear esta estructura de carpetas:
```

appcobranzatdc/
│
├── appCobranzaTDC.py # Aplicación principal de Streamlit
├── data/ # Carpeta para datasets
│ └── df_result_nuevos.csv # Dataset de recuperacion de cobranza despues de aplicar el modelos
| └── data/recuperacion_deuda_tdc_san.csv # Dataset de recuperacion de cobranza para entrenar el modelo
├── GHA_Proyecto_Final_Ciencia_Datos.ipynb # Notebook del analisis de los datos, EDA, entrenamiento modelo
