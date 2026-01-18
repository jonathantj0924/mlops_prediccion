# ============================================
# IMPORTACIONES
# ============================================

from shiny import App, render, ui, reactive
import numpy as np
import matplotlib.pyplot as plt

# ============================================
# MODELO SIMULADO
# ============================================
#import joblib
import joblib
import funciones85 as mis_funciones
import pandas as pd
import sys

# TRUCO DE COMPATIBILIDAD:
# Le decimos a Python que si el modelo pregunta por estas funciones, 
# las busque en nuestro archivo 'mis_funciones'
#esto guardar en el repositotio base de funciones85 no gusrada mala prctica
#las versiones debe ser las misma donde entrene le modelo 
sys.modules['__main__'].column_ratio = mis_funciones.column_ratio
sys.modules['__main__'].ratio_nombre = mis_funciones.ratio_nombre
sys.modules['__main__'].ClusterSimilaridad = mis_funciones.ClusterSimilaridad

modelo = joblib.load('modelo_entrenado.joblib')


# ============================================
# DATOS SIMULADOS PARA HISTOGRAMAS
# ============================================


import pandas as pd

df = pd.read_csv('mis_datos_entrenamiento.csv')
df['housing_median_age']



# Generar datos simulados para los histogramas
datos_anios = df['housing_median_age']  # Años de vivienda
datos_habitaciones = df['total_rooms'] # Total habitaciones
datos_dormitorios = df['total_bedrooms']  # Total dormitorios
datos_poblacion = df['population']  # Población
datos_hogares = df['households'] # Hogares
datos_ingresos = df['median_income']*10000  # Ingresos medios

# ============================================
# UI
# ============================================

app_ui = ui.page_fluid(
    
    
    ui.tags.style("""
        
        /* ========================================
           CONTENEDOR PRINCIPAL
           ======================================== */
        .container-fluid {
            max-width: 1350px;              /* Ancho máximo del dashboard */
            margin: 0 auto;                 /* Centra el contenedor horizontalmente */
            padding: 8px;                   /* Espacio interno del contenedor */
            transform: scale(0.85);         /* Reduce TODO al 85% del tamaño original */
            transform-origin: top center;   /* Punto de origen para la escala (arriba-centro) */
            width: 100%;                    /* Ancho base responsivo */
            overflow-x: hidden;             /* Elimina scroll horizontal global */
        }
        
        /* ========================================
           BODY (CUERPO DE LA PÁGINA)
           ======================================== */
        body {
            font-size: 7px !important;      /* Tamaño de fuente base */
            overflow-x: hidden;             /* Previene scroll horizontal en body */
        }
        
        /* ========================================
           TÍTULO PRINCIPAL (H2)
           ======================================== */
        h2 {
            font-size: 25px !important;     /* Tamaño de fuente del título */
            background: #45558a;  /* color fondo*/
            color: white !important;        /* Color del texto (blanco) */
            padding: 10px 15px !important;  /* Espacio interno del título (arriba/abajo izq/der) */
            border-radius: 8px !important;  /* Esquinas redondeadas */
            margin-bottom: 15px !important; /* Espacio debajo del título */
            box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;  /* Sombra sutil */
            font-weight: 600 !important;    /* Grosor de la fuente (semi-bold) */
        }
        
        /* ========================================
           ENCABEZADO DE CARDS
           ======================================== */
        .card-header {
            font-size: 14px !important;     /* Tamaño de fuente del encabezado */
            padding: 5px 10px !important;   /* Espacio interno del encabezado */
        }
        
        /* ========================================
           CARDS (TARJETAS)
           ======================================== */
        .card {
            margin-bottom: 4px !important;  /* Espacio entre cards verticalmente */
            overflow: hidden;               /* Elimina scroll dentro de cards */
        }
        
        /* ========================================
           CUERPO DE CARDS
           ======================================== */
        .card-body {
            padding: 8px !important;        /* Espacio interno del cuerpo del card */
            overflow-x: hidden;             /* Previene scroll horizontal en card-body */
        }
        
        /* ========================================
           INPUTS Y SELECTS (CONTROLES DE FORMULARIO)
           ======================================== */
        .form-control, .form-select {
            font-size: 12px !important;     /* Tamaño de fuente de los inputs */
            padding: 3px 6px !important;    /* Espacio interno de los inputs */
            margin-bottom: 2px !important;  /* Espacio debajo de cada input */
        }
        
        /* ========================================
           LABELS (ETIQUETAS DE INPUTS)
           ======================================== */
        label {
            font-size: 12px !important;     /* Tamaño de fuente de las etiquetas */
            margin-bottom: 1px !important;  /* Espacio debajo de cada etiqueta */
        }
        
        /* ========================================
           FILAS DENTRO DE CARDS
           ======================================== */
        .card-body .row {
            margin-bottom: 3px !important;  /* Espacio entre filas de inputs */
            margin-left: 0 !important;      /* Elimina margen izquierdo negativo (previene scroll) */
            margin-right: 0 !important;     /* Elimina margen derecho negativo (previene scroll) */
        }
        
        /* ========================================
           GRUPOS DE FORMULARIO
           ======================================== */
        .form-group {
            margin-bottom: 3px !important;  /* Espacio debajo de cada grupo de formulario */
        }
        
        /* ========================================
           BOTONES
           ======================================== */
        .btn {
            font-size: 14px !important;     /* Tamaño de fuente del botón */
            padding: 4px 8px !important;   /* Espacio interno del botón */
        }
        
        /* ========================================
           CONTENEDOR DE BOTÓN CENTRADO
           ======================================== */
        .btn-container {
            text-align: center;             /* Centra el botón horizontalmente */
            margin-top: 8px;                /* Espacio arriba del botón */
        }
        
        /* ========================================
           TEXTO PREFORMATEADO (OUTPUT)
           ======================================== */
        pre {
            font-size: 11px !important;     /* Tamaño de fuente del output de texto */
        }
        
            #prediccion {
            font-size: 25px;         /* Tamaño destacado para el resultado */
            background-color: #e9ecef; 
            border-radius: 4px;
            padding: 10px;
            overflow-x: hidden; 
        }
        
        
        /* Ajusta todos los contenedores de gráficos de Shiny */
        .shiny-plot-output {
            height: 300px !important; /* Altura fija para que se vean compactos */
            max-height: 225px;
            width: 100% !important;
        }
        
        /* Opcional: Quitar espacio extra de las tarjetas para que queden más pegados */
        .card-body {
            padding: 10px !important;
        }
        
    """),
    
    
    
    
    
    # Título principal de la aplicación
    ui.h2("Machine Learning - Predictor de Avaluos Inmobiliarios con IA"),
    
    # Fila principal que contiene las 3 columnas
    ui.row(
        # ========== COLUMNA 1: INPUTS (3 espacios de ancho) ==========
        ui.column(
            3,
            ui.card(
                ui.card_header("📥 Ingrese Datos de Entrada"),
                
                
                # Fila 1: Longitud y Latitud
                ui.row(
                    ui.column(6, ui.input_numeric("feature1", "Longitud:", value=-122.23)),
                    ui.column(6, ui.input_numeric("feature2", "Latitud:", value=37.88)),
                ),
                
                # Fila 2: Años de vivienda y Total habitaciones
                ui.row(
                    ui.column(6, ui.input_numeric("feature3", "Edad Media Distrit", value=15, min=0, max=200)),
                    ui.column(6, ui.input_numeric("feature4", "N° habitaciones", value=880)),
                ),
                
                # Fila 3: Total dormitorios y Población
                ui.row(
                    ui.column(6, ui.input_numeric("feature5", "Total Dormitorios:", value=750)),
                    ui.column(6, ui.input_numeric("feature6", "Población Distrit:", value=5000)),
                ),
                
                # Fila 4: Hogares e Ingresos medios
                ui.row(
                    ui.column(6, ui.input_numeric("feature7", "N° hogares Distrit", value=1000)),
                    ui.column(6, ui.input_numeric("feature8", "Ingreso Distrit", value= 85000)),
                ),
                
                # Fila 5: Cercanía al mar (selector dropdown)
                ui.row(
                    ui.column(12, 
                        ui.input_select(
                            "feature9", 
                            "Cercanía al mar:", 
                            choices={
                                "NEAR BAY": "Cerca de bahía",
                                "NEAR OCEAN": "Cerca del océano",
                                "<1H OCEAN": "A menos de 1h del océano",
                                "INLAND": "Interior",
                                "ISLAND": "Isla"
                            },
                            selected="NEAR BAY"
                        )
                    ),
                ),

                # Botón de predicción centrado
                ui.div(
                    ui.input_action_button("predecir", "Ejecutar Predicción", class_="btn-primary"),
                    class_="btn-container"
                ),
                ui.p("Desarrollado: Jonathan Tumipamba", style="color: gray; font-size: 14px"),
            ),
        ),
        
        # ========== COLUMNA 2: HISTOGRAMAS (6 espacios de ancho) ==========
        ui.column(
            6,
            ui.card(
                ui.card_header("📊 Distribución de Variables y Datos Ingresados"),
                
                # Primera fila de histogramas
                ui.row(
                    ui.column(4, ui.output_plot("hist_anios")),
                    ui.column(4, ui.output_plot("hist_habitaciones")),
                    ui.column(4, ui.output_plot("hist_dormitorios")),
                ),
                
                # Segunda fila de histogramas
                ui.row(
                    ui.column(4, ui.output_plot("hist_poblacion")),
                    ui.column(4, ui.output_plot("hist_hogares")),
                    ui.column(4, ui.output_plot("hist_ingresos")),
                ),
            ),
        ),
        
        # ========== COLUMNA 3: OUTPUTS (3 espacios de ancho) ==========
        ui.column(
            3,
            ui.card(
                ui.card_header("📊 Predicción del Modelo"),
                ui.output_text_verbatim("prediccion"),
                ui.output_ui("tarjetaprediccion"),
                ui.card_footer( # Trajet interseante m
                               
                    ui.div("RF LMRegressor | R²: 0.81 | Validación K-Fold", 
                        style="font-size: 0.75rem; color: #888; margin-bottom: 5px; border-bottom: 1px solid #eee;"),
                    # Tu firma actual
                    ui.div(
                        "© 2026 Jonathan ",
                        ui.a(" | Codigo", href="tu_link", target="_blank"),
                        style="font-size: 0.8rem; text-align: center;"
                    )
                )
            ),
        ),
    ),
)

# ============================================
# SERVIDOR
# ============================================

   
def server(input, output, session):

    # --- LÓGICA DE CÁLCULO (Se comparte entre los dos outputs) ---
    @reactive.calc
    @reactive.event(input.predecir)
    def calculo_modelo():
        # 1. Creamos el DataFrame
        X_df = pd.DataFrame({
            'longitude': [float(input.feature1())],
            'latitude': [float(input.feature2())],
            'housing_median_age': [float(input.feature3())],
            'total_rooms': [float(input.feature4())],
            'total_bedrooms': [float(input.feature5())],
            'population': [float(input.feature6())],
            'households': [float(input.feature7())],
            'median_income': [float(input.feature8())/10000],
            'ocean_proximity': [input.feature9()]
        })
        
        # 2. Realizamos la predicción
        pred = modelo.predict(X_df)[0]
        
        # Retornamos un diccionario con el valor y los datos originales para el verbatim
        return {
            "valor": pred,
            "inputs": [input.feature1(), input.feature2(), input.feature3(), 
                       input.feature4(), input.feature5(), input.feature6(), 
                       input.feature7(), input.feature8(), input.feature9()]
        }

    # --- OUTPUT 1: El Verbatim que ya tenías ---
    @render.text
    def prediccion():
        datos = calculo_modelo() # Llama al cálculo
        p = datos["valor"]
        f = datos["inputs"]
        ##Predicción del Precio: ${p:,.2f}
        return f"""
Características usadas:
━━━━━━━━━━━━━━━━━━━━━━━━
Coordenadas: ({f[0]}, {f[1]})
Años de vivienda: {f[2]} años
Total habitaciones: {f[3]}
Total dormitorios: {f[4]}
Población: {f[5]}
Hogares: {f[6]}
Ingresos medios: ${f[7]}k
Cercanía al mar: {f[8]}
"""
    
       
    #logica diferente en esta enviamos la tarjeta el ui a la interfaz -aseguramos que la atraejat apresca cuando se ejecute l calculo
    @render.ui
    def tarjetaprediccion():
        datos = calculo_modelo() # Usa el MISMO cálculo sin ejecutar el modelo de nuevo
        p = datos["valor"]
        
        return ui.value_box(
        # 1. Cambiar tamaño del TÍTULO (opcional)
        ui.span("PREDICCION VALOR ESTIMADO DE MERCADO", style="font-size: 0.8rem; text-align: center;"),        
        # 2. Cambiar tamaño del VALOR (el número)
        ui.span(f"${p:,.2f}", style="font-size: 1.2rem; font-weight: bold; text-align: center;"),        
        theme="primary"
    )
        
        
        
    # Histograma 1: Años de vivienda
    @render.plot
    def hist_anios():
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.hist(datos_anios, bins=30, color='steelblue', alpha=0.6)
        # Línea vertical para el valor del usuario
        ax.axvline(x=input.feature3(), color='red', linestyle='--', linewidth=2)
        ax.set_title('Años de Vivienda', fontsize=9)
        #ax.set_xlabel('Años', fontsize=8)
        ax.set_ylabel('Frecuencia', fontsize=8)
        plt.tight_layout()
        return fig
    
    # Histograma 2: Total habitaciones
    @render.plot
    def hist_habitaciones():
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.hist(datos_habitaciones, bins=30, color='coral', alpha=0.6)
        # Línea vertical para el valor del usuario
        ax.axvline(x=input.feature4(), color='red', linestyle='--', linewidth=2)
        ax.set_title('Total Habitaciones', fontsize=9)
        #ax.set_xlabel('Habitaciones', fontsize=8)
        ax.set_ylabel('Frecuencia', fontsize=8)
        plt.tight_layout()
        return fig
    
    # Histograma 3: Total dormitorios
    @render.plot
    def hist_dormitorios():
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.hist(datos_dormitorios, bins=30, color='mediumseagreen', alpha=0.6)
        # Línea vertical para el valor del usuario
        ax.axvline(x=input.feature5(), color='red', linestyle='--', linewidth=2)
        ax.set_title('Total Dormitorios', fontsize=9)
        #ax.set_xlabel('Dormitorios', fontsize=8)
        ax.set_ylabel('Frecuencia', fontsize=8)
        plt.tight_layout()
        return fig
    
    # Histograma 4: Población
    @render.plot
    def hist_poblacion():
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.hist(datos_poblacion, bins=30, color='mediumpurple', alpha=0.6)
        # Línea vertical para el valor del usuario
        ax.axvline(x=input.feature6(), color='red', linestyle='--', linewidth=2)
        ax.set_title('Población', fontsize=9)
        #ax.set_xlabel('Población', fontsize=8)
        ax.set_ylabel('Frecuencia', fontsize=8)
        plt.tight_layout()
        return fig
    
    # Histograma 5: Hogares
    @render.plot
    def hist_hogares():
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.hist(datos_hogares, bins=30, color='gold', alpha=0.6)
        # Línea vertical para el valor del usuario
        ax.axvline(x=input.feature7(), color='red', linestyle='--', linewidth=2)
        ax.set_title('N° Hogares', fontsize=9)
        #ax.set_xlabel('Hogares', fontsize=8)
        ax.set_ylabel('Frecuencia', fontsize=8)
        plt.tight_layout()
        return fig
    
    # Histograma 6: Ingresos medios
    @render.plot
    def hist_ingresos():
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.hist(datos_ingresos, bins=30, color='tomato', alpha=0.6)
        # Línea vertical para el valor del usuario
        ax.axvline(x=input.feature8(), color='red', linestyle='--', linewidth=2)
        ax.set_title('Ingresos Medios', fontsize=9)
        #ax.set_xlabel('Ingresos (miles $)', fontsize=8)
        ax.set_ylabel('Frecuencia', fontsize=8)
        plt.tight_layout()
        return fig

app = App(app_ui, server)



# Diseñé una interfaz reactiva en Shiny for Python 
# que no solo entrega la predicción, sino que visualiza 
# la posición del dato ingresado respecto a la distribución histórica
# (los histogramas del centro), validando el resultado con un R² de 0.82 mediante Random Forest."