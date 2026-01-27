import sys

import funciones85 as mis_funciones
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shiny import App, reactive, render, ui

# Compatibilidad para cargar el modelo entrenado con funciones externas
sys.modules["__main__"].column_ratio = mis_funciones.column_ratio
sys.modules["__main__"].ratio_nombre = mis_funciones.ratio_nombre
sys.modules["__main__"].ClusterSimilaridad = mis_funciones.ClusterSimilaridad

modelo = joblib.load("modelo_entrenado.joblib")

df = pd.read_csv("mis_datos_entrenamiento.csv")
df["housing_median_age"]

datos_anios = df["housing_median_age"]
datos_habitaciones = df["total_rooms"]
datos_dormitorios = df["total_bedrooms"]
datos_poblacion = df["population"]
datos_hogares = df["households"]
datos_ingresos = df["median_income"] * 10000

app_ui = ui.page_fluid(
    ui.tags.style("""
        .container-fluid {
            max-width: 1350px;
            margin: 0 auto;
            padding: 8px;
            transform: scale(0.85);
            transform-origin: top center;
            width: 100%;
            overflow-x: hidden;
        }

        body {
            font-size: 7px !important;
            overflow-x: hidden;
        }

        h2 {
            font-size: 25px !important;
            background: #45558a;
            color: white !important;
            padding: 10px 15px !important;
            border-radius: 8px !important;
            margin-bottom: 15px !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
            font-weight: 600 !important;
        }

        .card-header {
            font-size: 14px !important;
            padding: 5px 10px !important;
        }

        .card {
            margin-bottom: 4px !important;
            overflow: hidden;
        }

        .card-body {
            padding: 8px !important;
            overflow-x: hidden;
        }

        .form-control, .form-select {
            font-size: 12px !important;
            padding: 3px 6px !important;
            margin-bottom: 2px !important;
        }

        label {
            font-size: 12px !important;
            margin-bottom: 1px !important;
        }

        .card-body .row {
            margin-bottom: 3px !important;
            margin-left: 0 !important;
            margin-right: 0 !important;
        }

        .form-group {
            margin-bottom: 3px !important;
        }

        .btn {
            font-size: 14px !important;
            padding: 4px 8px !important;
        }

        .btn-container {
            text-align: center;
            margin-top: 8px;
        }

        pre {
            font-size: 11px !important;
        }

        #prediccion {
            font-size: 25px;
            background-color: #e9ecef;
            border-radius: 4px;
            padding: 10px;
            overflow-x: hidden;
        }

        .shiny-plot-output {
            height: 300px !important;
            max-height: 225px;
            width: 100% !important;
        }

        .card-body {
            padding: 10px !important;
        }
    """),
    ui.h2("Machine Learning - Predictor de Avaluos Inmobiliarios con IA"),
    ui.row(
        ui.column(
            3,
            ui.card(
                ui.card_header("📥 Ingrese Datos de Entrada"),
                ui.row(
                    ui.column(
                        6, ui.input_numeric("feature1", "Longitud:", value=-122.23)
                    ),
                    ui.column(6, ui.input_numeric("feature2", "Latitud:", value=37.88)),
                ),
                ui.row(
                    ui.column(
                        6,
                        ui.input_numeric(
                            "feature3", "Edad Media Distrit", value=15, min=0, max=200
                        ),
                    ),
                    ui.column(
                        6, ui.input_numeric("feature4", "N° habitaciones", value=880)
                    ),
                ),
                ui.row(
                    ui.column(
                        6, ui.input_numeric("feature5", "Total Dormitorios:", value=750)
                    ),
                    ui.column(
                        6,
                        ui.input_numeric("feature6", "Población Distrit:", value=5000),
                    ),
                ),
                ui.row(
                    ui.column(
                        6,
                        ui.input_numeric("feature7", "N° hogares Distrit", value=1000),
                    ),
                    ui.column(
                        6, ui.input_numeric("feature8", "Ingreso Distrit", value=85000)
                    ),
                ),
                ui.row(
                    ui.column(
                        12,
                        ui.input_select(
                            "feature9",
                            "Cercanía al mar:",
                            choices={
                                "NEAR BAY": "Cerca de bahía",
                                "NEAR OCEAN": "Cerca del océano",
                                "<1H OCEAN": "A menos de 1h del océano",
                                "INLAND": "Interior",
                                "ISLAND": "Isla",
                            },
                            selected="NEAR BAY",
                        ),
                    ),
                ),
                ui.div(
                    ui.input_action_button(
                        "predecir", "Ejecutar Predicción", class_="btn-primary"
                    ),
                    class_="btn-container",
                ),
                ui.p(
                    "Desarrollado: Jonathan Tumipamba",
                    style="color: gray; font-size: 14px",
                ),
            ),
        ),
        ui.column(
            6,
            ui.card(
                ui.card_header("📊 Distribución de Variables y Datos Ingresados"),
                ui.row(
                    ui.column(4, ui.output_plot("hist_anios")),
                    ui.column(4, ui.output_plot("hist_habitaciones")),
                    ui.column(4, ui.output_plot("hist_dormitorios")),
                ),
                ui.row(
                    ui.column(4, ui.output_plot("hist_poblacion")),
                    ui.column(4, ui.output_plot("hist_hogares")),
                    ui.column(4, ui.output_plot("hist_ingresos")),
                ),
            ),
        ),
        ui.column(
            3,
            ui.card(
                ui.card_header("📊 Predicción del Modelo"),
                ui.output_text_verbatim("prediccion"),
                ui.output_ui("tarjetaprediccion"),
                ui.card_footer(
                    ui.div(
                        "GLM LinearRegressor | RMSE: 47189.7 | Validación K-Fold",
                        style="font-size: 0.75rem; color: #888; margin-bottom: 5px; border-bottom: 1px solid #eee;",
                    ),
                    ui.div(
                        "© 2026 JonathanTJ ",
                        ui.a(
                            " | NoteBook",
                            href="https://019bfbfb-49ae-2883-87d6-7f6b8d7aa2ce.share.connect.posit.cloud/",
                            target="_blank",
                        ),
                        style="font-size: 0.8rem; text-align: center;",
                    ),
                ),
            ),
        ),
    ),
)


def server(input, output, session):
    # Cálculo central compartido por los outputs
    @reactive.calc
    @reactive.event(input.predecir)
    def calculo_modelo():
        X_df = pd.DataFrame(
            {
                "longitude": [float(input.feature1())],
                "latitude": [float(input.feature2())],
                "housing_median_age": [float(input.feature3())],
                "total_rooms": [float(input.feature4())],
                "total_bedrooms": [float(input.feature5())],
                "population": [float(input.feature6())],
                "households": [float(input.feature7())],
                "median_income": [float(input.feature8()) / 10000],
                "ocean_proximity": [input.feature9()],
            }
        )

        pred = modelo.predict(X_df)[0]

        return {
            "valor": pred,
            "inputs": [
                input.feature1(),
                input.feature2(),
                input.feature3(),
                input.feature4(),
                input.feature5(),
                input.feature6(),
                input.feature7(),
                input.feature8(),
                input.feature9(),
            ],
        }

    @render.text
    def prediccion():
        datos = calculo_modelo()
        p = datos["valor"]
        f = datos["inputs"]
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

    @render.ui
    def tarjetaprediccion():
        datos = calculo_modelo()
        p = datos["valor"]

        return ui.value_box(
            ui.span(
                "PREDICCION VALOR ESTIMADO DE MERCADO",
                style="font-size: 0.8rem; text-align: center;",
            ),
            ui.span(
                f"${p:,.2f}",
                style="font-size: 1.2rem; font-weight: bold; text-align: center;",
            ),
            theme="primary",
        )

    @render.plot
    def hist_anios():
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.hist(datos_anios, bins=30, color="steelblue", alpha=0.6)
        ax.axvline(x=input.feature3(), color="red", linestyle="--", linewidth=2)
        ax.set_title("Años de Vivienda", fontsize=9)
        ax.set_ylabel("Frecuencia", fontsize=8)
        plt.tight_layout()
        return fig

    @render.plot
    def hist_habitaciones():
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.hist(datos_habitaciones, bins=30, color="coral", alpha=0.6)
        ax.axvline(x=input.feature4(), color="red", linestyle="--", linewidth=2)
        ax.set_title("Total Habitaciones", fontsize=9)
        ax.set_ylabel("Frecuencia", fontsize=8)
        plt.tight_layout()
        return fig

    @render.plot
    def hist_dormitorios():
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.hist(datos_dormitorios, bins=30, color="mediumseagreen", alpha=0.6)
        ax.axvline(x=input.feature5(), color="red", linestyle="--", linewidth=2)
        ax.set_title("Total Dormitorios", fontsize=9)
        ax.set_ylabel("Frecuencia", fontsize=8)
        plt.tight_layout()
        return fig

    @render.plot
    def hist_poblacion():
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.hist(datos_poblacion, bins=30, color="mediumpurple", alpha=0.6)
        ax.axvline(x=input.feature6(), color="red", linestyle="--", linewidth=2)
        ax.set_title("Población", fontsize=9)
        ax.set_ylabel("Frecuencia", fontsize=8)
        plt.tight_layout()
        return fig

    @render.plot
    def hist_hogares():
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.hist(datos_hogares, bins=30, color="gold", alpha=0.6)
        ax.axvline(x=input.feature7(), color="red", linestyle="--", linewidth=2)
        ax.set_title("N° Hogares", fontsize=9)
        ax.set_ylabel("Frecuencia", fontsize=8)
        plt.tight_layout()
        return fig

    @render.plot
    def hist_ingresos():
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.hist(datos_ingresos, bins=30, color="tomato", alpha=0.6)
        ax.axvline(x=input.feature8(), color="red", linestyle="--", linewidth=2)
        ax.set_title("Ingresos Medios", fontsize=9)
        ax.set_ylabel("Frecuencia", fontsize=8)
        plt.tight_layout()
        return fig


app = App(app_ui, server)
