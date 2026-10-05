"""Análisis y modelamiento del comercio exterior chileno con datos de Aduana (DUS/DIN).

Módulos según la fase de CRISP-DM que implementan:
    comprensión de los datos   download, ingest, schema, references
    preparación de los datos   clean, features, data
    modelado                   graph, activation, forecast, classify
    evaluación                 splits (y las métricas de cada modelo)
    despliegue (simulado)      drift

    causal recorre una segunda iteración del ciclo completo para una pregunta causal (aranceles de EE.UU.), y
    experiment una tercera, con el diseño y análisis de un experimento A/B/n semi-sintético.
"""
