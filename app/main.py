"""API de prédiction d'annulation — Escale Atlantique (mission FastIA).# application FastAPI (lifespan, /health, /predict à compléter)""" 

import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from time import perf_counter

from loguru import logger # 4. Journalisation avec Loguru

from app.schemas import Booking, PredictionResponse

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT_DIR / "model" / "cancellation_model.joblib"))

# 4.1 Chemin vers le fichier de log
LOG_PATH = ROOT_DIR / "logs" / "api.log"

# Objets partagés par toutes les requêtes (ici : le modèle)
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 4.2 Configuration du logger avec Loguru
    sink_id = logger.add(
        LOG_PATH,
        level="INFO",
        rotation="5 MB",
        retention="7 days",
        compression="zip",
        encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}",
        diagnose=False,
    )
        
    try:
        # Exécuté UNE fois au démarrage : on charge le modèle en mémoire
        state["model"] = joblib.load(MODEL_PATH)
        logger.info("Modèle chargé")
        yield
    finally:
        # Exécuté à l'arrêt du serveur
        state.clear()
        # 4.3 Journalisation de l'arrêt de l'application
        logger.info("Arrêt de l'application")
        logger.remove(sink_id)



app = FastAPI(
    title="Escale Atlantique — Prédiction d'annulation",
    description="Estime la probabilité qu'une réservation soit annulée.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    """Le service est-il en vie, et le modèle chargé ?"""
    return {"status": "ok", "model_loaded": "model" in state}


@app.post("/predict", response_model=PredictionResponse)
def predict(booking: Booking) -> PredictionResponse:
    """Prédit l'annulation d'une réservation.
    """
    # 4.4 journalisation de la requête de prédiction
    start = perf_counter()
    #input_data = booking.model_dump()
    response = None
    
    # 4.1 Initialisation de la réponse et du chronomètre
    try:
        # 3.1 Récupération du modèle depuis l'état partagé
        model = state.get("model")
        if model is None:
            raise HTTPException(status_code=503, detail="Le modèle n'est pas chargé")
        
        # 3.2 Préparation des données pour le modèle
        features = pd.DataFrame([booking.model_dump()])
        
        prediction = model.predict(features)[0]
        
        # 3.3  Indice de la classe correspondant à l'annulation (1) dans la liste des classes du modèle
        cancellation_index = model.classes_.tolist().index(1)
        # 3.4 Probabilité d'annulation selon le modèle
        cancellation_probability = model.predict_proba(features)[
            0, cancellation_index
        ]
        
        # 3.5 Construction de la réponse à retourner
        response =  PredictionResponse(
            annulation=bool(prediction),
            probabilite_annulation=float(cancellation_probability),
        )
        return response
    
    # 4.5 Journalisation de la durée de la requête et du résultat
    finally:
        duration_ms = (perf_counter() - start) * 1000
        
        if response is not None:
            logger.error(
                "POST /predict | entrée={} | échec | durée_ms={:.2f}",
                booking.model_dump(),
                duration_ms,
            )
        else:
            logger.info(
                "POST /predict | entrée={} | succès | durée_ms={:.2f}",
                booking.model_dump(),
                duration_ms,
            )