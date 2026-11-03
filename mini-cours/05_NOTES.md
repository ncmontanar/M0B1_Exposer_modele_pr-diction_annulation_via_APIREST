
## Tâche 2 . 

### 2.1 dans le fichier en contexte main.py explique à quoi sert le lifespan ?

il est utilisé pour : localiser le modèle à charger *joblib.load(MODEL_PATH)* & effacer l'état du modele à l'arret (state.clear()) ; lifespan  = pronostique de duréée d'vie (Le `lifespan` dans FastAPI est utilisé pour définir des actions à exécuter au démarrage et à l'arrêt de l'application. Dans le fichier `main.py`)

Où le modèle est-il chargé ? : dans le dictionnaire *model* de l'état partagé (`state["model"]`);

Que fait la fonction /predict ? : envoie une requete (post) au serveur avec l'info sur la prediction d'une annulation d'une réservation à partir des données fournies.

### 2.2 
app/schemas.py : quels champs une réservation doit-elle contenir ?   
    hotel: str
    lead_time: int
    arrival_date_month: str
    stays_in_weekend_nights: int
    stays_in_week_nights: int
    adults: int
    children: int
    market_segment: str
    deposit_type: str
    customer_type: str
    country: str
    adr: float
    previous_cancellations: int
    total_of_special_requests: int
    required_car_parking_spaces: int

Lesquels sont déjà contraints ? *Aucun*

### 2.3 
data/hotel_bookings_sample.csv :
    quelles sont les valeurs possibles de :
        hotel  : *city hotel , resort hotel*
        deposit_type : *No Deposit, Non Refund,  Refoundable*
        market_segment :    *Aviation*
                            *Complementary*
                            *Corporate*
                            *Direct*
                            *Groups*
                            *Offline TA/TO*
                            *Online TA*
    Quelles plages pour 
            lead_time : 1- 594
            adr :  0.0 - 314.67


model/train_baseline.py : 
    quelles colonnes le modèle utilise-t-il :

        ```python
        CATEGORICAL_FEATURES = [
            "hotel",
            "arrival_date_month",
            "market_segment",
            "deposit_type",
            "customer_type",
            "country",
            ]
            NUMERICAL_FEATURES = [
            "lead_time",
            "stays_in_weekend_nights",
            "stays_in_week_nights",
            "adults",
            "children",
            "adr",
            "previous_cancellations",
            "total_of_special_requests",
            "required_car_parking_spaces",
            ]
        ```

    Pourquoi n'a-t-on pas besoin d'encoder les variables nous-mêmes avant d'appeler predict ?: la funcion *def build_pipeline() * s'occupe "d'encoder" (asigner valeurs en nombres que le modèle peut comprendre pour le modèle) les variables catégorielles et la normalisation des variables numériques.

## Tâche 3 — Implémenter /predict


```json
{
  "hotel": "Resort Hotel",
  "lead_time": 107,
  "arrival_date_month": "July",
  "stays_in_weekend_nights": 2,
  "stays_in_week_nights": 5,
  "adults": 2,
  "children": 0,
  "market_segment": "Online TA",
  "deposit_type": "No Deposit",
  "customer_type": "Transient",
  "country": "PRT",
  "adr": 110.7,
  "previous_cancellations": 0,
  "total_of_special_requests": 2,
  "required_car_parking_spaces": 0
}
```

### Tâche 4 — Journaliser avec Loguru 



