# Prévision de la demande de vélos en libre-service — projet MLOps (5MLDE)

Chaîne MLOps complète et exécutable en local : ingestion de données vérifiée, contrôles qualité bloquants, entraînement orchestré, suivi et registre de modèles avec règles de mise en service et retour arrière, API de prédiction tracée, collecte de feedbacks, supervision et tests de charge avec recommandation de dimensionnement.

Le modèle est volontairement simple ; l'objet du projet est la chaîne opérationnelle autour de lui.

**Binôme :** Rania Chanoun et Elodie Andriamahanintsoa.

- **Notebook de démonstration commenté :** [`notebooks/projet_mlops_velos.ipynb`](notebooks/projet_mlops_velos.ipynb) — exécutable en local **ou sur Google Colab** (voir §4)
- **Synthèse des choix et résultats OPS :** [`SYNTHESE_OPS.md`](SYNTHESE_OPS.md)

---

## 1. Sujet

Un opérateur de vélos en libre-service veut prévoir le **nombre de locations par heure** pour répartir sa flotte et planifier ses équipes. Le modèle prédit `cnt` (locations horaires) à partir de l'heure, du type de jour et de la météo.

## 2. Données

| | |
|---|---|
| Jeu | *Bike Sharing Dataset* — Fanaee-T, H. & Gama, J. (2013) |
| Source officielle | <https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset> |
| Fichier utilisé | `hour.csv` — 17 379 lignes horaires, 2011-01-01 → 2012-12-31 |
| Licence | **CC BY 4.0** (réutilisation libre avec citation de la source) |
| Données personnelles | aucune (comptages agrégés par heure) |

**Récupération :** `python -m bikeml setup` télécharge l'archive UCI (repli automatique sur un miroir public du même fichier si UCI est inaccessible), **vérifie l'empreinte SHA-256** fixée dans la configuration et écrit une fiche de provenance `data/raw/PROVENANCE.json` (source, date, empreinte, licence, citation). Les données ne sont pas versionnées dans le dépôt.

**Rejeu par lots :** le jeu est historique ; on simule l'arrivée de nouvelles données en rejouant l'historique : un lot initial (année 2011), puis un lot par mois de 2012. `python -m bikeml release` dépose le lot suivant dans `data/incoming/`, ce qui déclenche le pipeline.

## 3. Architecture

```
data/incoming ──► validation ──► (rejet) quarantaine + rapport + alerte
                      │
                      ▼
               data/accepted ──► entraînement ──► MLflow (run + version)
                                                     │
                     évaluation challenger / champion / référence naïve (même période)
                                                     │
                          promu : alias @champion ──► API rechargée à chaud
                          rejeté : tag + alerte, champion conservé

API FastAPI ──► data/ops.db (requêtes, prédictions, feedbacks, runs, alertes) ──► dashboard Streamlit
```

| Brique | Outil | Fichiers |
|---|---|---|
| Configuration | YAML unique + variables d'environnement | `configs/config.yaml`, `.env.example` |
| Ingestion / qualité | pandas | `src/bikeml/data/` |
| Modèle | scikit-learn (HistGradientBoosting, cible log) | `src/bikeml/model/train.py` |
| Suivi / registre | MLflow (SQLite local) | `src/bikeml/model/registry.py` |
| Orchestration | Prefect 3 | `src/bikeml/orchestration/` |
| Service | FastAPI + uvicorn | `src/bikeml/api/` |
| Persistance des traces | SQLite (WAL) | `src/bikeml/monitoring/store.py` |
| Alertes | logs + base + webhook optionnel | `src/bikeml/monitoring/alerts.py` |
| Supervision | Streamlit, métriques Prometheus (`/metrics`) | `dashboard/app.py` |
| Charge | Locust + psutil | `load_tests/`, `scripts/run_load_tests.py` |
| Tests | pytest | `tests/` |

## 4. Installation

Prérequis : Python 3.12 ou 3.13 (ou Google Colab), aucun service externe.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows : .venv\Scripts\activate
pip install -r requirements.txt      # versions figées
pip install -e . --no-deps           # paquet bikeml
```

(`make install` fait la même chose. Sous Windows sans `make`, utiliser directement les commandes `python -m bikeml ...` indiquées ci-dessous.)

### Sur Google Colab

1. Importer `notebooks/projet_mlops_velos.ipynb` dans Colab (*Fichier → Importer un notebook*).
2. Exécuter la première cellule de code : elle demande `velo-mlops.zip` (ou le trouve à la racine de *Mon Drive*), installe les dépendances manquantes et redémarre la session une fois.
3. *Exécution → Tout exécuter*. L'API et le dashboard s'ouvrent via des liens générés par Colab ; une cellule finale permet de sauvegarder les résultats dans Drive.

Testé avec Python 3.12, pandas 2.2, NumPy 2.0 et scikit-learn 1.6 (versions de Colab) ainsi qu'avec les versions figées de `requirements.txt`.

## 5. Reproduire le fonctionnement

```bash
python -m bikeml setup            # données + lots de rejeu
python -m bikeml release          # dépose le lot initial (2011)
python -m bikeml run              # pipeline : validation -> entraînement -> promotion
python -m bikeml models           # versions, alias, statuts, décisions

# arrivée des mois suivants : à la main...
python -m bikeml release && python -m bikeml run
# ... ou en déclenchement automatique (surveille data/incoming)
python -m bikeml watch

python -m bikeml api              # API sur http://127.0.0.1:8000 (doc : /docs)
streamlit run dashboard/app.py    # supervision sur http://localhost:8501
mlflow ui --backend-store-uri sqlite:///mlflow.db   # interface MLflow
```

Exemple d'appel :

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
  -d '{"season":3,"mnth":7,"hr":17,"holiday":0,"weekday":3,"workingday":1,"weathersit":1,"temp":0.72,"atemp":0.66,"hum":0.45,"windspeed":0.19}'
# {"request_id":"...","prediction":848.02,"model_version":"13","latency_ms":15.1}

curl -X POST http://127.0.0.1:8000/feedback -H "Content-Type: application/json" \
  -d '{"request_id":"<request_id ci-dessus>","actual_cnt":812}'
```

### Incidents contrôlés

```bash
python -m bikeml release-bad && python -m bikeml run   # lot invalide -> quarantaine, alerte, code retour 1
MLFLOW_TRACKING_URI=http://127.0.0.1:5001 python -m bikeml api --port 8001
                                                       # registre indisponible -> /health "degraded", /predict 503
python -m bikeml rollback                              # retour à la version précédente (API rechargée ;
                                                       # @previous ne pointe jamais sur la version retirée)
python -m bikeml retrain --set max_iter=30             # version moins bonne -> rejetée par les règles
```

### Tests

```bash
python -m pytest                       # 33 tests, sans réseau ni données réelles
python scripts/run_load_tests.py       # campagne de charge -> reports/load_tests/ (traces dans data/ops_loadtest.db)
```

### Orchestration avec l'interface Prefect (optionnel)

```bash
prefect server start                   # http://127.0.0.1:4200
python -m bikeml serve                 # déploiement planifié toutes les 30 s
```

### Avec Docker (déploiement complet)

Une seule image sert à quatre services : serveur MLflow (registre + artefacts), pipeline (`watch`), API et dashboard.

```bash
docker compose up -d --build                                   # démarre les 4 services
docker compose run --rm pipeline python -m bikeml setup        # données + lots de rejeu
docker compose run --rm pipeline python -m bikeml release      # livre le lot 2011
# le service pipeline détecte le lot, entraîne, promeut ; l'API recharge le champion
docker compose run --rm pipeline python -m bikeml release      # mois suivant, etc.
docker compose logs -f pipeline                                # suivre les traitements
```

API : <http://localhost:8000/docs> · dashboard : <http://localhost:8501> · MLflow : <http://localhost:5000>

- Les données et la base des traces vivent dans le volume `bikeml-data`, le registre MLflow dans `mlflow-data` : ils survivent à un redémarrage des conteneurs.
- MLflow 3 refuse les en-têtes `Host` inconnus (protection DNS rebinding) : le serveur est lancé avec `--allowed-hosts mlflow:5000,...`, sinon les autres conteneurs reçoivent une erreur 403.
- Incident « dépendance indisponible » : `docker compose stop mlflow` → l'API garde le modèle chargé, un rechargement renvoie 503 et une alerte ; `docker compose start mlflow` puis `POST /admin/reload` rétablit tout.
- Chaque worker de l'API est limité à un thread de calcul (`OMP_NUM_THREADS=1`).

### Repartir de zéro

`make reset` (ou supprimer `data/`, `mlruns/`, `mlflow.db`, `reports/validation/`). Avec Docker : `docker compose down -v` (supprime aussi les volumes).

## 6. Configuration et secrets

Tout le paramétrage est dans `configs/config.yaml` : chemins, règles de validation, variables et hyperparamètres, règles de promotion, port de l'API, paramètres de charge et SLO. Aucun secret n'est versionné ; les valeurs sensibles passent par l'environnement (voir `.env.example`) :

| Variable | Rôle |
|---|---|
| `ALERT_WEBHOOK_URL` | webhook (Slack, Discord, Teams) qui reçoit les alertes |
| `MLFLOW_TRACKING_URI` | serveur MLflow distant (défaut : `sqlite:///mlflow.db`) |
| `BIKEML_API_URL` | API à notifier après une promotion |
| `BIKEML_CONFIG` | autre fichier de configuration |
| `BIKEML_ADMIN_TOKEN` | jeton exigé par `POST /admin/reload` (en-tête `X-Admin-Token`) ; vide = pas de contrôle |
| `BIKEML_OPS_DB` | autre base de traces (utilisé par les tests de charge) |

## 7. Structure du dépôt

```
configs/config.yaml          configuration
src/bikeml/                  code (CLI, données, modèle, orchestration, API, supervision)
dashboard/app.py             dashboard Streamlit
load_tests/locustfile.py     scénario de charge
scripts/run_load_tests.py    campagne de charge + recommandation
tests/                       tests automatisés
notebooks/                   démonstration commentée
reports/load_tests/          derniers résultats de charge
docs/                        captures du dashboard
Dockerfile                   image unique (API, pipeline, dashboard, MLflow)
docker-compose.yml           déploiement local des 4 services
.streamlit/config.toml       thème du dashboard
```
