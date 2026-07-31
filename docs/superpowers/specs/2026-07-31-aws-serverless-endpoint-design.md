# Design : SageMaker Serverless Inference Endpoint

## Contexte

Suite du sous-projet SageMaker Training : un job d'entraînement a produit un
checkpoint PatchCore (`model.tar.gz`) pour la catégorie `bottle`, stocké sur
S3 (bucket `aws-anomalies-mvtec-romeo`). Ce sous-projet expose ce modèle via
un endpoint HTTP managé par AWS (SageMaker Serverless Inference), pour que
l'inférence soit accessible sans gérer d'instance dédiée. Le sous-projet
suivant (Lambda + API Gateway) invoquera cet endpoint — il n'est pas couvert
ici.

Région unique `eu-west-1`, comme le reste du projet.

## Architecture

```
model.tar.gz (S3)  →  SageMaker Model  →  EndpointConfig (Serverless)  →  Endpoint
                             ↑
                   container BYOC (serveur HTTP maison)
                        /ping, /invocations
```

SageMaker dézippe automatiquement `model.tar.gz` (pointé par `ModelDataUrl`)
vers `/opt/ml/model/` dans le container au démarrage — même mécanisme que
pour n'importe quel modèle SageMaker, aucune adaptation de l'artefact de
training n'est nécessaire.

## Composants

### 1. `src/aws/serve.py`

Serveur HTTP minimal (Flask) exposant deux routes :

- `GET /ping` : retourne `200` si l'`AnomalyDetector` est chargé en mémoire.
  Le modèle est chargé une seule fois, au démarrage du process, pas par
  requête. Si le chargement échoue au démarrage, le serveur ne démarre pas
  du tout — SageMaker verra le health check échouer en boucle et remontera
  l'échec de déploiement (comportement voulu : pas d'endpoint "sain" avec un
  modèle cassé).
- `POST /invocations` : reçoit l'image en body binaire (`Content-Type:
  image/jpeg` ou `image/png`), appelle `AnomalyDetector.predict()`, renvoie
  `{"score": float, "is_anomaly": bool}` en JSON.
  - Body non décodable comme image → `400` avec `{"error": "..."}`.
  - Toute autre exception → `500` générique, message loggé sur
    stdout/stderr (capturé par CloudWatch Logs automatiquement, comme pour
    le job de training). Pas de retry ni de fallback : une image invalide
    est une erreur client, pas une raison de réessayer.

Le modèle (chemin fixe `/opt/ml/model/model.ckpt`, tel que déposé par
SageMaker) et le seuil de décision (`config/threshold.yaml`, catégorie
`bottle`, copié dans l'image comme le reste de `config/`) sont lus au
démarrage du serveur.

### 2. `Dockerfile.sagemaker-serve`

Même squelette que `Dockerfile.sagemaker-train` (`python:3.10-slim`, `uv
sync --frozen --no-dev --no-install-project`, `COPY src/` et `COPY
config/`), seul l'`ENTRYPOINT` change pour lancer `src.aws.serve` au lieu du
script de training.

### 3. `src/aws/deploy_endpoint.py`

CLI qui construit un `sagemaker.model.Model` (image ECR de serving +
`model_data` = URI S3 du `model.tar.gz` produit par le job de training) et
appelle `.deploy()` avec un `ServerlessInferenceConfig` :

- `memory_size_in_mb=2048`
- `max_concurrency=1`

Réutilise le rôle d'exécution IAM déjà créé pour le training
(`aws-anomalies-sagemaker-execution`) si ses permissions suffisent, sinon
l'étend avec les permissions Serverless Inference nécessaires (à valider en
phase d'implémentation, une fois les erreurs IAM réelles constatées — comme
pour les deux sous-projets précédents).

## Tests

- Unitaires sur `serve.py` : routage `/ping` et `/invocations`, parsing du
  body image, gestion des cas d'erreur (image invalide → 400), avec
  `AnomalyDetector` mocké.
- Smoke test local Docker : `docker run` de l'image de serving en local
  (checkpoint de test monté en volume à `/opt/ml/model/`), puis `curl` sur
  `/ping` et `/invocations` avec une vraie image du jeu de test `bottle` —
  avant tout déploiement AWS réel, même filet de sécurité que pour le
  training BYOC.

## Vérification post-déploiement (sur AWS réel)

1. Invoquer l'endpoint réel via `boto3` (`sagemaker-runtime`,
   `invoke_endpoint`) sur une image défectueuse et une image saine du jeu de
   test `bottle`.
2. Comparer les scores obtenus via l'endpoint aux scores obtenus localement
   avec le même checkpoint (`AnomalyDetector` local) — ils doivent être
   identiques, seul le transport HTTP change.
3. **Détruire l'endpoint après vérification** (`delete_endpoint` +
   `delete_endpoint_config` + `delete_model`) : contrairement au job de
   training qui s'arrête de lui-même, un endpoint Serverless facture tant
   qu'il existe, même sans trafic.

## Hors périmètre

- L'intégration Lambda + API Gateway (sous-projet suivant).
- Le monitoring CloudWatch dédié à l'endpoint (alarmes, dashboards) — prévu
  plus tard dans la roadmap AWS.
- Le support d'autres catégories que `bottle` sur cet endpoint (une seule
  catégorie à la fois, cohérent avec le choix fait pour le training).
