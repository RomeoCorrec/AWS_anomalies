# Lambda + API Gateway — design

**Date :** 2026-08-05
**Sous-projet :** 4e et dernier maillon de la chaîne d'inférence AWS, après S3/compte,
SageMaker Training, et SageMaker Serverless Inference Endpoint (voir
`docs/aws-architecture.md`).

## Contexte

L'endpoint SageMaker Serverless (`aws-anomalies-bottle`) existe et a été validé
(scores 0.9098 / 0.2531 sur des images de test). Il est invoqué aujourd'hui
directement via `aws sagemaker-runtime invoke-endpoint` en CLI. Ce sous-projet
expose cette même invocation derrière une API HTTP publique, via Lambda +
API Gateway, sans passer par les credentials AWS du poste local.

## Architecture

```
Client
  │  POST /predict  (Content-Type: image/*, header x-api-key: <secret>)
  ▼
API Gateway (HTTP API)
  │
  ├─► Lambda Authorizer (valide x-api-key contre une variable d'environnement)
  │
  ▼ (si autorisé)
Lambda "predict" (Python 3.12, zip, boto3 seul — pas de dépendance externe)
  │  sagemaker_runtime.invoke_endpoint(
  │      EndpointName="aws-anomalies-bottle",
  │      Body=<bytes bruts du corps>,
  │      ContentType="application/x-image",
  │  )
  ▼
SageMaker Serverless Endpoint (aws-anomalies-bottle, sous-projet précédent)
  │
  ▼
{"score": float, "is_anomaly": bool} — relayé tel quel par la Lambda au client
```

Deux fonctions Lambda distinctes, toutes deux packagées en zip Python simple
(pas de container BYOC — aucune dépendance ML ici, contrairement aux
sous-projets training/serving) :

- **`predict`** : relaie le corps binaire de la requête HTTP vers
  `invoke_endpoint`, retourne la réponse JSON de l'endpoint telle quelle.
- **`authorizer`** : Lambda authorizer HTTP API au format simple response,
  compare `event["headers"]["x-api-key"]` à une variable d'environnement.

**Type d'API Gateway :** HTTP API (pas REST API). Plus simple, moins cher,
support natif du binaire sans configuration de Content-Types explicite.
Contrepartie assumée : HTTP API ne supporte pas les clés API/Usage Plans
natifs d'AWS (fonctionnalité REST API uniquement) — d'où l'authorizer Lambda
custom pour obtenir un contrôle d'accès équivalent.

**Route unique :** `POST /predict`.

## Composants et fichiers

- `src/aws/lambda_predict.py` — handler de la Lambda `predict`.
- `src/aws/lambda_authorizer.py` — handler de la Lambda `authorizer`.
- `src/aws/deploy_api.py` — script de déploiement (SDK `boto3`), crée/met à
  jour : les 2 fonctions Lambda, leurs rôles d'exécution IAM, l'API Gateway
  HTTP API, la route `POST /predict`, l'authorizer, et les permissions
  d'invocation Lambda↔API Gateway (`lambda:AddPermission`).

## Gestion d'erreurs

- Image invalide → l'endpoint SageMaker renvoie déjà 400
  `{"error": "invalid image"}` (comportement déjà vérifié en sous-projet 3) ;
  la Lambda `predict` relaie ce statut et ce corps tels quels.
- Endpoint SageMaker non déployé/inexistant → `invoke_endpoint` lève une
  `ClientError` (`ValidationException` ou équivalent) ; la Lambda `predict`
  catch cette exception et retourne 503 avec un message explicite
  (`{"error": "endpoint unavailable"}`) plutôt qu'un 500 générique opaque.
- Clé API absente ou invalide → l'authorizer refuse (403 renvoyé par API
  Gateway) ; la Lambda `predict` n'est jamais invoquée, donc aucun appel à
  l'endpoint (donc aucun coût) sur une requête non autorisée.
- Timeout Lambda `predict` : ≥30s, pour couvrir le cold start du endpoint
  Serverless (garde-fou du projet).

## IAM

- **Rôle d'exécution `predict`** (nouveau) : `sagemaker:InvokeEndpoint`
  scopé à l'ARN de `aws-anomalies-bottle` uniquement, + permissions
  CloudWatch Logs standard (`logs:CreateLogGroup`, `logs:CreateLogStream`,
  `logs:PutLogEvents`) scopées au log group de cette fonction.
- **Rôle d'exécution `authorizer`** (nouveau) : uniquement les permissions
  CloudWatch Logs standard — aucun accès à un autre service AWS n'est
  nécessaire, la validation se fait entièrement en mémoire contre une
  variable d'environnement.
- **`aws-anomalies-local`** (étendu) : permissions pour créer/modifier les 2
  fonctions Lambda, l'API Gateway HTTP API et ses routes/authorizers, les 2
  nouveaux rôles IAM d'exécution — toutes scopées aux ressources préfixées
  `aws-anomalies-*`, aucun wildcard `*`.

## Secret de l'authorizer

Stocké en variable d'environnement de la Lambda `authorizer`, fixée au
déploiement par `deploy_api.py` (paramètre CLI, jamais commité en clair dans
le repo). Pas de rotation ni d'audit d'accès nécessaires pour ce portfolio —
Secrets Manager serait disproportionné ici.

## Tests

**Unitaires** (`tests/test_lambda_predict.py`, `tests/test_lambda_authorizer.py`),
mock `boto3` comme déjà pratiqué pour `deploy_endpoint.py` :
- `predict` : image valide → relaie score/is_anomaly ; endpoint renvoie 400
  → relaie tel quel ; endpoint absent (`ClientError`) → 503.
- `authorizer` : clé valide → policy `Allow` ; clé invalide ou absente →
  policy `Deny`.

**E2E manuel** (comme pour le sous-projet 3, pas de repo file) :
1. Redéployer l'endpoint SageMaker (`deploy_endpoint.py`, déjà existant).
2. Déployer Lambda + API Gateway (`deploy_api.py`).
3. Appeler l'URL réelle avec `curl` (image binaire + header `x-api-key`),
   vérifier `score`/`is_anomaly` contre les valeurs de référence connues
   (0.9098 défectueux / 0.2531 sain).
4. Vérifier le rejet sans clé API (403) et avec une image invalide (400).
5. Détruire l'endpoint SageMaker (coût à l'usage). Lambda et API Gateway
   n'ont aucun coût au repos (facturés à l'invocation) — peuvent rester en
   place ou être détruits selon préférence à ce moment-là.

## Hors périmètre

- Terraform (tâche séparée du CLAUDE.md, pas encore commencée).
- Monitoring CloudWatch dédié (alarmes, dashboards).
- Rate limiting / throttling au-delà des quotas par défaut d'API Gateway.
- Rotation du secret de l'authorizer.
- Support d'autres catégories que `bottle`.
