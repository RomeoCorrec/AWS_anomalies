# AWS Anomalies — détection d'anomalies visuelles sur pièces industrielles

Détection d'anomalies visuelles non supervisée (dataset [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad))
avec [PatchCore](https://arxiv.org/abs/2106.08265) via [anomalib](https://github.com/open-edge-platform/anomalib),
packagée et déployée end-to-end sur AWS (SageMaker, Lambda, API Gateway), infra pilotée en Terraform.

Projet portfolio : l'objectif n'est pas la vitesse de livraison mais la compréhension et
la défendabilité de chaque décision technique.

## Catégories et modèles couverts

| Catégorie | Difficulté | Backbone par défaut |
|---|---|---|
| `bottle` | défauts nets | WideResNet50 |
| `carpet` | texture | WideResNet50 |
| `screw` | petits défauts, cas difficile | WideResNet50 |

Backbone alternatif comparé : DINOv2 ViT-S. Ablation sur `coreset_sampling_ratio`
(0.01 / 0.1 / 0.25) — résultats complets dans `results/experiments.csv`.

## Architecture

```
Local
  anomalib (entraînement PatchCore) → checkpoint → AnomalyDetector → Docker

AWS
  S3 (dataset + artefacts)
    → SageMaker Training Job (BYOC, ECR aws-anomalies-train)
    → S3 (model.tar.gz)
    → SageMaker Serverless Inference (BYOC, ECR aws-anomalies-serve)
    → Lambda aws-anomalies-predict (relai)
    → API Gateway HTTP API (POST /predict, authorizer x-api-key)

Infra permanente (S3, ECR, IAM, Lambda, API Gateway) gérée en Terraform (terraform/).
L'endpoint SageMaker Serverless et les Training Jobs restent hors périmètre Terraform :
ce sont des ressources éphémères, recréées à la demande pour maîtriser les coûts.
```

Détail factuel de chaque sous-projet AWS : `docs/aws-architecture.md`.

## Structure du repo

```
src/
  models/        entraînement (train.py), déploiement local (deploy.py), inférence (detector.py)
  eval/          métriques, calibration de seuil (threshold.py), logging des résultats
  aws/           launchers SageMaker, handlers Lambda, serveur d'inférence BYOC
  config.py      chargement des configs YAML (experiment/model/data)
config/
  experiment/    une config par (catégorie, backbone, ratio) — combinaisons de model.yaml + data.yaml
  model/         hyperparamètres PatchCore par backbone
  data/          config du dataset MVTec AD
  threshold.yaml seuils de décision calibrés par catégorie
terraform/       infra AWS permanente (S3, ECR, IAM, Lambda, API Gateway)
docs/            specs et plans d'implémentation par sous-projet, recap d'architecture AWS
results/         résultats d'expériences (CSV, reproductibles) et checkpoints déployés
tests/           tests unitaires
```

## Setup local

```bash
uv sync
```

Le dataset MVTec AD est attendu sous `data/mvtec/<categorie>/`.

## Entraînement local

```bash
uv run python -m src.models.train --experiment config/experiment/bottle_wideresnet50.yaml
```

Chaque run écrit une ligne dans `results/experiments.csv` avec la config complète en
colonnes (reproductible, comparable entre runs).

## Calibration du seuil

```bash
uv run python -m src.eval.threshold --experiment config/experiment/bottle_wideresnet50.yaml
```

Génère les seuils candidats (`results/threshold_candidates_<categorie>.csv`). Le choix
du seuil retenu et sa justification métier sont documentés séparément (décision humaine,
pas automatisée).

## Inférence locale

```python
from pathlib import Path
from src.models.detector import AnomalyDetector

detector = AnomalyDetector(
    experiment_path=Path("config/experiment/bottle_wideresnet50.yaml"),
    checkpoint_path=Path("results/deployed/bottle/model.ckpt"),
    threshold=0.523,
)
detector.predict(Path("data/mvtec/bottle/test/broken_large/000.png"))
# {"score": 0.9098, "is_anomaly": True}
```

## Déploiement AWS

Voir `terraform/README.md` pour l'infra permanente, et `docs/aws-architecture.md` pour
le détail de chaque sous-projet (S3, SageMaker Training, SageMaker Serverless Inference,
Lambda + API Gateway).

Séquence type pour un test end-to-end :

```bash
# 1. Entraîner (SageMaker Training Job, éphémère)
uv run python -m src.aws.launch_training ...

# 2. Déployer l'endpoint Serverless (éphémère, coût par requête)
uv run python -m src.aws.deploy_endpoint \
  --image-uri <account>.dkr.ecr.eu-west-1.amazonaws.com/aws-anomalies-serve:latest \
  --role-arn arn:aws:iam::<account>:role/aws-anomalies-sagemaker-execution \
  --model-data-url s3://<bucket>/output/.../model.tar.gz \
  --endpoint-name aws-anomalies-bottle

# 3. Appeler via l'API (infra permanente Terraform)
curl -X POST "$(terraform -chdir=terraform output -raw predict_invoke_url)" \
  -H "x-api-key: <secret>" \
  -H "Content-Type: image/png" \
  --data-binary @data/mvtec/bottle/test/broken_large/000.png

# 4. Détruire l'endpoint (garde-fou coût — jamais laisser tourner entre les sessions)
aws sagemaker delete-endpoint --endpoint-name aws-anomalies-bottle
aws sagemaker delete-endpoint-config --endpoint-config-name aws-anomalies-bottle
```

**Garde-fou** : l'endpoint SageMaker Serverless facture en continu tant qu'il existe —
toujours le détruire après une session de test. Lambda et API Gateway n'ont pas de coût
au repos.

## Décisions d'architecture

**BYOC (Bring Your Own Container) plutôt que les conteneurs SageMaker intégrés.**
PatchCore via `anomalib` n'a pas de conteneur SageMaker officiel maintenu ; empaqueter
son propre serveur Flask (`src/aws/serve.py`, routes `/ping` + `/invocations`) donne un
contrôle total sur les dépendances (`anomalib`, `torch` CPU) et évite de contraindre le
projet aux hyperparamètres exposés par un conteneur générique. Coût : maintenance du
Dockerfile de serving et du contrat HTTP soi-même.

**SageMaker Serverless Inference plutôt qu'un endpoint temps réel dédié.**
Le trafic attendu (tests ponctuels, pas de production continue) ne justifie pas un
endpoint facturé à l'heure en continu. Le compromis est la latence de cold start (d'où
le timeout Lambda fixé à 30s minimum) et `memory_size_in_mb=4096` (2048 MB s'est révélé
insuffisant en test réel — le modèle PatchCore + WideResNet50 dépasse ce budget mémoire
au chargement).

**Lambda + authorizer API-key plutôt qu'IAM auth ou Cognito.**
Un authorizer `REQUEST` avec un secret partagé (`x-api-key`) est suffisant pour un usage
personnel/démo à un seul consommateur, et bien plus simple à mettre en place et à
expliquer qu'IAM SigV4 ou un pool Cognito. Ne conviendrait pas en l'état pour plusieurs
consommateurs ou une rotation de clé fréquente (limitation connue, voir la review de
`src/aws/deploy_api.py`).

**Terraform : import des ressources existantes, jamais de recréation.**
Le compte AWS avait déjà S3/ECR/IAM en place avant l'introduction de Terraform (sous-
projets 1-3, provisionnés manuellement). Plutôt que de détruire/recréer (risque de perte
de données sur le bucket, de changement d'URI ECR), toutes les ressources préexistantes
sont importées telles quelles, et leur policy de permissions (S3/ECR/IAM growth au fil
des sous-projets) n'est volontairement *pas* redéclarée en HCL — la redéclarer sans
connaître son contenu exact aurait risqué de retirer une permission encore utilisée par
un script existant, ou de provoquer un auto-verrouillage IAM.

**State Terraform local plutôt qu'un backend S3 distant.**
Choix assumé pour un projet solo : pas de risque d'écriture concurrente sur le state, pas
d'infra supplémentaire (bucket S3 + verrou DynamoDB) à maintenir pour un seul opérateur.
Un vrai projet d'équipe nécessiterait un backend partagé avec verrouillage.

**`deploy_api.py` conservé en parallèle de Terraform malgré la duplication.**
Terraform est désormais le chemin de déploiement normal pour Lambda/API Gateway, mais le
script manuel `deploy_api.py` reste dans le repo comme trace du chemin de déploiement
initial (avant Terraform) et comme filet de secours ponctuel. Risque documenté et assumé
(cf. `terraform/README.md`) : ne jamais le relancer après un `terraform apply`, sous
peine de divergence entre le state Terraform et la réalité AWS.

**Principe du moindre privilège appliqué de façon incrémentale, pas anticipée.**
Chaque permission IAM a été ajoutée au moment où une action précise échouait avec
`AccessDenied`, jamais en anticipant un besoin futur avec un wildcard de confort. Plus
lent à itérer, mais garantit qu'aucune permission accordée n'est inutilisée — vérifiable
en relisant l'historique git des policies.

## Tests

```bash
uv run pytest
```
