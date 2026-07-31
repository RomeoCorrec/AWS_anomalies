# Phase AWS — Entraînement SageMaker

Date : 2026-07-29

## Contexte

Deuxième sous-projet de la phase AWS (cf. `2026-07-29-aws-s3-account-design.md` pour le
premier : compte, IAM, bucket S3, dataset `bottle` uploadé). Ce sous-projet couvre
l'entraînement de PatchCore sur `bottle` via un **SageMaker Training Job**, avec un
container custom (BYOC — Bring Your Own Container), conformément à l'architecture cible
du CLAUDE.md (S3 → SageMaker Training Job → ECR).

Garde-fous applicables : région unique `eu-west-1`, moindre privilège IAM (extension
incrémentale, jamais de wildcard), instance facturée à l'heure d'usage signalée avant
création (`ml.m5.xlarge`, ~0.23 $/h en eu-west-1).

## Décisions retenues

- **Approche training** : container custom (BYOC) plutôt que Script Mode — réutilise
  l'expérience acquise avec le `Dockerfile` d'inférence local et correspond à
  l'architecture cible du projet.
- **Instance** : `ml.m5.xlarge` (4 vCPU / 16 Go, CPU seulement — cohérent avec l'approche
  CPU-first du reste du projet). Facturée uniquement pour la durée du job.
- **Catégorie** : `bottle` uniquement (même incrément minimal que les sous-projets
  précédents).
- **Déclenchement** : SDK Python `sagemaker` (nouvelle dépendance), pas la CLI brute —
  cohérent avec le reste du projet, déjà piloté par des scripts Python paramétrables.
- **Réutilisation de code** : le nouveau point d'entrée SageMaker importe
  `build_datamodule`/`build_model` de `src/models/train.py`, exactement comme
  `src/models/deploy.py` — aucune divergence entre le chemin d'entraînement local et le
  chemin SageMaker au niveau de la construction du modèle.

## Architecture

### Container de training (`Dockerfile.sagemaker-train`, nouveau)

- Base identique au `Dockerfile` d'inférence existant (`python:3.10-slim` + `uv sync
  --frozen --no-dev`).
- Embarque `src/`, `config/` (y compris les fichiers d'expérience).
- `ENTRYPOINT` sur un nouveau script `src/aws/train_entrypoint.py`, pas sur
  `src/models/train.py` directement — les deux diffèrent sur l'origine des données et la
  destination du checkpoint (voir ci-dessous) ; réutilisation via import, pas
  duplication.

### `src/aws/train_entrypoint.py` (nouveau)

1. Lit le chemin de la config d'expérience à utiliser (hyperparamètre SageMaker,
   ex. `config/experiment/bottle_wideresnet50.yaml`, embarqué dans l'image).
2. Charge la config via `load_experiment_config`, puis **surcharge `cfg.root`** avec le
   chemin fixe `/opt/ml/input/data/training` (`DEFAULT_DATA_ROOT`), où SageMaker monte les
   données du canal `training` — le contenu S3 `mvtec/` — dans le container. Ce chemin
   n'est **pas** fourni par une variable d'environnement : `SM_CHANNEL_TRAINING` n'est
   injectée que par le SageMaker Training Toolkit, absent de ce container BYOC minimal ;
   `os.environ.get("SM_CHANNEL_TRAINING", ...)` n'existe que comme filet de repli utilisé
   par les tests locaux, la production passe toujours par le chemin fixe. C'est la seule
   différence de config avec un entraînement local.
3. Appelle `build_datamodule(cfg)` / `build_model(cfg)` (de `train.py`, importés tels
   quels) puis `Engine().fit(...)`.
4. Copie le checkpoint résultant (même mécanisme que `deploy_checkpoint`, via
   `engine.trainer.checkpoint_callback.best_model_path`) vers `SM_MODEL_DIR`
   (`/opt/ml/model` par défaut) sous le nom `model.ckpt`. SageMaker compresse
   automatiquement le contenu de ce dossier en `model.tar.gz` et l'upload vers le chemin
   S3 de sortie du job à la fin de l'exécution.

### IAM — deux identités distinctes

- **`aws-anomalies-local`** (existant, étendu) : nouvelles permissions ajoutées à sa
  policy existante (pas de nouvelle policy séparée) :
  - ECR : `ecr:GetAuthorizationToken`, `ecr:BatchCheckLayerAvailability`,
    `ecr:PutImage`, `ecr:InitiateLayerUpload`, `ecr:UploadLayerPart`,
    `ecr:CompleteLayerUpload`, `ecr:CreateRepository` (une fois, pour créer le repo),
    scopées à l'ARN du repo `aws-anomalies-train`.
  - SageMaker : `sagemaker:CreateTrainingJob`, `sagemaker:DescribeTrainingJob`,
    `sagemaker:StopTrainingJob`, scopées par condition de nom de ressource
    (`aws-anomalies-*`) si le support de policy le permet, sinon documentée comme
    exception explicite (SageMaker ne supporte pas toujours le scoping par ARN sur
    `CreateTrainingJob`).
  - `iam:PassRole`, restreint à l'ARN du rôle `aws-anomalies-sagemaker-execution`
    ci-dessous (empêche `aws-anomalies-local` de faire passer n'importe quel rôle à
    SageMaker).
- **Nouveau rôle `aws-anomalies-sagemaker-execution`** (assumé par le service
  SageMaker, jamais par un utilisateur) :
  - Trust policy : principal `sagemaker.amazonaws.com`.
  - Permissions : lecture sur `s3://aws-anomalies-mvtec-romeo/mvtec/*`, écriture
    sur `s3://aws-anomalies-mvtec-romeo/output/*` (chemin de sortie des jobs), pull sur le
    repo ECR `aws-anomalies-train`, écriture de logs CloudWatch
    (`logs:CreateLogGroup`/`CreateLogStream`/`PutLogEvents`, scopé au groupe de logs
    SageMaker).
  - Créé via la console IAM par l'utilisateur (comme le compte et l'utilisateur IAM du
    premier sous-projet) — l'assistant fournit le JSON exact des policies et de la trust
    policy.

### ECR

- Nouveau repo `aws-anomalies-train`, région `eu-west-1`.
- Build de `Dockerfile.sagemaker-train` en local, tag vers l'URI ECR, push après
  authentification (`aws ecr get-login-password | docker login`).

### Déclenchement (`src/aws/launch_training.py`, nouveau)

- Dépendance ajoutée : `sagemaker` (SDK Python).
- Construit un `sagemaker.estimator.Estimator` : image = URI ECR poussée, rôle =
  `aws-anomalies-sagemaker-execution`, `instance_type="ml.m5.xlarge"`,
  `instance_count=1`, canal d'entrée `training` = URI S3
  `s3://aws-anomalies-mvtec-romeo/mvtec/` (parent de `bottle/`, pour que le datamodule
  retrouve `bottle/` en sous-répertoire du root monté), hyperparamètre = chemin de la
  config d'expérience.
- `.fit(...)`, bloquant jusqu'à la fin du job (logs streamés dans le terminal via le SDK).
- Imprime l'URI S3 du `model.tar.gz` produit.
- CLI : `python -m src.aws.launch_training --experiment config/experiment/bottle_wideresnet50.yaml`.

## Vérification

1. Le job SageMaker doit se terminer avec le statut `Completed` (pas juste lancé sans
   erreur).
2. Téléchargement du `model.tar.gz` résultant en local (`aws s3 cp`), extraction du
   `model.ckpt`.
3. **Ce checkpoint doit être chargeable par `AnomalyDetector` existant** (même classe que
   pour le déploiement local) et produire un score cohérent sur une image de test connue
   (`data/mvtec/bottle/test/broken_large/000.png`, déjà utilisée comme cas de vérification
   au sous-projet précédent) — preuve que l'entraînement cloud produit un artefact
   utilisable par le code d'inférence déjà écrit, pas seulement qu'un job a un statut vert.

## Hors périmètre (sous-projets suivants)

- SageMaker Serverless Inference (utilisera ce même `model.tar.gz` ou un réentraînement
  ultérieur).
- Lambda + API Gateway.
- Terraform complet.
- Entraînement `screw`/`carpet` sur SageMaker (viendra si besoin, même mécanisme,
  nouveau fichier d'expérience).
- **Risque de coût à anticiper** : le canal SageMaker pointe sur le préfixe partagé
  `mvtec/` plutôt que sur un préfixe spécifique à `bottle`. Un canal SageMaker télécharge
  toutes les clés sous le préfixe donné — aujourd'hui seul `bottle` s'y trouve, donc c'est
  gratuit, mais dès que `screw` ou `carpet` sera uploadé sous `mvtec/`, chaque job
  d'entraînement `bottle` téléchargera aussi les autres catégories, triplant le temps de
  transfert et le coût facturé sans bénéfice. À ce moment-là, donner à chaque catégorie
  son propre préfixe de canal (ex. `s3://…/channels/bottle/bottle/`) plutôt que de
  partager le préfixe parent.

## Tests prévus

- `tests/test_train_entrypoint.py` : vérifie que le point d'entrée surcharge bien
  `cfg.root` depuis une variable d'environnement factice et copie le checkpoint vers le
  répertoire `SM_MODEL_DIR` attendu — `Engine`, `build_datamodule`, `build_model`
  monkeypatchés (même style que `tests/test_deploy.py`), aucun accès réseau ni
  entraînement réel dans ce test.
- Pas de test automatisé pour `launch_training.py` ni pour l'infra IAM/ECR
  (provisioning, pas de logique applicative) — vérification par exécution réelle décrite
  ci-dessus.
