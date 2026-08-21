# Terraform — infra permanente — design

**Date :** 2026-08-21
**Sous-projet :** Infrastructure as Code pour l'infra AWS permanente, provisionnée
jusqu'ici manuellement (voir `docs/aws-architecture.md`). Item du CLAUDE.md :
"Terraform complet".

## Contexte

Quatre sous-projets AWS ont été livrés en pilotant l'infra à la main (console/CLI)
ou via des scripts `boto3` (`src/aws/deploy_endpoint.py`, `launch_training.py`,
`deploy_api.py`). Les ressources permanentes (bucket S3, repos ECR, rôles IAM)
existent déjà sur le compte AWS ; les Lambdas et l'API Gateway du sous-projet 4
(`docs/superpowers/specs/2026-08-05-aws-lambda-api-gateway-design.md`) n'ont pas
encore été déployées. Ce sous-projet remplace le pilotage manuel/ad-hoc de
l'infra permanente par Terraform, source de vérité exécutable à la place du
document descriptif `docs/aws-architecture.md`.

## Périmètre

**Dans le périmètre — infra permanente, sans coût au repos :**
- Bucket S3 (`aws-anomalies-mvtec-romeo`)
- Repos ECR (`aws-anomalies-train`, `aws-anomalies-serve`)
- IAM : user `aws-anomalies-local`, rôle `aws-anomalies-sagemaker-execution`,
  et les 2 nouveaux rôles d'exécution Lambda (`predict`, `authorizer`)
- Lambda `predict` et `authorizer`
- API Gateway HTTP API (route `POST /predict`, intégration, authorizer)

**Hors périmètre — ressources éphémères, facturées à l'usage, allumées/éteintes
délibérément par session de test :**
- SageMaker Model / EndpointConfig / Endpoint (reste piloté par
  `src/aws/deploy_endpoint.py`)
- SageMaker Training Job (reste piloté par `src/aws/launch_training.py`)
- `src/aws/deploy_api.py` reste dans le repo comme chemin de déploiement
  alternatif/manuel des Lambdas + API Gateway ; Terraform devient le chemin
  principal recommandé. Les deux ne doivent jamais tourner l'un après l'autre
  sans réconciliation — voir section Risques.

Ce découpage a été validé explicitement avec l'utilisateur : gérer l'endpoint
serverless en "état désiré permanent" via Terraform serait contre-intuitif
pour une ressource qu'on éteint volontairement entre les sessions de test.

## Architecture

```
terraform/
├── providers.tf      # provider aws, région eu-west-1, version pinning
├── variables.tf       # api_key_secret (sensitive), autres variables
├── s3.tf               # bucket (import)
├── ecr.tf              # 2 repos (import)
├── iam.tf              # user + rôle SageMaker (import) + 2 rôles Lambda (nouveaux)
├── lambda.tf            # 2 fonctions Lambda, data.archive_file pour le zip
├── api_gateway.tf        # HTTP API, route, intégration, authorizer, permissions
├── outputs.tf              # URL d'invocation API Gateway
├── terraform.tfvars.example # format attendu, sans valeur réelle (committé)
└── terraform.tfvars          # valeurs réelles, gitignoré
```

Un seul module racine, pas de sous-modules : un seul environnement, une seule
région, conforme au garde-fou du projet et à la convention "pas de
sur-ingénierie avant qu'un deuxième cas d'usage existe" (ici, il n'y en a
qu'un).

## State

Local (`terraform/terraform.tfstate`, gitignoré). Pas de backend S3 distant :
projet solo, un seul contributeur, pas de risque de state lock concurrent.
Le compromis est documenté dans le README (un vrai projet d'équipe utiliserait
un backend S3 + verrouillage DynamoDB).

## Import des ressources existantes

Les ressources créées manuellement entrent dans le state via `terraform
import` (pas de recréation, zéro downtime, zéro risque sur le dataset déjà
uploadé ou les credentials IAM en cours d'usage) :

```bash
terraform import aws_s3_bucket.mvtec aws-anomalies-mvtec-romeo
terraform import aws_ecr_repository.train aws-anomalies-train
terraform import aws_ecr_repository.serve aws-anomalies-serve
terraform import aws_iam_user.local aws-anomalies-local
terraform import aws_iam_role.sagemaker_execution aws-anomalies-sagemaker-execution
```

Après import, `terraform plan` doit afficher zéro changement sur ces
ressources avant tout `apply`. Si le plan montre une divergence (ex. une
policy IAM légèrement différente de ce qui existe réellement), le code
Terraform est ajusté pour matcher l'état réel — jamais l'inverse à ce stade.

## Composants nouveaux (créés directement par Terraform)

- **Rôle d'exécution `predict`** : `sagemaker:InvokeEndpoint` scopé à l'ARN de
  `aws-anomalies-bottle`, + permissions CloudWatch Logs standard scopées au
  log group de la fonction.
- **Rôle d'exécution `authorizer`** : uniquement CloudWatch Logs standard.
- **Lambda `predict`** : runtime Python 3.12, code zippé via
  `data.archive_file` depuis `src/aws/lambda_predict.py`, timeout 30s
  (garde-fou cold start Serverless).
- **Lambda `authorizer`** : runtime Python 3.12, code zippé depuis
  `src/aws/lambda_authorizer.py`, variable d'environnement `API_KEY_SECRET`
  alimentée par `var.api_key_secret`.
- **API Gateway HTTP API** : route unique `POST /predict`, intégration Lambda
  proxy, authorizer Lambda au format simple response, permission
  `lambda:AddPermission` pour chaque Lambda invoquée par API Gateway.

Toutes les permissions IAM scopées aux ARN exacts des ressources concernées,
aucun wildcard `*`, conformément au garde-fou du projet.

## Secret

`api_key_secret` est une variable Terraform `sensitive = true`, fournie via
`terraform/terraform.tfvars` (gitignoré). `terraform.tfvars.example` est
committé avec un placeholder, pour que quelqu'un qui clone le repo sache quoi
renseigner. Jamais de valeur réelle en clair dans le code ou l'historique git.

## Erreurs et risques

- **Double gestion Lambda/API Gateway** (Terraform + `deploy_api.py`) : si
  `deploy_api.py` est relancé après un `terraform apply`, il modifierait des
  ressources gérées par Terraform en dehors de son state, causant une
  divergence détectée au prochain `terraform plan`. Documenté dans le README
  comme piège connu — en pratique, Terraform est le chemin à utiliser une
  fois ce sous-projet livré.
- **Import partiel/échoué** : si un `terraform import` échoue à mi-parcours
  (ex. mauvais ID de ressource), le state reste dans un état incohérent.
  Mitigation : importer une ressource à la fois, vérifier `terraform plan`
  après chaque import avant de continuer.
- **Divergence de policy IAM** : les policies IAM créées à la main peuvent ne
  pas correspondre exactement à ce que `terraform plan` génère par défaut
  (ordre des statements, format). Mitigation : ajuster le HCL pour matcher
  l'existant plutôt que de laisser Terraform proposer une modification non
  désirée sur des rôles déjà en service.

## Tests

Pas de tests automatisés Terraform (disproportionné pour ce portfolio, cf.
raisonnement déjà appliqué à Secrets Manager dans le sous-projet 4).
Vérification manuelle :

1. `terraform validate` — syntaxe et cohérence interne.
2. `terraform plan` après les imports — doit montrer 0 changement sur les
   ressources existantes.
3. `terraform apply` — crée les 2 Lambdas et l'API Gateway.
4. Test E2E réel : redéployer l'endpoint SageMaker (`deploy_endpoint.py`),
   appeler l'URL API Gateway avec `curl` (image + `x-api-key`), vérifier
   `score`/`is_anomaly` contre les valeurs de référence connues (0.9098
   défectueux / 0.2531 sain), vérifier le rejet sans clé (403).
5. Détruire l'endpoint SageMaker après le test (garde-fou du projet). Les
   ressources Terraform (Lambda, API Gateway) n'ont pas de coût au repos et
   peuvent rester en place.

## Hors périmètre

- Backend de state distant (S3 + DynamoDB).
- Modules Terraform réutilisables (un seul environnement à ce stade).
- CI/CD pour appliquer Terraform automatiquement.
- Migration de `deploy_endpoint.py`/`launch_training.py` vers Terraform.
- Suppression de `deploy_api.py`.
