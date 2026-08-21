# Terraform — infra permanente

Gère S3, ECR, IAM (import des ressources existantes + rôles Lambda), les
Lambdas `predict`/`authorizer`, et l'API Gateway HTTP API. Ne gère pas
l'endpoint SageMaker Serverless ni les Training Jobs (restent pilotés par
`src/aws/deploy_endpoint.py` et `src/aws/launch_training.py` — ressources
éphémères, allumées/éteintes à la demande pour maîtriser les coûts).

**Risque connu :** `src/aws/deploy_api.py` reste dans le repo comme chemin
de déploiement manuel alternatif pour les Lambdas/API Gateway. Ne jamais le
relancer après un `terraform apply` : il modifierait des ressources gérées
par Terraform en dehors de son state, ce qui apparaîtrait comme une
divergence au prochain `terraform plan`. Terraform est le chemin à utiliser
en usage normal.

## Prérequis

- Terraform >= 1.5
- `aws-anomalies-local` configuré en local (`aws sts get-caller-identity`)
- Policy étendue selon `docs/superpowers/plans/2026-08-21-terraform-infra.md`, Task 1

## Setup

\`\`\`bash
cd terraform
cp terraform.tfvars.example terraform.tfvars
# éditer terraform.tfvars avec un vrai secret
terraform init
\`\`\`

## State

Local (`terraform.tfstate`, gitignoré). Choix assumé pour un projet solo —
un vrai projet d'équipe utiliserait un backend S3 + verrouillage DynamoDB
pour éviter les écritures concurrentes sur le state.

## Ressources existantes importées

Voir Task 3-4 du plan d'implémentation pour les commandes `terraform
import` exactes. Après import, `terraform plan` doit montrer 0 changement
sur ces ressources avant tout `apply`.

## Déploiement

\`\`\`bash
terraform plan
terraform apply
\`\`\`

## Destruction

Terraform ne gère pas l'endpoint SageMaker (coût continu) — rien à détruire
ici entre les sessions de test. Les Lambdas et l'API Gateway n'ont pas de
coût au repos.
