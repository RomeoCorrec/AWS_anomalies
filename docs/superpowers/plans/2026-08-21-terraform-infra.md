# Terraform infra permanente — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remplacer le pilotage manuel/ad-hoc de l'infra AWS permanente (S3, ECR,
IAM, Lambda, API Gateway) par Terraform, en importont les ressources existantes
sans les recréer, et en créant les Lambdas + API Gateway pas encore déployées.

**Architecture:** un module racine unique `terraform/` avec des fichiers `.tf`
par service (`s3.tf`, `ecr.tf`, `iam.tf`, `lambda.tf`, `api_gateway.tf`), state
local gitignoré, secret via `terraform.tfvars` gitignoré.

**Tech Stack:** Terraform >= 1.5, provider `hashicorp/aws` ~> 5.0, provider
`hashicorp/archive` ~> 2.4 (zip des Lambdas).

**Spec:** `docs/superpowers/specs/2026-08-21-terraform-infra-design.md`

## Global Constraints

- Région unique `eu-west-1` (garde-fou CLAUDE.md).
- Aucun wildcard `*` sur une `Resource` IAM, sauf exception documentée et déjà
  établie dans le projet (`ecr:GetAuthorizationToken` avec `Resource: "*"`,
  imposé par AWS ; API Gateway v2 impose la même contrainte sur les actions de
  création d'API — à documenter de la même façon si rencontrée).
- Timeout Lambda `predict` = 30s minimum (cold start SageMaker Serverless).
- State Terraform local, gitignoré. Secret `api_key_secret` jamais commité.
- Compte AWS : `155466261331`. Bucket S3 : `aws-anomalies-mvtec-romeo`.
  Endpoint SageMaker existant (recréé à la demande) : `aws-anomalies-bottle`.
- Ne jamais recréer une ressource existante (bucket, repos ECR, IAM
  user/role) : toujours importer, jamais laisser `terraform apply` la
  détruire/recréer. Si un `terraform plan` propose de détruire ou remplacer
  une ressource importée, **s'arrêter et ne pas `apply`** avant d'avoir
  compris pourquoi.

---

## Contexte pour l'exécutant : accès IAM actuel

Le seul credential AWS disponible localement est l'utilisateur
`aws-anomalies-local`, volontairement restreint au principe du moindre
privilège (garde-fou du projet). Il n'a **pas** aujourd'hui les permissions
de lecture nécessaires pour que Terraform gère cette infra (vérifié : accès
refusé sur `s3:GetEncryptionConfiguration`, `ecr:DescribeRepositories`,
`iam:GetRole`, `iam:GetUser`, etc). La Task 1 ci-dessous étend sa policy —
cette extension doit être appliquée par l'utilisateur via la console AWS ou
un compte root/admin, l'assistant ne peut pas se l'auto-accorder (et ne le
devrait pas : un utilisateur capable de modifier sa propre policy IAM est un
risque de sécurité classique).

Comme pour chaque sous-projet précédent, cette policy de départ sera
probablement incomplète : si `terraform plan`/`apply` échoue avec
`AccessDenied` sur une action précise, ajouter cette action exacte à la
policy (jamais un wildcard de complaisance) et relancer. C'est la méthode
déjà utilisée sur ce projet (cf. `docs/superpowers/specs/2026-07-29-aws-s3-account-design.md`).

---

### Task 1: Étendre la policy IAM d'`aws-anomalies-local` (étape manuelle utilisateur)

**Files:**
- Aucun fichier repo modifié — action console/CLI AWS par l'utilisateur, avec
  des credentials plus privilégiés que `aws-anomalies-local`.

- [ ] **Step 1: Attacher la policy inline suivante à `aws-anomalies-local`**

Nom suggéré : `aws-anomalies-terraform-bootstrap`. JSON à coller dans la
console IAM (ou via `aws iam put-user-policy` avec un profil admin) :

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "S3BucketConfig",
      "Effect": "Allow",
      "Action": [
        "s3:GetBucketLocation",
        "s3:GetBucketPolicy",
        "s3:GetBucketVersioning",
        "s3:GetBucketTagging",
        "s3:PutBucketTagging",
        "s3:GetEncryptionConfiguration",
        "s3:PutEncryptionConfiguration",
        "s3:GetBucketPublicAccessBlock",
        "s3:PutBucketPublicAccessBlock"
      ],
      "Resource": "arn:aws:s3:::aws-anomalies-mvtec-romeo"
    },
    {
      "Sid": "EcrRead",
      "Effect": "Allow",
      "Action": [
        "ecr:DescribeRepositories",
        "ecr:ListTagsForResource",
        "ecr:GetRepositoryPolicy"
      ],
      "Resource": [
        "arn:aws:ecr:eu-west-1:155466261331:repository/aws-anomalies-train",
        "arn:aws:ecr:eu-west-1:155466261331:repository/aws-anomalies-serve"
      ]
    },
    {
      "Sid": "IamReadExisting",
      "Effect": "Allow",
      "Action": [
        "iam:GetUser",
        "iam:ListUserPolicies",
        "iam:ListAttachedUserPolicies",
        "iam:GetRole",
        "iam:ListRolePolicies",
        "iam:ListAttachedRolePolicies",
        "iam:ListInstanceProfilesForRole"
      ],
      "Resource": [
        "arn:aws:iam::155466261331:user/aws-anomalies-local",
        "arn:aws:iam::155466261331:role/aws-anomalies-sagemaker-execution"
      ]
    },
    {
      "Sid": "IamManageNewLambdaRoles",
      "Effect": "Allow",
      "Action": [
        "iam:CreateRole",
        "iam:GetRole",
        "iam:DeleteRole",
        "iam:TagRole",
        "iam:PutRolePolicy",
        "iam:GetRolePolicy",
        "iam:DeleteRolePolicy",
        "iam:ListRolePolicies",
        "iam:ListAttachedRolePolicies"
      ],
      "Resource": [
        "arn:aws:iam::155466261331:role/aws-anomalies-predict-lambda-execution",
        "arn:aws:iam::155466261331:role/aws-anomalies-authorizer-lambda-execution"
      ]
    },
    {
      "Sid": "LambdaManage",
      "Effect": "Allow",
      "Action": [
        "lambda:GetFunction",
        "lambda:CreateFunction",
        "lambda:UpdateFunctionCode",
        "lambda:UpdateFunctionConfiguration",
        "lambda:DeleteFunction",
        "lambda:GetPolicy",
        "lambda:AddPermission",
        "lambda:RemovePermission",
        "lambda:TagResource",
        "lambda:ListTags"
      ],
      "Resource": [
        "arn:aws:lambda:eu-west-1:155466261331:function:aws-anomalies-predict",
        "arn:aws:lambda:eu-west-1:155466261331:function:aws-anomalies-authorizer"
      ]
    },
    {
      "Sid": "ApiGatewayManage",
      "Effect": "Allow",
      "Action": [
        "apigateway:GET",
        "apigateway:POST",
        "apigateway:PUT",
        "apigateway:PATCH",
        "apigateway:DELETE"
      ],
      "Resource": "arn:aws:apigateway:eu-west-1::/apis*"
    }
  ]
}
```

Note sur `ApiGatewayManage` : API Gateway v2 exige `Resource:
"arn:aws:apigateway:{region}::/apis*"` (avec le wildcard de fin, seule forme
supportée par le service pour ces actions — au même titre que
`ecr:GetAuthorizationToken` avec `Resource: "*"`, c'est une contrainte
imposée par AWS, pas un relâchement de notre part). Documenter cette
exception dans `docs/aws-architecture.md` à la Task 7.

- [ ] **Step 2: Vérifier l'accès**

```bash
aws s3api get-bucket-encryption --bucket aws-anomalies-mvtec-romeo
aws ecr describe-repositories --repository-names aws-anomalies-train aws-anomalies-serve
aws iam get-role --role-name aws-anomalies-sagemaker-execution
```

Expected: les 3 commandes répondent sans `AccessDenied`.

- [ ] **Step 3: Commit** — rien à committer dans le repo pour cette task
  (action AWS pure). Passer à la Task 2.

---

### Task 2: Scaffolding du module Terraform

**Files:**
- Create: `terraform/providers.tf`
- Create: `terraform/variables.tf`
- Create: `terraform/outputs.tf`
- Create: `terraform/terraform.tfvars.example`
- Create: `terraform/README.md`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `var.aws_region` (default `"eu-west-1"`), `var.account_id`
  (default `"155466261331"`), `var.api_key_secret` (sensitive, pas de
  défaut) — consommées par les tasks suivantes.

- [ ] **Step 1: Créer `terraform/providers.tf`**

```hcl
terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    archive = {
      source  = "hashicorp/archive"
      version = "~> 2.4"
    }
  }
}

provider "aws" {
  region = var.aws_region
}
```

- [ ] **Step 2: Créer `terraform/variables.tf`**

```hcl
variable "aws_region" {
  description = "Région AWS unique du projet."
  type        = string
  default     = "eu-west-1"
}

variable "account_id" {
  description = "ID du compte AWS du projet."
  type        = string
  default     = "155466261331"
}

variable "api_key_secret" {
  description = "Secret x-api-key attendu par la Lambda authorizer. Jamais commité."
  type        = string
  sensitive   = true
}
```

- [ ] **Step 3: Créer `terraform/outputs.tf` (vide pour l'instant, complété Task 6)**

```hcl
# Complété à la Task 7 avec l'URL d'invocation de l'API Gateway.
```

- [ ] **Step 4: Créer `terraform/terraform.tfvars.example`**

```hcl
api_key_secret = "remplace-moi-par-un-secret-fort"
```

- [ ] **Step 5: Ajouter les entrées Terraform à `.gitignore`**

Ouvrir `.gitignore` et ajouter à la fin :

```
terraform/.terraform/
terraform/terraform.tfstate
terraform/terraform.tfstate.backup
terraform/terraform.tfvars
terraform/*.tfplan
terraform/build/
```

- [ ] **Step 6: Créer `terraform/README.md`**

```markdown
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
```

- [ ] **Step 7: Commit**

```bash
cd "C:\Users\romeo\Documents\Projets_ENS\AWS_anomalies"
git add terraform/providers.tf terraform/variables.tf terraform/outputs.tf terraform/terraform.tfvars.example terraform/README.md .gitignore
git commit -m "chore: scaffold Terraform module for permanent AWS infra"
```

---

### Task 3: S3 — déclarer et importer le bucket

**Files:**
- Create: `terraform/s3.tf`

**Interfaces:**
- Consumes: aucune (ressource racine).
- Produces: `aws_s3_bucket.mvtec` — référencé nulle part ailleurs (aucune
  autre ressource ne dépend du bucket dans ce module).

- [ ] **Step 1: Créer `terraform/s3.tf`**

```hcl
resource "aws_s3_bucket" "mvtec" {
  bucket = "aws-anomalies-mvtec-romeo"
}

resource "aws_s3_bucket_public_access_block" "mvtec" {
  bucket = aws_s3_bucket.mvtec.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "mvtec" {
  bucket = aws_s3_bucket.mvtec.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}
```

- [ ] **Step 2: Init et import**

```bash
cd terraform
terraform init
terraform import aws_s3_bucket.mvtec aws-anomalies-mvtec-romeo
terraform import aws_s3_bucket_public_access_block.mvtec aws-anomalies-mvtec-romeo
terraform import aws_s3_bucket_server_side_encryption_configuration.mvtec aws-anomalies-mvtec-romeo
```

- [ ] **Step 3: Vérifier qu'il n'y a aucun changement**

```bash
terraform plan
```

Expected: `No changes.` pour les 3 ressources S3. Si un changement apparaît
(ex. `block_public_acls` différent), ajuster le HCL du Step 1 pour matcher
l'état réel affiché dans le plan — ne jamais laisser un `apply` modifier une
config de sécurité existante sans l'avoir compris.

- [ ] **Step 4: Commit**

```bash
cd "C:\Users\romeo\Documents\Projets_ENS\AWS_anomalies"
git add terraform/s3.tf
git commit -m "feat(terraform): import existing S3 bucket"
```

---

### Task 4: ECR — déclarer et importer les 2 repos

**Files:**
- Create: `terraform/ecr.tf`

**Interfaces:**
- Produces: `aws_ecr_repository.train`, `aws_ecr_repository.serve` —
  référencées nulle part ailleurs dans ce module (les images sont pushées
  par les scripts existants, pas par Terraform).

- [ ] **Step 1: Créer `terraform/ecr.tf`**

```hcl
resource "aws_ecr_repository" "train" {
  name                 = "aws-anomalies-train"
  image_tag_mutability = "MUTABLE"
}

resource "aws_ecr_repository" "serve" {
  name                 = "aws-anomalies-serve"
  image_tag_mutability = "MUTABLE"
}
```

- [ ] **Step 2: Import**

```bash
terraform import aws_ecr_repository.train aws-anomalies-train
terraform import aws_ecr_repository.serve aws-anomalies-serve
```

- [ ] **Step 3: Vérifier**

```bash
terraform plan
```

Expected: `No changes.` sur les 2 repos ECR (en plus des ressources S3 déjà
propres). Si `image_tag_mutability` diffère, ajuster le HCL pour matcher
l'existant.

- [ ] **Step 4: Commit**

```bash
cd "C:\Users\romeo\Documents\Projets_ENS\AWS_anomalies"
git add terraform/ecr.tf
git commit -m "feat(terraform): import existing ECR repositories"
```

---

### Task 5: IAM — importer user/rôle existants, créer les rôles Lambda

**Files:**
- Create: `terraform/iam.tf`

**Interfaces:**
- Produces: `aws_iam_role.predict_lambda.arn`, `aws_iam_role.authorizer_lambda.arn`
  — consommées par Task 6 (Lambda).
- Consumes: `var.account_id`, `var.aws_region` (Task 2).

Décision de scope : `aws_iam_user.local` et `aws_iam_role.sagemaker_execution`
sont importés en tant qu'objets seulement (nom, trust policy pour le rôle) —
leurs policies de permission (étendues manuellement sur 4 sous-projets) ne
sont **pas** redéclarées ici. Deux raisons : (1) leur contenu exact n'est pas
connu avec certitude sans introspection complète, et deviner risquerait de
retirer une permission dont un script existant dépend ; (2) modifier via
Terraform la policy de l'utilisateur qui exécute Terraform lui-même est un
risque classique d'auto-verrouillage. Comme ces policies ne sont déclarées
dans aucune ressource `.tf`, Terraform ne les touchera pas.

- [ ] **Step 1: Créer `terraform/iam.tf`**

```hcl
# --- Ressources existantes, importées (objets seulement, pas leurs policies) ---

resource "aws_iam_user" "local" {
  name = "aws-anomalies-local"
}

resource "aws_iam_role" "sagemaker_execution" {
  name = "aws-anomalies-sagemaker-execution"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect    = "Allow"
        Principal = { Service = "sagemaker.amazonaws.com" }
        Action    = "sts:AssumeRole"
      }
    ]
  })
}

# --- Nouveaux rôles pour les Lambdas ---

data "aws_iam_policy_document" "lambda_assume_role" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "predict_lambda" {
  name               = "aws-anomalies-predict-lambda-execution"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume_role.json
}

resource "aws_iam_role_policy" "predict_lambda" {
  name = "aws-anomalies-predict-lambda-policy"
  role = aws_iam_role.predict_lambda.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "InvokeEndpoint"
        Effect   = "Allow"
        Action   = "sagemaker:InvokeEndpoint"
        Resource = "arn:aws:sagemaker:${var.aws_region}:${var.account_id}:endpoint/aws-anomalies-bottle"
      },
      {
        Sid    = "Logs"
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
        ]
        Resource = "arn:aws:logs:${var.aws_region}:${var.account_id}:log-group:/aws/lambda/aws-anomalies-predict:*"
      },
    ]
  })
}

resource "aws_iam_role" "authorizer_lambda" {
  name               = "aws-anomalies-authorizer-lambda-execution"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume_role.json
}

resource "aws_iam_role_policy" "authorizer_lambda" {
  name = "aws-anomalies-authorizer-lambda-policy"
  role = aws_iam_role.authorizer_lambda.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "Logs"
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
        ]
        Resource = "arn:aws:logs:${var.aws_region}:${var.account_id}:log-group:/aws/lambda/aws-anomalies-authorizer:*"
      },
    ]
  })
}
```

- [ ] **Step 2: Import des 2 ressources existantes**

```bash
terraform import aws_iam_user.local aws-anomalies-local
terraform import aws_iam_role.sagemaker_execution aws-anomalies-sagemaker-execution
```

- [ ] **Step 3: Vérifier**

```bash
terraform plan
```

Expected: `No changes` sur `aws_iam_user.local` et
`aws_iam_role.sagemaker_execution` ; le plan propose la **création** de
`aws_iam_role.predict_lambda`, `aws_iam_role_policy.predict_lambda`,
`aws_iam_role.authorizer_lambda`, `aws_iam_role_policy.authorizer_lambda`
(nouvelles ressources, pas encore créées). C'est attendu à ce stade — elles
seront réellement créées à l'`apply` final (Task 7).

- [ ] **Step 4: Commit**

```bash
cd "C:\Users\romeo\Documents\Projets_ENS\AWS_anomalies"
git add terraform/iam.tf
git commit -m "feat(terraform): import IAM user/role, declare Lambda execution roles"
```

---

### Task 6: Lambda — fonctions predict et authorizer

**Files:**
- Create: `terraform/lambda.tf`

**Interfaces:**
- Consumes: `aws_iam_role.predict_lambda.arn`, `aws_iam_role.authorizer_lambda.arn`
  (Task 5).
- Produces: `aws_lambda_function.predict.invoke_arn`,
  `aws_lambda_function.predict.function_name`,
  `aws_lambda_function.authorizer.invoke_arn`,
  `aws_lambda_function.authorizer.function_name` — consommées par Task 7
  (API Gateway).

- [ ] **Step 1: Créer `terraform/lambda.tf`**

```hcl
data "archive_file" "predict" {
  type        = "zip"
  source_file = "${path.module}/../src/aws/lambda_predict.py"
  output_path = "${path.module}/build/lambda_predict.zip"
}

resource "aws_lambda_function" "predict" {
  function_name = "aws-anomalies-predict"
  runtime       = "python3.12"
  handler       = "lambda_predict.handler"
  role          = aws_iam_role.predict_lambda.arn
  timeout       = 30

  filename         = data.archive_file.predict.output_path
  source_code_hash = data.archive_file.predict.output_base64sha256

  environment {
    variables = {
      SAGEMAKER_ENDPOINT_NAME = "aws-anomalies-bottle"
    }
  }
}

data "archive_file" "authorizer" {
  type        = "zip"
  source_file = "${path.module}/../src/aws/lambda_authorizer.py"
  output_path = "${path.module}/build/lambda_authorizer.zip"
}

resource "aws_lambda_function" "authorizer" {
  function_name = "aws-anomalies-authorizer"
  runtime       = "python3.12"
  handler       = "lambda_authorizer.handler"
  role          = aws_iam_role.authorizer_lambda.arn
  timeout       = 10

  filename         = data.archive_file.authorizer.output_path
  source_code_hash = data.archive_file.authorizer.output_base64sha256

  environment {
    variables = {
      API_KEY = var.api_key_secret
    }
  }
}
```

Ne pas fixer de variable `AWS_REGION` ici : c'est un nom d'environnement
réservé par Lambda (déjà corrigé dans `deploy_api.py`, cf. commit
`bbf970b`) — `lambda_predict.py` retombe sur son défaut `"eu-west-1"` en son
absence.

- [ ] **Step 2: Valider et vérifier le plan**

```bash
terraform validate
terraform plan
```

Expected: `terraform validate` répond `Success`. `terraform plan` propose la
création des 2 fonctions Lambda (pas encore déployées, cf. réponse à la
question de cadrage), sans toucher aux ressources déjà importées.

- [ ] **Step 3: Commit**

```bash
cd "C:\Users\romeo\Documents\Projets_ENS\AWS_anomalies"
git add terraform/lambda.tf
git commit -m "feat(terraform): declare predict and authorizer Lambda functions"
```

---

### Task 7: API Gateway, apply final, et validation E2E

**Files:**
- Create: `terraform/api_gateway.tf`
- Modify: `terraform/outputs.tf`
- Modify: `docs/aws-architecture.md`

**Interfaces:**
- Consumes: `aws_lambda_function.predict.invoke_arn`/`function_name`,
  `aws_lambda_function.authorizer.invoke_arn`/`function_name` (Task 6).

- [ ] **Step 1: Créer `terraform/api_gateway.tf`**

```hcl
resource "aws_apigatewayv2_api" "main" {
  name          = "aws-anomalies-api"
  protocol_type = "HTTP"
}

resource "aws_apigatewayv2_integration" "predict" {
  api_id                 = aws_apigatewayv2_api.main.id
  integration_type       = "AWS_PROXY"
  integration_uri        = aws_lambda_function.predict.invoke_arn
  integration_method     = "POST"
  payload_format_version = "2.0"
}

resource "aws_apigatewayv2_authorizer" "api_key" {
  api_id                            = aws_apigatewayv2_api.main.id
  authorizer_type                   = "REQUEST"
  name                               = "api-key-authorizer"
  authorizer_uri                    = aws_lambda_function.authorizer.invoke_arn
  authorizer_payload_format_version = "2.0"
  enable_simple_responses           = true
  identity_sources                  = ["$request.header.x-api-key"]
}

resource "aws_apigatewayv2_route" "predict" {
  api_id             = aws_apigatewayv2_api.main.id
  route_key          = "POST /predict"
  target             = "integrations/${aws_apigatewayv2_integration.predict.id}"
  authorization_type = "CUSTOM"
  authorizer_id      = aws_apigatewayv2_authorizer.api_key.id
}

resource "aws_apigatewayv2_stage" "default" {
  api_id      = aws_apigatewayv2_api.main.id
  name        = "$default"
  auto_deploy = true
}

resource "aws_lambda_permission" "predict" {
  statement_id  = "apigateway-invoke-predict"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.predict.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.main.execution_arn}/*/*/predict"
}

resource "aws_lambda_permission" "authorizer" {
  statement_id  = "apigateway-invoke-authorizer"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.authorizer.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.main.execution_arn}/authorizers/${aws_apigatewayv2_authorizer.api_key.id}"
}
```

- [ ] **Step 2: Compléter `terraform/outputs.tf`**

```hcl
output "predict_invoke_url" {
  description = "URL d'invocation de POST /predict."
  value       = "${aws_apigatewayv2_api.main.api_endpoint}/predict"
}
```

- [ ] **Step 3: Plan puis apply**

```bash
terraform validate
terraform plan
```

Expected: seules des créations (API Gateway, route, intégration, authorizer,
stage, 2 permissions Lambda, 2 rôles Lambda + policies de la Task 5, 2
fonctions Lambda de la Task 6). Aucune destruction/remplacement sur les
ressources importées (S3, ECR, IAM user/role). Si c'est le cas :

```bash
terraform apply
```

Confirmer avec `yes` à l'invite. Si une erreur `AccessDenied` apparaît,
identifier l'action manquante dans le message d'erreur et l'ajouter à la
policy `aws-anomalies-terraform-bootstrap` de la Task 1 (une action précise
à la fois, jamais un wildcard), puis relancer `terraform apply`.

- [ ] **Step 4: Validation E2E réelle**

Redéployer l'endpoint SageMaker (ressource hors périmètre Terraform,
toujours pilotée par le script existant) :

```bash
cd "C:\Users\romeo\Documents\Projets_ENS\AWS_anomalies"
python -m src.aws.deploy_endpoint
```

Récupérer l'URL Terraform et tester :

```bash
cd terraform
INVOKE_URL=$(terraform output -raw predict_invoke_url)
cd ..
curl -s -X POST "$INVOKE_URL" \
  -H "x-api-key: <valeur de terraform.tfvars>" \
  -H "Content-Type: image/png" \
  --data-binary @data/mvtec/bottle/test/broken_large/000.png
```

Expected: réponse JSON avec un score proche de la référence connue (0.9098
sur une image défectueuse `bottle`). Puis vérifier le rejet sans clé :

```bash
curl -s -o /dev/null -w "%{http_code}\n" -X POST "$INVOKE_URL" \
  -H "Content-Type: image/png" \
  --data-binary @data/mvtec/bottle/test/broken_large/000.png
```

Expected: `403`.

- [ ] **Step 5: Détruire l'endpoint SageMaker (garde-fou du projet)**

```bash
python -m src.aws.deploy_endpoint --destroy
```

(ou la commande équivalente déjà utilisée dans
`docs/superpowers/plans/2026-07-31-aws-serverless-endpoint.md`, Task 6).
Lambda et API Gateway n'ont pas de coût au repos — peuvent rester en place.

- [ ] **Step 6: Mettre à jour `docs/aws-architecture.md`**

Dans la section "Sous-projet 4" (ou en créer une si absente, en suivant le
même format que les sous-projets 1-3), ajouter une ligne indiquant que
l'infra permanente (S3, ECR, IAM, Lambda, API Gateway) est désormais gérée
par Terraform (`terraform/`), avec un renvoi vers
`docs/superpowers/specs/2026-08-21-terraform-infra-design.md`. Retirer
"Terraform (infra actuellement provisionnée manuellement...)" de la section
"Hors périmètre" du document, puisque ce n'est plus vrai.

- [ ] **Step 7: Commit**

```bash
git add terraform/api_gateway.tf terraform/outputs.tf docs/aws-architecture.md
git commit -m "feat(terraform): add API Gateway HTTP API, complete permanent infra"
```
