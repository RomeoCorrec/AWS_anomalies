# Architecture AWS — état actuel

Récapitulatif factuel des ressources AWS créées pour ce projet, à date. Document
généré automatiquement à partir des specs/plans de chaque sous-projet — à
tenir à jour manuellement si des ressources changent hors de ce workflow.

**Région unique : `eu-west-1`**
**Compte AWS : `155466261331`**

## Vue d'ensemble

```
S3 (mvtec/bottle/)
   │
   ▼
SageMaker Training Job (BYOC, ECR: aws-anomalies-train)
   │
   ▼
S3 (output/.../model.tar.gz)
   │
   ▼
SageMaker Model → EndpointConfig (Serverless) → Endpoint
   (BYOC, ECR: aws-anomalies-serve)
   │
   ▼
HTTP /invocations → {"score": float, "is_anomaly": bool}
```

Le sous-projet suivant (Lambda + API Gateway, pas commencé) invoquera cet
endpoint depuis une Lambda exposée via API Gateway.

## Sous-projet 1 — S3 et compte AWS

**Spec :** `docs/superpowers/specs/2026-07-29-aws-s3-account-design.md`

| Ressource | Détail |
|---|---|
| Bucket S3 | `aws-anomalies-mvtec-romeo` — public access block activé, chiffrement SSE-S3 |
| Structure S3 | `mvtec/bottle/` (dataset uploadé), `output/` (artefacts de training) |
| Utilisateur IAM | `aws-anomalies-local` — utilisé pour piloter le projet via CLI/SDK depuis le poste local |
| Budget | Alerte de dépenses à 5 $/mois |

## Sous-projet 2 — SageMaker Training

**Spec :** `docs/superpowers/specs/2026-07-29-aws-sagemaker-training-design.md`
**Plan :** `docs/superpowers/plans/2026-07-29-aws-sagemaker-training.md`

| Ressource | Détail |
|---|---|
| Repo ECR | `aws-anomalies-train` — image BYOC de training |
| Rôle IAM | `aws-anomalies-sagemaker-execution` — assumé par `sagemaker.amazonaws.com`, lit `mvtec/*`, écrit `output/*`, pull ECR training, écrit les logs CloudWatch `/aws/sagemaker/TrainingJobs/*` |
| Code | `src/aws/train_entrypoint.py` (point d'entrée container), `src/aws/launch_training.py` (launcher SDK `sagemaker`) |
| Instance | `ml.m5.xlarge` (facturée à l'heure d'utilisation, job ponctuel) |
| Résultat | `model.tar.gz` sur `s3://aws-anomalies-mvtec-romeo/output/aws-anomalies-train-2026-07-31-09-04-02-108/output/model.tar.gz`, catégorie `bottle` |

Un Training Job ne laisse aucune ressource facturée après `Completed` — pas
de nettoyage requis pour ce sous-projet.

## Sous-projet 3 — SageMaker Serverless Inference Endpoint

**Spec :** `docs/superpowers/specs/2026-07-31-aws-serverless-endpoint-design.md`
**Plan :** `docs/superpowers/plans/2026-07-31-aws-serverless-endpoint.md`

| Ressource | Détail |
|---|---|
| Repo ECR | `aws-anomalies-serve` — image BYOC de serving |
| Rôle IAM | `aws-anomalies-sagemaker-execution` (étendu) — pull ECR serving, lit `output/*` (model.tar.gz), écrit les logs CloudWatch `/aws/sagemaker/Endpoints/*` |
| Code | `src/aws/serve.py` (serveur Flask, routes `/ping` et `/invocations`), `src/aws/deploy_endpoint.py` (launcher SDK `sagemaker`, `Model.deploy()`) |
| Config Serverless | `memory_size_in_mb=4096`, `max_concurrency=1` (2048 MB s'est révélé insuffisant en test réel) |
| État actuel | **Détruit** — endpoint `aws-anomalies-bottle` créé, invoqué avec succès (score 0.9098 sur image défectueuse, 0.2531 sur image saine), puis supprimé (`delete-endpoint` + `delete-endpoint-config`) conformément au garde-fou du projet |

Un endpoint Serverless facture en continu tant qu'il existe — contrairement
au Training Job, il doit être explicitement recréé pour être retesté (voir
commande dans `docs/superpowers/plans/2026-07-31-aws-serverless-endpoint.md`,
Task 6).

## IAM — permissions actuelles

Principe du moindre privilège tout du long : aucune permission `*` sur les
ressources, sauf `ecr:GetAuthorizationToken` (imposé par AWS). Les
permissions temporaires (`ecr:CreateRepository`, lecture de logs CloudWatch
pour debug) ont été ajoutées puis retirées une fois leur usage terminé.

- **`aws-anomalies-local`** (utilisateur, pilote le projet) : S3 (lecture
  `mvtec/*`, écriture `output/*`), ECR (pull/push sur les deux repos),
  SageMaker (Create/Describe/Delete Model/EndpointConfig/Endpoint et
  TrainingJob, scopés aux ressources `aws-anomalies-*`), `iam:PassRole`
  restreint à l'ARN du rôle d'exécution.
- **`aws-anomalies-sagemaker-execution`** (rôle, assumé par le service
  SageMaker) : S3, ECR pull, CloudWatch Logs — voir détail par sous-projet
  ci-dessus.

## Hors périmètre (pas encore construit)

- Lambda + API Gateway (prochain sous-projet)
- Terraform (infra actuellement provisionnée manuellement via console/CLI)
- Monitoring CloudWatch dédié (alarmes, dashboards)
- Support d'autres catégories que `bottle` (`carpet`, `screw`)
