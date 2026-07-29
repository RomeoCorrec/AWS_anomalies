# Phase AWS — S3 et compte AWS

Date : 2026-07-29

## Contexte

Première tranche de la phase AWS (S3, SageMaker, ECR, Serverless Inference, Lambda,
API Gateway, Terraform — cf. architecture cible du CLAUDE.md). La phase est trop large
pour un seul spec/plan : elle est découpée en sous-projets suivant l'ordre de la
checklist du projet. Ce spec couvre uniquement le premier sous-projet : **compte AWS,
utilisateur IAM dédié, bucket S3, upload du dataset `bottle`**.

Garde-fous du CLAUDE.md applicables dès cette étape : région unique `eu-west-1`,
principe du moindre privilège sur tous les rôles/policies IAM (jamais de wildcard `*`
sur les ressources), signaler toute ressource facturée à l'heure avant de la créer (S3
est facturé au stockage/usage, pas à l'heure — pas de signal nécessaire ici).

## Décisions retenues

- **Compte AWS** : créé par l'utilisateur (console web), pas par l'assistant. Les
  identifiants ne sont jamais manipulés par l'assistant — l'utilisateur exécute lui-même
  `aws configure` avec les clés qu'il génère.
- **Utilisateur IAM** : `aws-anomalies-local`, dédié à ce projet. Policy scopée
  strictement à ce sous-projet : `s3:PutObject`, `s3:GetObject`, `s3:ListBucket`,
  `s3:DeleteObject`, restreinte à l'ARN du bucket créé ici (pas d'accès à d'autres
  buckets, pas d'autre service). Les sous-projets suivants (SageMaker, ECR, Lambda,
  API Gateway) étendront cette policy de façon incrémentale, permission par permission,
  au fur et à mesure des besoins réels — pas de policy large créée par anticipation.
- **Région** : `eu-west-1` (fixe pour tout le projet).
- **Nommage du bucket** : `aws-anomalies-mvtec-<suffixe choisi par l'utilisateur>` (les
  noms de bucket S3 sont globalement uniques ; l'utilisateur choisit le suffixe).
- **Configuration du bucket** : blocage de l'accès public activé (défaut de sécurité),
  versioning désactivé (non nécessaire pour un dataset statique), chiffrement par défaut
  SSE-S3 (gratuit, standard, suffisant — pas de donnée sensible, MVTec AD est un dataset
  public).
- **Approche IaC** : CLI/console pour ce sous-projet et les suivants (S3, SageMaker,
  Lambda, API Gateway) ; le Terraform équivalent sera écrit à la fin, dans le sous-projet
  dédié « Terraform complet », une fois toute l'architecture connue et stable. Objectif :
  avancer vite à chaque étape sans réécrire l'IaC à chaque changement de direction.
- **Contenu uploadé** : dataset `bottle` uniquement à ce stade (même incrément minimal
  que le reste du projet — `bottle` sert de cas de référence). `screw` et `carpet`
  seront uploadés avec le sous-projet SageMaker Training, quand ils seront réellement
  nécessaires à un entraînement cloud.
- **Structure S3** : miroir direct de la structure locale `data/mvtec/bottle/` :
  ```
  s3://<bucket>/mvtec/bottle/train/good/...
  s3://<bucket>/mvtec/bottle/test/<defect_type>/...
  s3://<bucket>/mvtec/bottle/ground_truth/<defect_type>/...
  ```
- **Upload** : `aws s3 sync data/mvtec/bottle s3://<bucket>/mvtec/bottle`, réutilise le
  dataset déjà téléchargé en local (pas de re-téléchargement depuis MVTec/HuggingFace).
- **Vérification** : après upload, comparer le nombre de fichiers listés sur S3
  (`aws s3 ls --recursive | wc -l`) au nombre de fichiers locaux
  (`find data/mvtec/bottle -type f | wc -l`) — les deux comptes doivent correspondre.
  Pas de vérification de contenu au-delà (pas de checksum) : le dataset est public et
  statique, un comptage de fichiers suffit à détecter un upload incomplet.

## Étapes

1. **Utilisateur** crée le compte AWS si besoin (hors scope assistant).
2. **Utilisateur** crée l'utilisateur IAM `aws-anomalies-local` et sa policy (l'assistant
   fournit le JSON de la policy et les commandes/étapes console, l'utilisateur exécute
   la création dans son compte).
3. **Utilisateur** génère les clés d'accès et lance `aws configure` (ou configure un
   profil nommé) en local.
4. **Assistant** vérifie que `aws sts get-caller-identity` répond avec l'ARN attendu
   (`aws-anomalies-local`), confirmant que la CLI est bien configurée et scopée au bon
   utilisateur — sans jamais voir ni manipuler les clés elles-mêmes.
5. **Assistant** crée le bucket S3 dans `eu-west-1` (nom confirmé par l'utilisateur),
   configure le blocage d'accès public.
6. **Assistant** lance l'upload `bottle` via `aws s3 sync`.
7. **Assistant** vérifie par comptage de fichiers (étape ci-dessus).

## Policy IAM proposée

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AwsAnomaliesBucketAccess",
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:GetObject",
        "s3:DeleteObject"
      ],
      "Resource": "arn:aws:s3:::aws-anomalies-mvtec-<suffixe>/*"
    },
    {
      "Sid": "AwsAnomaliesBucketList",
      "Effect": "Allow",
      "Action": "s3:ListBucket",
      "Resource": "arn:aws:s3:::aws-anomalies-mvtec-<suffixe>"
    }
  ]
}
```

## Hors périmètre (sous-projets suivants)

- Entraînement SageMaker (utilisera ce même bucket comme source de données, et son
  propre besoin en `screw`/`carpet`).
- ECR, SageMaker Serverless Inference.
- Lambda + API Gateway.
- Terraform complet (toute l'infra ci-dessus, une fois stabilisée).
- README et benchmark final.

## Tests prévus

Aucun test automatisé pour ce sous-projet : il s'agit exclusivement de provisioning
d'infrastructure (compte, IAM, bucket) et d'un transfert de fichiers, pas de code
applicatif. La vérification se fait par comptage de fichiers (cf. ci-dessus), pas par
suite pytest.
