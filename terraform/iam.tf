# --- Ressources existantes, importées (objets seulement, pas leurs policies) ---

resource "aws_iam_user" "local" {
  name = "aws-anomalies-local"
}

resource "aws_iam_role" "sagemaker_execution" {
  name        = "aws-anomalies-sagemaker-execution"
  description = "Allows SageMaker notebook instances, training jobs, and models to access S3, ECR, and CloudWatch on your behalf."

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
