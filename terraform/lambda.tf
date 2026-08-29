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
