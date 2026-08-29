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
  api_id                             = aws_apigatewayv2_api.main.id
  authorizer_type                    = "REQUEST"
  name                                = "api-key-authorizer"
  authorizer_uri                     = aws_lambda_function.authorizer.invoke_arn
  authorizer_payload_format_version  = "2.0"
  enable_simple_responses            = true
  identity_sources                   = ["$request.header.x-api-key"]
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
