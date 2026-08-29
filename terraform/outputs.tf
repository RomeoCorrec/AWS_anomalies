output "predict_invoke_url" {
  description = "URL d'invocation de POST /predict."
  value       = "${aws_apigatewayv2_api.main.api_endpoint}/predict"
}
