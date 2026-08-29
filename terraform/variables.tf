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
