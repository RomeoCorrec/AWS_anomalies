resource "aws_ecr_repository" "train" {
  name                 = "aws-anomalies-train"
  image_tag_mutability = "MUTABLE"
}

resource "aws_ecr_repository" "serve" {
  name                 = "aws-anomalies-serve"
  image_tag_mutability = "MUTABLE"
}
