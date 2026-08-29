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
