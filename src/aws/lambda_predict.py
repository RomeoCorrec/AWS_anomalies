"""Lambda "predict" : relaie une image binaire vers l'endpoint SageMaker Serverless."""
from __future__ import annotations

import base64
import json
import os

import boto3
from botocore.exceptions import ClientError

ENDPOINT_NAME = os.environ.get("SAGEMAKER_ENDPOINT_NAME", "aws-anomalies-bottle")
AWS_REGION = os.environ.get("AWS_REGION", "eu-west-1")

_runtime_client = None


def _get_runtime_client():
    """Retourne un client sagemaker-runtime, créé une seule fois par exécution froide."""
    global _runtime_client
    if _runtime_client is None:
        _runtime_client = boto3.client("sagemaker-runtime", region_name=AWS_REGION)
    return _runtime_client


def _response(status_code: int, body: dict) -> dict:
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def handler(event: dict, context) -> dict:
    """Point d'entrée Lambda : relaie le corps de la requête API Gateway vers l'endpoint SageMaker."""
    raw_body = event.get("body") or ""
    if event.get("isBase64Encoded"):
        body = base64.b64decode(raw_body)
    else:
        body = raw_body.encode("utf-8")

    content_type = event.get("headers", {}).get("content-type", "application/x-image")

    try:
        response = _get_runtime_client().invoke_endpoint(
            EndpointName=ENDPOINT_NAME, ContentType=content_type, Body=body
        )
    except ClientError as exc:
        error = exc.response.get("Error", {})
        if error.get("Code") == "ModelError":
            status_code = exc.response.get("OriginalStatusCode", 500)
            try:
                payload = json.loads(exc.response.get("OriginalMessage", "{}"))
            except (TypeError, ValueError):
                payload = {"error": exc.response.get("OriginalMessage", "model error")}
            return _response(status_code, payload)
        return _response(503, {"error": "endpoint unavailable"})

    payload = json.loads(response["Body"].read())
    return _response(200, payload)
