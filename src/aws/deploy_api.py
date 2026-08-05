"""Déploie les Lambdas predict/authorizer et l'API Gateway HTTP API devant l'endpoint SageMaker."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

LAMBDA_RUNTIME = "python3.12"


def zip_source(source_path: Path) -> bytes:
    """Zippe un unique fichier Python source pour le déploiement Lambda."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(source_path, arcname=source_path.name)
    return buffer.getvalue()


def deploy_lambda_function(
    lambda_client,
    function_name: str,
    handler: str,
    role_arn: str,
    zip_bytes: bytes,
    environment: dict[str, str] | None = None,
    timeout: int = 30,
) -> str:
    """Crée la fonction Lambda si absente, sinon met à jour son code. Retourne l'ARN de la fonction."""
    try:
        existing = lambda_client.get_function(FunctionName=function_name)
        lambda_client.update_function_code(FunctionName=function_name, ZipFile=zip_bytes)
        return existing["Configuration"]["FunctionArn"]
    except lambda_client.exceptions.ResourceNotFoundException:
        response = lambda_client.create_function(
            FunctionName=function_name,
            Runtime=LAMBDA_RUNTIME,
            Role=role_arn,
            Handler=handler,
            Code={"ZipFile": zip_bytes},
            Timeout=timeout,
            Environment={"Variables": environment or {}},
        )
        return response["FunctionArn"]
