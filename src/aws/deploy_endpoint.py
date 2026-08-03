"""Déploie le checkpoint entraîné derrière un endpoint SageMaker Serverless Inference (BYOC)."""
from __future__ import annotations

import argparse

from sagemaker.estimator import Estimator  # noqa: F401 - import order workaround for sagemaker circular import
from sagemaker.model import Model
from sagemaker.serverless import ServerlessInferenceConfig

DEFAULT_MEMORY_SIZE_MB = 2048
DEFAULT_MAX_CONCURRENCY = 1


def deploy_endpoint(
    image_uri: str,
    role_arn: str,
    model_data_url: str,
    endpoint_name: str,
    memory_size_in_mb: int = DEFAULT_MEMORY_SIZE_MB,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
) -> str:
    """Déploie le modèle derrière un endpoint Serverless, retourne le nom de l'endpoint."""
    model = Model(image_uri=image_uri, model_data=model_data_url, role=role_arn)
    serverless_config = ServerlessInferenceConfig(
        memory_size_in_mb=memory_size_in_mb, max_concurrency=max_concurrency
    )
    model.deploy(endpoint_name=endpoint_name, serverless_inference_config=serverless_config)
    return endpoint_name


def main() -> None:
    """CLI : déploie l'endpoint Serverless Inference pour un checkpoint PatchCore donné."""
    parser = argparse.ArgumentParser(description="Déploie un endpoint SageMaker Serverless Inference (BYOC).")
    parser.add_argument("--image-uri", required=True)
    parser.add_argument("--role-arn", required=True)
    parser.add_argument("--model-data-url", required=True)
    parser.add_argument("--endpoint-name", required=True)
    parser.add_argument("--memory-size-mb", type=int, default=DEFAULT_MEMORY_SIZE_MB)
    parser.add_argument("--max-concurrency", type=int, default=DEFAULT_MAX_CONCURRENCY)
    args = parser.parse_args()

    endpoint_name = deploy_endpoint(
        image_uri=args.image_uri,
        role_arn=args.role_arn,
        model_data_url=args.model_data_url,
        endpoint_name=args.endpoint_name,
        memory_size_in_mb=args.memory_size_mb,
        max_concurrency=args.max_concurrency,
    )
    print(f"Endpoint déployé : {endpoint_name}")


if __name__ == "__main__":
    main()
