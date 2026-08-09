"""Déploie les Lambdas predict/authorizer et l'API Gateway HTTP API devant l'endpoint SageMaker."""
from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

LAMBDA_RUNTIME = "python3.12"
PREDICT_FUNCTION_NAME = "aws-anomalies-predict"
AUTHORIZER_FUNCTION_NAME = "aws-anomalies-authorizer"
API_NAME = "aws-anomalies-api"
PREDICT_SOURCE_PATH = Path(__file__).parent / "lambda_predict.py"
AUTHORIZER_SOURCE_PATH = Path(__file__).parent / "lambda_authorizer.py"


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


def deploy_api(
    lambda_client,
    apigw_client,
    predict_role_arn: str,
    authorizer_role_arn: str,
    api_key: str,
    region: str = "eu-west-1",
    account_id: str = "155466261331",
) -> str:
    """Déploie les 2 Lambdas et l'API Gateway HTTP API, retourne l'URL d'invocation de POST /predict."""
    predict_arn = deploy_lambda_function(
        lambda_client,
        function_name=PREDICT_FUNCTION_NAME,
        handler="lambda_predict.handler",
        role_arn=predict_role_arn,
        zip_bytes=zip_source(PREDICT_SOURCE_PATH),
        environment={"SAGEMAKER_ENDPOINT_NAME": "aws-anomalies-bottle"},
        timeout=30,
    )
    authorizer_arn = deploy_lambda_function(
        lambda_client,
        function_name=AUTHORIZER_FUNCTION_NAME,
        handler="lambda_authorizer.handler",
        role_arn=authorizer_role_arn,
        zip_bytes=zip_source(AUTHORIZER_SOURCE_PATH),
        environment={"API_KEY": api_key},
        timeout=10,
    )

    api = apigw_client.create_api(Name=API_NAME, ProtocolType="HTTP")
    api_id = api["ApiId"]

    integration = apigw_client.create_integration(
        ApiId=api_id,
        IntegrationType="AWS_PROXY",
        IntegrationUri=predict_arn,
        PayloadFormatVersion="2.0",
        IntegrationMethod="POST",
    )

    authorizer = apigw_client.create_authorizer(
        ApiId=api_id,
        AuthorizerType="REQUEST",
        Name="api-key-authorizer",
        AuthorizerUri=(
            f"arn:aws:apigateway:{region}:lambda:path/2015-03-31/functions/{authorizer_arn}/invocations"
        ),
        AuthorizerPayloadFormatVersion="2.0",
        EnableSimpleResponses=True,
        IdentitySource=["$request.header.x-api-key"],
    )

    apigw_client.create_route(
        ApiId=api_id,
        RouteKey="POST /predict",
        Target=f"integrations/{integration['IntegrationId']}",
        AuthorizationType="CUSTOM",
        AuthorizerId=authorizer["AuthorizerId"],
    )

    apigw_client.create_stage(ApiId=api_id, StageName="$default", AutoDeploy=True)

    lambda_client.add_permission(
        FunctionName=PREDICT_FUNCTION_NAME,
        StatementId=f"apigateway-invoke-predict-{api_id}",
        Action="lambda:InvokeFunction",
        Principal="apigateway.amazonaws.com",
        SourceArn=f"arn:aws:execute-api:{region}:{account_id}:{api_id}/*/*/predict",
    )
    lambda_client.add_permission(
        FunctionName=AUTHORIZER_FUNCTION_NAME,
        StatementId=f"apigateway-invoke-authorizer-{api_id}",
        Action="lambda:InvokeFunction",
        Principal="apigateway.amazonaws.com",
        SourceArn=f"arn:aws:execute-api:{region}:{account_id}:{api_id}/authorizers/{authorizer['AuthorizerId']}",
    )

    return f"{api['ApiEndpoint']}/predict"


def main() -> None:
    """CLI : déploie les Lambdas et l'API Gateway HTTP API devant l'endpoint SageMaker."""
    import boto3

    parser = argparse.ArgumentParser(description="Déploie Lambda + API Gateway devant l'endpoint SageMaker.")
    parser.add_argument("--predict-role-arn", required=True)
    parser.add_argument("--authorizer-role-arn", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--region", default="eu-west-1")
    parser.add_argument("--account-id", default="155466261331")
    args = parser.parse_args()

    lambda_client = boto3.client("lambda", region_name=args.region)
    apigw_client = boto3.client("apigatewayv2", region_name=args.region)

    invoke_url = deploy_api(
        lambda_client,
        apigw_client,
        predict_role_arn=args.predict_role_arn,
        authorizer_role_arn=args.authorizer_role_arn,
        api_key=args.api_key,
        region=args.region,
        account_id=args.account_id,
    )
    print(f"API déployée : {invoke_url}")


if __name__ == "__main__":
    main()
