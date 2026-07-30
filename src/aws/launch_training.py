"""Lance un SageMaker Training Job PatchCore (BYOC) sur les données déjà uploadées sur S3."""
from __future__ import annotations

import argparse

from sagemaker.estimator import Estimator

DEFAULT_INSTANCE_TYPE = "ml.m5.xlarge"


def build_estimator(image_uri: str, role_arn: str, output_path: str, instance_type: str = DEFAULT_INSTANCE_TYPE) -> Estimator:
    """Construit l'Estimator SageMaker pour l'image de training BYOC."""
    return Estimator(
        image_uri=image_uri,
        role=role_arn,
        instance_count=1,
        instance_type=instance_type,
        output_path=output_path,
    )


def launch_training(
    image_uri: str,
    role_arn: str,
    training_data_uri: str,
    output_path: str,
    experiment_path: str,
    instance_type: str = DEFAULT_INSTANCE_TYPE,
) -> str:
    """Lance le Training Job, attend sa fin, retourne l'URI S3 du model.tar.gz produit."""
    estimator = build_estimator(image_uri, role_arn, output_path, instance_type)
    estimator.set_hyperparameters(experiment=experiment_path)
    estimator.fit({"training": training_data_uri})
    return estimator.model_data


def main() -> None:
    """CLI : lance un Training Job SageMaker pour une config d'expérience donnée."""
    parser = argparse.ArgumentParser(description="Lance un SageMaker Training Job PatchCore (BYOC).")
    parser.add_argument("--image-uri", required=True)
    parser.add_argument("--role-arn", required=True)
    parser.add_argument("--training-data-uri", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--instance-type", default=DEFAULT_INSTANCE_TYPE)
    args = parser.parse_args()

    model_data = launch_training(
        image_uri=args.image_uri,
        role_arn=args.role_arn,
        training_data_uri=args.training_data_uri,
        output_path=args.output_path,
        experiment_path=args.experiment,
        instance_type=args.instance_type,
    )
    print(f"Modèle entraîné disponible sur S3 : {model_data}")


if __name__ == "__main__":
    main()
