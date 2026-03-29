import mlflow
import optuna
import yaml
from src.model.model_training import train_model_cv, train_model_final
from config.config import EXPERIMENTATION_DIR
import torch
import argparse

mlflow.set_tracking_uri("sqlite:///mlflow.db")

def run_experiment(config_path):
    """Starts MLFlow Logging and Optuna Study

    Args:
        config_path (str): The file path of the YAML file for experimentation
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    mlflow.set_experiment(config["experiment"]["experiment_name"])

    def objective(trial):
        """Optuna Study

        Args:
            trial (optuna.trial.Trial): The optuna trial

        Returns:
            float: Returns accuracy of the current studied model
        """
        config["training"]["optimizer_params"]["lr"] = trial.suggest_float(
            name="lr",
            low=config["training"]["lr_range"][0],
            high=config["training"]["lr_range"][1],
            log=True
        )

        config["training"]["batch_size"] = trial.suggest_categorical(
            name="batch_size",
            choices=config["training"]["batch_size_options"]
        )

        with mlflow.start_run(run_name=f"trial_{trial.number}",nested=True):
            mlflow.log_params(trial.params)

            mlflow.log_param("epochs", config["training"]["epochs"])
            mlflow.log_param("n_splits", config["training"]["n_splits"])
            mlflow.log_dict(config["training"]["augmentations"], "augmentations.json")

            avg_acc, std_acc = train_model_cv(config)

            mlflow.log_metric("mean_cv_accuracy", avg_acc)
            mlflow.log_metric("std_cv_accuracy", std_acc)

            return avg_acc
        
    with mlflow.start_run(run_name=config["experiment"]["run_name"]) as parent_run:
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=config["experiment"]["n_trials"])

        config["training"]["optimizer_params"]["lr"] = study.best_params["lr"]
        config["training"]["batch_size"] = study.best_params["batch_size"]

        final_acc, model = train_model_final(config)

        dummy_input = torch.randn(1, 3, 224, 244).to(next(model.parameters()).device)
        signature = mlflow.models.infer_signature(
            model_input=dummy_input.cpu().numpy(),
            model_output=model(dummy_input).detach().cpu().numpy()
        )

        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            signature=signature
        )

        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_accuracy", study.best_value)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Generated Image Recognition")

    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="Write the name of the YAML config file"
    )

    args = parser.parse_args()
    if args.config:
        config_file = EXPERIMENTATION_DIR / args.config
        run_experiment(config_file)