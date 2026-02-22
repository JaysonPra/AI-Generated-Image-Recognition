import mlflow
import optuna
import yaml
from src.model.model_training import train_model

def run_experiment(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    mlflow.set_experiment(config["experiment"]["experiment_name"])

    def objective(trial):
        config["training"]["optimizer"]["lr"] = trial.suggest_float(
            name="lr",
            low=config["training"]["lr_range"][0],
            high=config["training"]["lr_range"][1],
            log=True
        )

        config["training"]["batch_size"] = trial.suggest_categorical(
            name="batch_size",
            choices=config["training"]["batch_size_options"]
        )

        with mlflow.start_run(run_name=f"trial_{trail.number}",nested=True):
            mlflow.log_params(trial.params)

            mlflow.log_param("epochs", config["training"]["epochs"])
            mlflow.log_param("n_splits", config["training"]["n_splits"])
            mlflow.log_dict(config["training"]["augmentations"], "augmentations.json")

            avg_acc, std_acc = train_model(config)

            mlflow.log_metric("mean_cv_accuracy", avg_acc)
            mlflow.log_metric("std_cv_accuracy", std_acc)

            return avg_acc
        
    mlflow.start_run(run_name=config["experiment"]["run_name"]):
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=config["experiment"]["n_trails"])

        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_accuracy", study.best_value)
