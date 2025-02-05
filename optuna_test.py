import optuna
import optunahub
from optuna.study import StudyDirection
from optuna_dashboard import run_server
import logging
import sys

def objective(trial: optuna.Trial):
    x = trial.suggest_float("x", -100, 100)
    y = trial.suggest_categorical("y", [-1, 0, 1])
    return x**2 + y


if __name__ == "__main__":
    #optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "RAGAS_workshop_test" 
    storage_name = "sqlite:///{}.db".format(study_name)
    module = optunahub.load_module(package="samplers/auto_sampler")
    directions = [StudyDirection.MAXIMIZE]

    study = optuna.create_study(
        study_name=study_name,
        sampler=module.AutoSampler(),
        storage=storage_name,
        load_if_exists= True,
        directions=directions
        )

    for _ in range(1):
            study.optimize(objective, n_trials=5)
    print(f"Best value: {study.best_value} (params: {study.best_params})")

    run_server(storage_name)