import optuna
import logging
import sys


def load_studies(base_name="FINAL_DE_selection_prob_jsu", count=4):
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    studies = []

    for i in range(1, count + 1):
        study_name = f"{base_name}{i}"
        storage_name = f"sqlite:///../trialfiles/{study_name}"
        study = optuna.load_study(
            study_name=study_name.replace(str(i), ""), storage=storage_name
        )

        studies.append(study)

        print(f"Loaded study {study_name} with {len(study.trials)} trials")

    return studies
