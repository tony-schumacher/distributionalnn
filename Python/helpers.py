import optuna
import logging
import sys


def load_studies(base_name="FINAL_DE_selection_prob_jsu", count=4):
    """
    Load Optuna studies based on the provided base name and count.
    
    Parameters:
    - base_name (str): The base name of the study. Default is "FINAL_DE_selection_prob_jsu".
    - count (int): The number of studies to load. The function will try to load studies 
                   from `base_name1` to `base_name[count]`. Default is 4.
    
    Returns:
    - list: A list of tuples where each tuple contains an Optuna study object and its 
            corresponding study name.
    
    Note:
    The studies are expected to be saved in an SQLite database located in the "../trialfiles/" directory.
    """
    
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    studies = []

    for i in range(1, count + 1):
        study_name = f"{base_name}{i}"
        storage_name = f"sqlite:///../trialfiles/{study_name}"
        study = optuna.load_study(
            study_name=study_name.replace(str(i), ""), storage=storage_name
        )

        studies.append((study, study_name))

        print(f"Loaded study {study_name} with {len(study.trials)} trials")

    return studies
