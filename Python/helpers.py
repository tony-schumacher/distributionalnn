import optuna
import logging
import sys
import pandas as pd
import re
import os


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


def read_csv_files_ending_with_number(directory):
    # Regular expression to match files ending with a number and .csv extension
    pattern = re.compile(r"\d+\.csv$")

    # List to hold DataFrames
    dfs = []

    # Iterate through files in the directory
    for file_name in os.listdir(directory):
        if pattern.search(file_name):
            # Read CSV file into DataFrame
            file_path = os.path.join(directory, file_name)
            df = pd.read_csv(file_path, index_col=0)
            dfs.append(df)

    return dfs


# todo handle missing values
def ensamble_forecast(directory):
    # Read the CSV files into DataFrames
    dfs = read_csv_files_ending_with_number(directory)

    print(f"Found {len(dfs)} CSV files")

    # Create a DataFrame to hold the original values and ensembled values
    ensembled_df = dfs[0].copy()
    ensembled_df.columns = [col + "_m1" for col in ensembled_df.columns]

    # rename real_m1 to real
    ensembled_df.rename(columns={"real_m1": "real"}, inplace=True)

    # Add the original values from other models
    for i, df in enumerate(dfs[1:], 2):
        ensembled_df[f"forecast_m{i}"] = df["forecast"]

    # Calculate the ensembled forecast
    ensembled_df["forecast_ensembled"] = sum(df["forecast"] for df in dfs) / len(dfs)
    ensembled_df["lower_bound_90_ensembled"] = sum(
        df["lower_bound_90"] for df in dfs
    ) / len(dfs)
    ensembled_df["upper_bound_90_ensembled"] = sum(
        df["upper_bound_90"] for df in dfs
    ) / len(dfs)
    ensembled_df["bound_50_ensembled"] = sum(df["bound_50"] for df in dfs) / len(dfs)

    # Select only the desired columns
    final_columns = (
        ["real"]
        + [f"forecast_m{i}" for i in range(1, len(dfs) + 1)]
        + [
            "forecast_ensembled",
            "bound_50_ensembled",
            "lower_bound_90_ensembled",
            "upper_bound_90_ensembled",
        ]
    )
    ensembled_df = ensembled_df[final_columns]

    # Save to CSV
    ensembled_df.to_csv(os.path.join(directory, "prediction_ensembled.csv"))

    print("Saved ensembled forecast to prediction_ensembled.csv")
