import pandas as pd
import numpy as np
import tensorflow as tf


import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from datetime import datetime, timedelta
from tensorflow import keras
import logging
import sys
import os
import optuna
import time
from multiprocessing import Pool
import json
from helpers import load_studies, ensamble_forecast

# Accepts arguments:
#     cty (currently only DE), default: DE
#     distribution (Normal, StudentT, JSU, SinhArcsinh and NormalInverseGaussian), default: Normal

distribution = "JSU"
paramcount = {
    "Normal": 2,
    "StudentT": 3,
    "JSU": 4,
    "SinhArcsinh": 4,
    "NormalInverseGaussian": 4,
    "Point": None,
}

INP_SIZE = 221
# activations, neurons and params read from the trials info

cty = "DE"

if len(sys.argv) > 1:
    print("Arguments are disabled for this script")


print("DDNN rolling ensemble", cty, distribution)


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))



if cty != "DE":
    raise ValueError("Incorrect country")
if distribution not in paramcount:
    raise ValueError("Incorrect distribution")

# read data file
data = pd.read_csv(f"../Datasets/{cty}.csv", index_col=0)
data.index = [datetime.strptime(e, "%Y-%m-%d %H:%M:%S") for e in data.index]


forecast_id = "2"
path_name = f"../forecasts_ddnn_{forecast_id}"
days_to_predict = 200
stop_after = 100 # TODO DELTE LATER
training_days = len(data) // 24 - days_to_predict


# create directory if it does not exist
if not os.path.exists(path_name):
    os.mkdir(path_name)



print("Days in data set:", (len(data) // 24))
print("Days in training set:", training_days)


def runoneday(inp):
    params, dayno = inp

    print("Day number ", dayno)

    df = data.iloc[dayno * 24 : dayno * 24 + training_days * 24 + 24]

    # print first and last date of df
    print(df.index[0], df.index[-1])

    # prepare the input/output dataframes
    Y = np.zeros((training_days, 24))
    # Yf = np.zeros((1, 24)) # no Yf for rolling prediction
    for d in range(training_days):
        Y[d, :] = df.loc[df.index[d * 24 : (d + 1) * 24], "Price"].to_numpy()
    Y = Y[7:, :]  # skip first 7 days
    # for d in range(1):
    #     Yf[d, :] = df.loc[df.index[(d+1092)*24:(d+1093)*24], 'Price'].to_numpy()
    X = np.zeros((training_days + 1, INP_SIZE))
    for d in range(7, training_days + 1):
        X[d, :24] = df.loc[
            df.index[(d - 1) * 24 : (d) * 24], "Price"
        ].to_numpy()  # D-1 price
        X[d, 24:48] = df.loc[
            df.index[(d - 2) * 24 : (d - 1) * 24], "Price"
        ].to_numpy()  # D-2 price
        X[d, 48:72] = df.loc[
            df.index[(d - 3) * 24 : (d - 2) * 24], "Price"
        ].to_numpy()  # D-3 price
        X[d, 72:96] = df.loc[
            df.index[(d - 7) * 24 : (d - 6) * 24], "Price"
        ].to_numpy()  # D-7 price
        X[d, 96:120] = df.loc[
            df.index[(d) * 24 : (d + 1) * 24], df.columns[1]
        ].to_numpy()  # D load forecast
        X[d, 120:144] = df.loc[
            df.index[(d - 1) * 24 : (d) * 24], df.columns[1]
        ].to_numpy()  # D-1 load forecast
        X[d, 144:168] = df.loc[
            df.index[(d - 7) * 24 : (d - 6) * 24], df.columns[1]
        ].to_numpy()  # D-7 load forecast
        X[d, 168:192] = df.loc[
            df.index[(d) * 24 : (d + 1) * 24], df.columns[2]
        ].to_numpy()  # D RES sum forecast
        X[d, 192:216] = df.loc[
            df.index[(d - 1) * 24 : (d) * 24], df.columns[2]
        ].to_numpy()  # D-1 RES sum forecast
        X[d, 216] = df.loc[
            df.index[(d - 2) * 24 : (d - 1) * 24 : 24], df.columns[3]
        ].to_numpy()  # D-2 EUA
        X[d, 217] = df.loc[
            df.index[(d - 2) * 24 : (d - 1) * 24 : 24], df.columns[4]
        ].to_numpy()  # D-2 API2_Coal
        X[d, 218] = df.loc[
            df.index[(d - 2) * 24 : (d - 1) * 24 : 24], df.columns[5]
        ].to_numpy()  # D-2 TTF_Gas
        X[d, 219] = df.loc[
            df.index[(d - 2) * 24 : (d - 1) * 24 : 24], df.columns[6]
        ].to_numpy()  # D-2 Brent oil
        X[d, 220] = df.index[d].weekday()
    # '''
    # input feature selection
    colmask = [False] * INP_SIZE
    if params["price_D-1"]:
        colmask[:24] = [True] * 24
    if params["price_D-2"]:
        colmask[24:48] = [True] * 24
    if params["price_D-3"]:
        colmask[48:72] = [True] * 24
    if params["price_D-7"]:
        colmask[72:96] = [True] * 24
    if params["load_D"]:
        colmask[96:120] = [True] * 24
    if params["load_D-1"]:
        colmask[120:144] = [True] * 24
    if params["load_D-7"]:
        colmask[144:168] = [True] * 24
    if params["RES_D"]:
        colmask[168:192] = [True] * 24
    if params["RES_D-1"]:
        colmask[192:216] = [True] * 24
    if params["EUA"]:
        colmask[216] = True
    if params["Coal"]:
        colmask[217] = True
    if params["Gas"]:
        colmask[218] = True
    if params["Oil"]:
        colmask[219] = True
    if params["Dummy"]:
        colmask[220] = True
    X = X[:, colmask]
    # '''
    Xf = X[-1:, :]
    X = X[7:-1, :]
    # begin building a model
    inputs = keras.Input(
        X.shape[1]
    )  # <= INP_SIZE as some columns might have been turned off
    # batch normalization
    batchnorm = True  # params['batch_normalization'] # trial.suggest_categorical('batch_normalization', [True, False])
    if batchnorm:
        norm = keras.layers.BatchNormalization()(inputs)
        last_layer = norm
    else:
        last_layer = inputs
    # dropout
    dropout = params["dropout"]  # trial.suggest_categorical('dropout', [True, False])
    if dropout:
        rate = params["dropout_rate"]  # trial.suggest_float('dropout_rate', 0, 1)
        drop = keras.layers.Dropout(rate)(last_layer)
        last_layer = drop
    # regularization of 1st hidden layer,
    # activation - output, kernel - weights/parameters of input
    regularize_h1_activation = params["regularize_h1_activation"]
    regularize_h1_kernel = params["regularize_h1_kernel"]
    h1_activation_rate = (
        0.0 if not regularize_h1_activation else params["h1_activation_rate_l1"]
    )
    h1_kernel_rate = 0.0 if not regularize_h1_kernel else params["h1_kernel_rate_l1"]
    # define 1st hidden layer with regularization
    hidden = keras.layers.Dense(
        params["neurons_1"],
        activation=params["activation_1"],
        # kernel_initializer='ones',
        kernel_regularizer=keras.regularizers.L1(h1_kernel_rate),
        activity_regularizer=keras.regularizers.L1(h1_activation_rate),
    )(last_layer)
    # regularization of 2nd hidden layer,
    # activation - output, kernel - weights/parameters of input
    regularize_h2_activation = params["regularize_h2_activation"]
    regularize_h2_kernel = params["regularize_h2_kernel"]
    h2_activation_rate = (
        0.0 if not regularize_h2_activation else params["h2_activation_rate_l1"]
    )
    h2_kernel_rate = 0.0 if not regularize_h2_kernel else params["h2_kernel_rate_l1"]
    # define 2nd hidden layer with regularization
    hidden = keras.layers.Dense(
        params["neurons_2"],
        activation=params["activation_2"],
        # kernel_initializer='ones',
        kernel_regularizer=keras.regularizers.L1(h2_kernel_rate),
        activity_regularizer=keras.regularizers.L1(h2_activation_rate),
    )(hidden)
    if paramcount[distribution] is None:
        outputs = keras.layers.Dense(24, activation="linear")(hidden)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=keras.optimizers.legacy.Adam(params["learning_rate"]),
            loss="mae",
            metrics="mae",
        )
    else:
        # now define parameter layers with their regularization
        param_layers = []
        param_names = ["loc", "scale", "tailweight", "skewness"]
        for p in range(paramcount[distribution]):
            regularize_param_kernel = params["regularize_" + param_names[p]]
            param_kernel_rate = (
                0.0
                if not regularize_param_kernel
                else params[str(param_names[p]) + "_rate_l1"]
            )
            param_layers.append(
                keras.layers.Dense(
                    24,
                    activation="linear",  # kernel_initializer='ones',
                    kernel_regularizer=keras.regularizers.L1(param_kernel_rate),
                )(hidden)
            )
        # concatenate the parameter layers to one
        linear = tf.keras.layers.concatenate(param_layers)
        # define outputs
        if distribution == "Normal":
            outputs = tfp.layers.DistributionLambda(
                lambda t: tfd.Normal(
                    loc=t[..., :24], scale=1e-3 + 3 * tf.math.softplus(t[..., 24:])
                )
            )(linear)
        elif distribution == "StudentT":
            outputs = tfp.layers.DistributionLambda(
                lambda t: tfd.StudentT(
                    loc=t[..., :24],
                    scale=1e-3 + 3 * tf.math.softplus(t[..., 24:48]),
                    df=1 + 3 * tf.math.softplus(t[..., 48:]),
                )
            )(linear)
        elif distribution == "JSU":
            outputs = tfp.layers.DistributionLambda(
                lambda t: tfd.JohnsonSU(
                    loc=t[..., :24],
                    scale=1e-3 + 3 * tf.math.softplus(t[..., 24:48]),
                    tailweight=1 + 3 * tf.math.softplus(t[..., 48:72]),
                    skewness=t[..., 72:],
                )
            )(linear)
        elif distribution == "SinhArcsinh":
            outputs = tfp.layers.DistributionLambda(
                lambda t: tfd.SinhArcsinh(
                    loc=t[..., :24],
                    scale=1e-3 + 3 * tf.math.softplus(t[..., 24:48]),
                    tailweight=1e-3 + 3 * tf.math.softplus(t[..., 48:72]),
                    skewness=t[..., 72:],
                )
            )(linear)
        elif distribution == "NormalInverseGaussian":
            outputs = tfp.layers.DistributionLambda(
                lambda t: tfd.NormalInverseGaussian(
                    loc=t[..., :24],
                    scale=1e-3 + 3 * tf.math.softplus(t[..., 24:48]),
                    tailweight=1e-3 + 3 * tf.math.softplus(t[..., 48:72]),
                    skewness=t[..., 72:],
                )
            )(linear)
        else:
            raise ValueError(f"Incorrect distribution {distribution}")
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=keras.optimizers.legacy.Adam(params["learning_rate"]),
            loss=lambda y, rv_y: -rv_y.log_prob(y),
            metrics="mae",
        )
    # '''
    # define callbacks
    callbacks = [keras.callbacks.EarlyStopping(patience=50, restore_best_weights=True)]
    perm = np.random.permutation(np.arange(X.shape[0]))
    VAL_DATA = 0.2
    trainsubset = perm[: int((1 - VAL_DATA) * len(perm))]
    valsubset = perm[int((1 - VAL_DATA) * len(perm)) :]
    model.fit(
        X[trainsubset],
        Y[trainsubset],
        epochs=1500,
        validation_data=(X[valsubset], Y[valsubset]),
        callbacks=callbacks,
        batch_size=32,
        verbose=False,
    )

    # metrics = model.evaluate(Xf, Yf) # for point its a list of one [loss, MAE]
    # we optimize the returned value, -1 will always take the model with best MAE

    # pred = model.predict(Xf)[0]

    print("Distribution top")
    # analyse Xf
    dist = model(Xf)
    if distribution == "Normal":
        getters = {"loc": dist.loc, "scale": dist.scale}
    elif distribution == "StudentT":
        getters = {"loc": dist.loc, "scale": dist.scale, "df": dist.df}
    elif distribution in {"JSU", "SinhArcsinh", "NormalInverseGaussian"}:
        getters = {
            "loc": dist.loc,
            "scale": dist.scale,
            "tailweight": dist.tailweight,
            "skewness": dist.skewness,
        }
    params = {k: [float(e) for e in v.numpy()[0]] for k, v in getters.items()}

    file_name = datetime.strftime(df.index[-24], "%Y-%m-%d")
    # json.dump(params, open(os.path.join(f'../distparams_probNN_{distribution.lower()}', f'{file_name}.json'), 'w'))
    pred = model.predict(np.tile(Xf, (10000, 1)))
    predDF = pd.DataFrame(index=df.index[-24:])
    predDF["real"] = df.loc[df.index[-24:], "Price"].to_numpy()
    predDF["forecast"] = pd.NA
    predDF.loc[predDF.index[:], "forecast"] = pred.mean(0)

    # print shape of pred
    print(pred.shape)

    # Calculate the 5th and 95th percentiles for each hour to get the 90% Prediction Intervals
    lower_bound_90 = np.percentile(pred, 5, axis=0)
    upper_bound_90 = np.percentile(pred, 95, axis=0)
    bound_50 = np.percentile(pred, 50, axis=0)

    # add the 90% Prediction Intervals to the dataframe
    predDF["lower_bound_90"] = pd.NA
    predDF["upper_bound_90"] = pd.NA
    predDF["bound_50"] = pd.NA
    predDF.loc[predDF.index[:], "lower_bound_90"] = lower_bound_90
    predDF.loc[predDF.index[:], "upper_bound_90"] = upper_bound_90
    predDF.loc[predDF.index[:], "bound_50"] = bound_50

    return predDF


def use_study(study_config):
    study, study_name = study_config
    print(f"Using study {study_name}")
    inputlist = [
        (study.best_params, day) for day in range(0, len(data) // 24 - training_days)
    ]

    file_name = f"prediction_{study_name}.json"
    file_path = os.path.join(path_name, file_name)
    
    # Read existing data from the JSON file
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            existing_data = json.load(file)
    else:
        existing_data = {}

    for e in inputlist:
        # stop after stop_after days
        if len(existing_data) >= stop_after * 24:
            print("Stopping after ", stop_after, " days")
            break
        prediction = runoneday(e)
        print(prediction)

        # Convert the index to a DateTime object (if it's in Unix timestamp format)
        prediction.index = pd.to_datetime(prediction.index, unit="ms")

        # Convert the entire index to strings in your desired format (e.g., "YYYY-MM-DD HH:MM:SS")
        prediction.index = prediction.index.strftime("%Y-%m-%d %H:%M:%S")
        prediction.index.name = "date"

        # Convert the DataFrame to a dictionary
        new_data = prediction.to_dict(orient="index")

        # Update existing data with the new day's prediction
        existing_data.update(new_data)

        # Write the updated data back to the JSON file
        with open(file_path, "w") as file:
            json.dump(existing_data, file, indent=2)

        # existing_data to csv file
        existing_data_df = pd.DataFrame.from_dict(existing_data, orient="index")
        existing_data_df.to_csv(os.path.join(path_name, f"prediction_{study_name}.csv"))

    # with Pool(max(os.cpu_count() // 4, 1)) as p:
    #     _ = p.map(runoneday, inputlist)


# Run the use_study function for each study in parallel
if __name__ == "__main__":
    study_count = 1 # 4
    study_configs = load_studies(
        base_name="FINAL_DE_selection_prob_jsu", count=study_count
    )

    print("Run only with 1 pool")

    # Use a Pool with 4 processes to run the use_studies function in parallel
    with Pool(1) as p:
        _ = p.map(use_study, study_configs)
        print("Finished running use_study in parallel")
    
    # calculate the ensemble forecast
    ensamble_forecast(path_name)
