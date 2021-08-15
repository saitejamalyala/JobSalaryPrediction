import csv
import json
import numpy as np
import os
import pandas as pd
import pickle

os.environ['SALARY_PRED'] = os.getcwd()

settings_json_path = os.path.expandvars("$SALARY_PRED/src/SETTINGS.json")
print(settings_json_path)

def get_paths():
    """
    Returns a dictionary of all paths in the project.
    """
    paths = json.loads(open(settings_json_path).read())
    for key in paths:
        paths[key] = os.path.expandvars(paths[key])
    return paths

def identity(x):
    """
    Simple linear function.
    """
    return x

# For pandas >= 10.1 this will trigger the columns to be parsed as strings
converters = { "FullDescription" : identity
             , "Title": identity
             , "LocationRaw": identity
             , "LocationNormalized": identity
             }

def get_train_df()->pd.DataFrame:
    """AI is creating summary for get_train_df

    Returns:
        [type]: [description]
    """
    train_path = get_paths()["train_data_path"]
    return pd.read_csv(train_path, converters=converters)

def get_valid_df()->pd.DataFrame:
    valid_path = get_paths()["valid_data_path"]
    return pd.read_csv(valid_path, converters=converters)

def get_test_df()->pd.DataFrame:
    test_path = get_paths()["test_data_path"]
    return pd.read_csv(test_path, converters=converters)

def save_model(model):
    out_path = get_paths()["model_path"]
    pickle.dump(model, open(out_path, "wb"))

def load_model():
    in_path = get_paths()["model_path"]
    return pickle.load(open(in_path, "rb"))

def write_submission(predictions):
    prediction_path = get_paths()["prediction_path"]
    writer = csv.writer(open(prediction_path, "w"), lineterminator="\n")
    valid = get_valid_df()
    rows = [x for x in zip(valid["Id"], predictions.flatten())]
    writer.writerow(("Id", "SalaryNormalized"))
    writer.writerows(rows)

def write_submission_test(predictions):
    prediction_path = get_paths()["test_prediction_path"]
    writer = csv.writer(open(prediction_path, "w"), lineterminator="\n")
    valid = get_test_df()
    rows = [x for x in zip(valid["Id"], predictions.flatten())]
    writer.writerow(("Id", "SalaryNormalized"))
    writer.writerows(rows)