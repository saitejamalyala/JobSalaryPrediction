import data_io
from features import FeatureMapper
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
import os

os.environ["SALARY_PRED"] = os.getcwd()
# print(os.getcwd(),'\n',os.environ)


def feature_extractor():
    """
    Extracts features from the training data and returns a feature mapper
    """
    features = [
        (
            "FullDescription-Bag of Words",
            "FullDescription",
            CountVectorizer(max_features=100),
        ),
        ("Title-Bag of Words", "Title", CountVectorizer(max_features=100)),
        ("LocationRaw-Bag of Words", "LocationRaw", CountVectorizer(max_features=100)),
        (
            "LocationNormalized-Bag of Words",
            "LocationNormalized",
            CountVectorizer(max_features=100),
        ),
    ]
    combined = FeatureMapper(features)
    return combined


def get_pipeline():
    """
    Defines a pipeline for the Regressor
    """
    features = feature_extractor()
    model = RandomForestRegressor(
        n_estimators=50, n_jobs=4, min_samples_split=30, random_state=42
    )
    # model = GradientBoostingRegressor(n_estimators=100, max_depth=8, min_samples_split=30, random_state=42)
    steps = [("extract_features", features), ("classify", model)]
    model = Pipeline(steps)
    return model


def main():
    print("Reading in the training data")
    train = data_io.get_train_df()

    print("Extracting features and training model")
    regressor = get_pipeline()
    regressor.fit(train, train["SalaryNormalized"])

    print("Saving the classifier")
    # model_name = 'GradientBoost.pickle'
    # pickle.dump(regressor, open(Path(f'C:\\Users\\Teja\\Downloads\\JobSalaryPrediction-RandomForestBenchmark\\JobSalaryPrediction-RandomForestBenchmark\\assets\\results\\{model_name}'), "wb"))
    data_io.save_model(regressor)


if __name__ == "__main__":
    main()
