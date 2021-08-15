Kaggle - Job Salary Prediction
==============================

This repo contains sample code for the [Job Salary Prediction competition](https://www.kaggle.com/c/job-salary-prediction/), hosted by [Kaggle](http://www.kaggle.com) with [Adzuna](http://www.adzuna.co.uk/).


Executing this benchmark requires Python 2.7 along with the following packages:

- pandas (version >=10.1)
- sklearn
- numpy
- streamlit

To run the benchmark,

1. [Download the data](https://www.kaggle.com/c/job-salary-prediction/data)
2. Modify SETTINGS.json to point to the training and validation data on your system, as well as a place to save the trained model and a place to save the submission
3. Train the model by running `python src/train.py`
4. Make predictions on the validation set by running `python src/predict.py`
5. To run the streamlit app `streamlit run src/app.py`
6. After 5 open the [Browser](http://localhost:8501/)

This benchmark took approximately 0.5 hours to execute on a Windows 10 laptop with 12GB of RAM and 8 cores at 2GHz.
