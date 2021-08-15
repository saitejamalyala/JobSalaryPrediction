# import data_io
import data_io
import numpy as np
import pandas as pd


def main():
    """
    Predict on validation set
    """
    print("Loading the Regressor")
    regressor = data_io.load_model()

    # print("Making predictions")
    # valid = data_io.get_valid_df()
    # predictions = regressor.predict(valid)
    # predictions = predictions.reshape(len(predictions), 1)

    # print("Writing predictions to file")
    # data_io.write_submission(predictions)

    # on test data
    print("Making predictions on test data")
    test = data_io.get_test_df()
    print(test.head())
    # print(test.columns)
    predictions = regressor.predict(test)
    # predictions = predictions.reshape(len(predictions), 1)
    # data_io.write_submission_test(predictions)

    """
        ground_truth = pd.read_csv(data_io.get_paths()['test_groud_truth'])
        print(np.shape(predictions),len(ground_truth))
        print(ground_truth.head())
        rmse = np.sqrt(np.mean((predictions[0] - ground_truth['SalaryNormalized'])**2))
        mae = np.mean(np.abs(predictions[0] - ground_truth['SalaryNormalized']))
        print(f'RMSE : {rmse}, MAE: {mae}')
    """

    # on train data
    """
        print("Making predictions on train data")
        train = data_io.get_train_df()
        predictions = regressor.predict(train)
        mse = np.mean((predictions - train["SalaryNormalized"])**2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - train["SalaryNormalized"]))

        print(f'mse:{mse:.2f}, rmse:{rmse:.2f}, mae: {mae:.2f}')
    """


def get_model():
    return data_io.load_model()


def get_prediction(test_data: pd.DataFrame) -> np.ndarray:
    model = data_io.load_model()
    return model.predict(test_data)


if __name__ == "__main__":
    main()
    regressor = get_model()
    data_to_predict = {
        "Id": 0,
        "Title": "Systems Engineering Consultant",
        "FullDescription": "The role will involve liaison with clients to define requirements and assumptions and provide advice on a range of Enterprise Architecture, Systems Engineering and SOSA issues. You will be required to apply systems engineering principles, such as definition and clarification of requirements, collation of data/information, and use of appropriate tools for data capture, analysis and presentation. You ll also be responsible for: Successfully delivering consultancy assignments, either individually or as part of a team Supporting the development of bids and responses to tenders Supporting the development of thought leadership material Requirements for CIS Systems Engineering Consultant Degree qualified in an Engineering/Mathematical or Scientific Discipline or have equivalent experience. A relevant higher degree and/or professional qualification, e.g. CEng, would be an advantage. Practical development of enterprise architectures and SOSA  identifying business processes, requirements, capabilities, development of asis and tobe reference architectures. Application of model driven engineering approaches to address stakeholder concerns, requirements and needs. Experience of the Systems Engineering lifecycle. Experience of Systems Engineering modelling tools and languages such as UML, Sisal, Somali, Modal, MOOD. Experience of abstraction and service based approaches to architecture. Experience of developing Information architectures. Excellent interpersonal skills capable of engaging across a wide range of stakeholders. Effective written and verbal communication skills, including excellent presentation and reportwriting skills. Flexible approach to work, able to work well alone and as part of a team. Desirable: Familiarity with MoD CADMID process for equipment procurement. Previous consultancy experience associated with military communications and information systems (CIS, ICT, NEC)  this may include a relevant MSc, PhD and/or work experience. Familiar with Object Oriented Design, Open systems and interfaces. Security  All candidates must be in a position to obtain UK security clearance CIS Systems Engineering Consultant Bristol  South West Salary",
        "LocationRaw": "Bristol, South West, South West",
        "LocationNormalized": "Bristol",
        "ContractType": "",
        "ContractTime": "",
        "Company": "",
        "Category": "",
        "SourceName": "",
    }
    test_df = pd.DataFrame(data_to_predict, index=[0])
    print(f"Prediction: {regressor.predict(test_df)}")
    # print(get_prediction(test_data=test_df))
