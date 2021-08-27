import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
from data_io import get_paths, load_model
import numpy as np

st.set_page_config(layout="wide", initial_sidebar_state="expanded")
# from src.predict import get_prediction
html_header = """
<div>
<h1 style="color:black;text-align:center;"> Interview Demo </h1> 
</div>"""
st.markdown(html_header, unsafe_allow_html=True)

st.header("Demo of **Job salary prediction **")

# print(os.path.join('.','assets','images','Survey-img.jpg'))
# image_main = Image.open(os.path.join('.','assets','images','Research-img.jpg'))

# st.image(image_main,caption='copyright: https://www.payscale.com/salary-calculator',)

st.write(
    """
### Data used for training: [Adzuna Job salary prediction Data set](https://www.kaggle.com/adzuna/job-salary-prediction)
"""
)
st.write('---')
st.sidebar.title("User Input Data")
st.sidebar.markdown(
    """
[Example of User Input Data](https://mega.nz/file/gskTWIyA#NSmmJxgqfRsU3jsrvTg3_6yg__fEn_Ny2ma4qpkUiTw)
"""
)

# collect data
use_default_data = st.sidebar.checkbox("Use default data", value=True,
                    help=
"""
    Id - A unique identifier for each job ad

    Title - A freetext field supplied to us by the job advertiser as the Title of the job ad.  Normally this is a summary of the job title or role.

    FullDescription - The full text of the job ad as provided by the job advertiser.  Where you see ***s, we have stripped values from the description in order to ensure that no salary information appears within the descriptions.  There may be some collateral damage here where we have also removed other numerics.

    LocationRaw - The freetext location as provided by the job advertiser.

    LocationNormalized - Adzunas normalised location from within our own location tree, interpreted by us based on the raw location.  Our normaliser is not perfect!

    ContractType - full_time or part_time, interpreted by Adzuna from description or a specific additional field we received from the advertiser.

    ContractTime - permanent or contract, interpreted by Adzuna from description or a specific additional field we received from the advertiser.

    Company - the name of the employer as supplied to us by the job advertiser.

    Category - which of 30 standard job categories this ad fits into, inferred in a very messy way based on the source the ad came from.  We know there is a lot of noise and error in this field.

    SalaryRaw - the freetext salary field we received in the job advert from the advertiser.

    SalaryNormalised - the annualised salary interpreted by Adzuna from the raw salary.  Note that this is always a single value based on the midpoint of any range found in the raw salary.  This is the value we are trying to predict.

    SourceName - the name of the website or advertiser from whom we received the job advert. 

    All of the data is real, live data used in job ads so is clearly subject to lots of real world noise, including but not limited to: ads that are not UK based, salaries that are incorrectly stated, fields that are incorrectly normalised and duplicate adverts.
"""
)
uploaded_file = st.sidebar.file_uploader("Upload your data file", type=".csv")

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    st.write("Data file is uploaded")
if use_default_data:
    with st.spinner("Loading default data...."):
        test_df = pd.read_csv(get_paths()["mini_test_data_path"])
        st.success("Default data is used")
else:

    def custom_user_ip():
        Title = st.sidebar.text_input(
            label="Title", value="Systems Engineering Consultant"
        )
        FullDescription = st.sidebar.text_area(
            label="Full Job Description",
            value="CIS Systems Engineering Consultant Bristol  South West Salary ****  **** Our client is looking for an experienced CIS Systems Engineering Consultant . \
            As CIS Systems Engineering Consultant you would work on a range of projects for MOD, other government departments and the defence industry, particularly in the areas of: \
            systems analysis, design, architecture definition, development, interoperability and through life management. \
            The role will involve liaison with clients to define requirements and assumptions and provide advice on a range of Enterprise Architecture,\
            Systems Engineering and SOSA issues. You will be required to apply systems engineering principles, such as definition and clarification of requirements,\
            collation of data/information, and use of appropriate tools for data capture, analysis and presentation.\
            You ll also be responsible for: Successfully delivering consultancy assignments, \
            either individually or as part of a team Supporting the development of bids and responses to tenders Supporting the development of thought leadership material Requirements for\
            CIS Systems Engineering Consultant Degree qualified in an Engineering/Mathematical or Scientific Discipline or have equivalent experience. \
            A relevant higher degree and/or professional qualification, e.g. CEng, would be an advantage. \
            Practical development of enterprise architectures and SOSA  identifying business processes, requirements, capabilities, development of asis and tobe reference architectures.\
            Application of model driven engineering approaches to address stakeholder concerns, requirements and needs.\
            Experience of the Systems Engineering lifecycle. Experience of Systems Engineering modelling tools and languages such as UML, Sisal, Somali, Modal, MOOD. \
            Experience of abstraction and service based approaches to architecture. Experience of developing Information architectures.\
            Excellent interpersonal skills capable of engaging across a wide range of stakeholders. \
            Effective written and verbal communication skills, including excellent presentation and reportwriting skills. \
            Flexible approach to work, able to work well alone and as part of a team. Desirable: Familiarity with MoD CADMID process for equipment procurement. \
            Previous consultancy experience associated with military communications and information systems (CIS, ICT, NEC)  this may include a relevant MSc, PhD and/or work experience.\
            Familiar with Object Oriented Design, Open systems and interfaces. Security  All candidates must be in a position to obtain UK security clearance \
            CIS Systems Engineering Consultant Bristol  South West Salary ****  ****",
        )

        LocationRaw = st.sidebar.text_input(
            label="Raw Location", value="Bristol, South West, South West", max_chars=100
        )
        LocationNormalized = st.sidebar.text_input(
            label="Location Normalized", value="Bristol", max_chars=50
        )
        st.sidebar.write(
            """- Please Enter locations from **UK** to get approriate results"""
        )
        data_to_predict = {
            "Id": "NA",
            "Title": Title,
            "FullDescription": FullDescription,
            "LocationRaw": LocationRaw,
            "LocationNormalized": LocationNormalized,
            "ContractType": "NA",
            "ContractTime": "NA",
            "Company": "NA",
            "Category": "NA",
            "SourceName": "NA",
        }
        test_df = pd.DataFrame(data_to_predict, index=[0])
        return test_df

    test_df = custom_user_ip()

st.subheader("User Job Data to predict salary")
st.dataframe(data=test_df.head())


def predict_callback():
    st.write('---')
    model = load_model()
    with st.spinner("Prediction engine running...."):
        prediction = model.predict(test_df)
    # prediction = get_prediction(test_data=test_df)
    disp_pred = pd.DataFrame(columns=["Id", "Title", "Location", "Predicted Salary"])
    disp_pred["Id"] = test_df["Id"]
    disp_pred["Predicted Salary"] = pd.DataFrame(data=prediction,dtype=int)
    disp_pred["Title"] = test_df["Title"]
    disp_pred["Location"] = test_df["LocationNormalized"]
    # print(disp_pred.head())
    st.subheader("Predicted salaries for respective Titles")
    st.dataframe(data=disp_pred)


if st.button("Predict Salary"):
    predict_callback()
