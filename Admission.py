# App to predict the chances of admission using a pre-trained ML model in Streamlit

# Import libraries
import streamlit as st
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')


# Set up the app title and image
st.title('Graduate Admission Predictor 🌟')
st.image('admission.jpg', use_column_width = True, 
         caption = "Predict your chances of admission based on your profile")

# NOTE: In Streamlit, use_column_width=True within st.image automatically 
# adjusts the image width to match the width of the column or container in 
# which the image is displayed. This makes the image responsive to different 
# screen sizes and layouts, ensuring it scales nicely without needing to 
# specify exact pixel dimensions.

st.write("This app uses multiple inputs to predict the probability of admission to graduate school.") 

# Reading the pickle file that we created before 
model_pickle = open('reg_admission.pickle', 'rb') 
reg_model = pickle.load(model_pickle) 
model_pickle.close()

# Load the default dataset
default_df = pd.read_csv('Admission_Predict.csv')

# Sidebar for user inputs with an expander
with st.sidebar.form("user_inputs_form"):
    st.header("Enter Your Profile Details")
    gre_score = st.number_input('GRE Score', min_value=0, max_value=340, value=320, step=1, help="Range: 0-340")
    toefl_score = st.number_input('TOEFL Score', min_value=0, max_value=120, value=100, step=1, help="Range: 0-120")
    cgpa = st.number_input('CGPA', min_value=0.0, max_value=10.0, value=8.0, step=0.1, help="Range: 0-10")
    research = st.selectbox('Research Experience', options=['Yes', 'No'], help="Have you conducted any research?")
    univ_rating = st.slider('University Rating', min_value=1, max_value=5, value=3, step=1, help="Rating of target university (1 to 5)")
    sop = st.slider('Statement of Purpose (SOP)', min_value=1.0, max_value=5.0, value=3.5, step=0.5)
    lor = st.slider('Letter of Recommendation (LOR)', min_value=1.0, max_value=5.0, value=3.5, step=0.5)
    submit_button = st.form_submit_button("Predict")


# Encode the inputs for model prediction
encode_df = default_df.copy()
encode_df = encode_df.drop(columns=['Chance of Admit'])

# Combine the list of user data as a row to default_df
encode_df.loc[len(encode_df)] = [gre_score, toefl_score, univ_rating, sop, lor, cgpa, research]

# Create dummies for encode_df
encode_dummy_df = pd.get_dummies(encode_df)

# Extract encoded user data
user_encoded_df = encode_dummy_df.tail(1)

# Get the prediction with its intervals
alpha = 0.1 # For 90% confidence level
prediction, intervals = reg_model.predict(user_encoded_df, alpha = alpha)
pred_value = prediction[0]
lower_limit = intervals[:, 0]
upper_limit = intervals[:, 1]

# Ensure limits are within [0, 1]
lower_limit = max(0, lower_limit[0][0])
upper_limit = min(1, upper_limit[0][0])

# Show the prediction on the app
st.write("## Predicting Admission Chance...")

# Display results using metric card
st.metric(label = "Predicted Admission Probability", value = f"{pred_value * 100:.2f}%")
st.write("With a 90% confidence interval:")
st.write(f"**Confidence Interval**: [{lower_limit* 100:.2f}%, {upper_limit* 100:.2f}%]")

# Additional tabs for DT model performance
st.subheader("Model Insights")
tab1, tab2, tab3, tab4 = st.tabs(["Feature Importance", 
                            "Histogram of Residuals", 
                            "Predicted Vs. Actual", 
                            "Coverage Plot"])
with tab1:
    st.write("### Feature Importance")
    st.image('feature_imp.svg')
    st.caption("Relative importance of features in prediction.")
with tab2:
    st.write("### Histogram of Residuals")
    st.image('residual_plot.svg')
    st.caption("Distribution of residuals to evaluate prediction quality.")
with tab3:
    st.write("### Plot of Predicted Vs. Actual")
    st.image('pred_vs_actual.svg')
    st.caption("Visual comparison of predicted and actual values.")
with tab4:
    st.write("### Coverage Plot")
    st.image('coverage.svg')
    st.caption("Range of predictions with confidence intervals.")

