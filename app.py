import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. Load Saved Assets ---
@st.cache_resource # Cache resource to avoid reloading on every rerun
def load_model_and_assets():
    scaler = joblib.load('scaler.joblib')
    expected_columns = joblib.load('expected_columns.joblib')
    # Load the best performing model (KNeighborsClassifier Undersampled)
    model = joblib.load('KNeighborsClassifier_Undersampled_model.joblib')
    return scaler, expected_columns, model

scaler, expected_columns, knn_undersampled_model = load_model_and_assets()

# --- 2. Define the Recommendation Engine (copy-paste from notebook) ---
def recommend_contact_strategy(customer_data, predicted_outcome):
    """
    Recommends an optimal contact channel and strategy based on customer features and predicted repayment outcome.
    Assumes customer_data is a Series representing a row from the original, unscaled DataFrame.
    """

    recommendation = {
        'Channel': 'Unspecified',
        'Strategy': 'General follow-up'
    }

    # Extract relevant original features
    days_past_due = customer_data['Days_Past_Due']
    response_rate = customer_data['Response_Rate']
    complaint_flag = customer_data['Complaint_Flag']
    payment_made = customer_data['Payment_Made_Last_30_Days']
    age = customer_data['Age']
    last_contact_channel = customer_data['Last_Contact_Channel']
    num_calls = customer_data['Number_of_Calls']
    credit_score = customer_data['Credit_Score']
    income = customer_data['Income']

    # Rule 1: Complaint Flag - prioritize resolution and less intrusive methods
    if complaint_flag:
        recommendation['Channel'] = 'Mail or Email'
        recommendation['Strategy'] = 'Address complaint first, then gentle reminder. Avoid aggressive contact.'
        return recommendation

    # Rule 2: High Risk (predicted_outcome=1 means likely to not repay) & Non-Responsive - urgent direct contact
    if predicted_outcome == 1 and days_past_due > 45 and response_rate < 0.2:
        if last_contact_channel in ['Email', 'SMS'] and num_calls < 5: # If digital failed and not too many calls
            recommendation['Channel'] = 'Phone'
            recommendation['Strategy'] = 'Urgent, direct conversation needed to understand issues. High priority.'
        elif last_contact_channel == 'Phone' and num_calls >= 5:
            # If phone calls are not working, try a different channel like Mail for formal notice
            recommendation['Channel'] = 'Mail'
            recommendation['Strategy'] = 'Formal notice required, consider alternative resolution paths.'
        else:
            recommendation['Channel'] = 'Phone'
            recommendation['Strategy'] = 'Immediate phone contact. Escalation likely needed if no response.'
        return recommendation

    # Rule 3: Customer with Recent Payment - Gentle follow-up, acknowledge payment
    if payment_made:
        recommendation['Channel'] = 'Email or SMS'
        recommendation['Strategy'] = 'Thank for recent payment, gentle reminder for outstanding. Maintain goodwill.'
        return recommendation

    # Rule 4: Older Customers / Lower Credit Score - More personalized, respectful approach
    if age >= 60 or credit_score < 550:
        if days_past_due > 30:
            recommendation['Channel'] = 'Phone or Mail'
            recommendation['Strategy'] = 'Personalized call to understand situation, offer flexible solutions. Avoid aggressive language.'
        else:
            recommendation['Channel'] = 'Mail'
            recommendation['Strategy'] = 'Gentle reminder via mail, offer assistance and clear contact options.'
        return recommendation

    # Rule 5: Medium Risk / Moderate Days Past Due - Try preferred channel first, then escalate
    if predicted_outcome == 1 and 15 < days_past_due <= 45:
        if last_contact_channel in ['Phone', 'Email', 'SMS']:
            recommendation['Channel'] = last_contact_channel  # Use last successful channel or attempt
            recommendation['Strategy'] = 'Standard follow-up. Reinforce consequences of non-payment.'
        else:
            recommendation['Channel'] = 'Email or SMS'
            recommendation['Strategy'] = 'Digital reminder, offering self-service payment options.'
        return recommendation

    # Rule 6: Low Risk / Early Days Past Due - Automated, non-intrusive reminders
    if days_past_due <= 15:
        recommendation['Channel'] = 'SMS or Email'
        recommendation['Strategy'] = 'Automated, friendly reminder of upcoming or recent due date.'
        return recommendation

    # Rule 7: High Response Rate - Leverage their responsiveness
    if response_rate >= 0.7:
        recommendation['Channel'] = last_contact_channel if last_contact_channel != 'Mail' else 'Email'
        recommendation['Strategy'] = 'Direct and concise communication, they are likely to respond to their preferred channel.'
        return recommendation

    # Default / General Strategy if no specific rules apply
    if predicted_outcome == 1:
        recommendation['Channel'] = 'Email or Phone'
        recommendation['Strategy'] = 'Standard follow-up, ascertain reasons for non-payment.'
    else: # predicted_outcome == 0 (likely to repay), but still past due
        recommendation['Channel'] = 'Email or SMS'
        recommendation['Strategy'] = 'Soft reminder, offer payment flexibility.'

    return recommendation


# --- 3. Streamlit UI Elements ---
st.title('Credit Risk Prediction & Contact Strategy Recommender')
st.write('Enter customer details to predict repayment outcome and get contact strategy recommendations.')

# Input fields for user data (mimic original dataframe features)
# Ensure these inputs match the features used in training, including their types.

# Numerical inputs (need to be scaled)
age = st.slider('Age', 18, 70, 30)
income = st.number_input('Income', 10000.0, 200000.0, 60000.0, step=1000.0)
loan_amount = st.number_input('Loan Amount', 1000.0, 50000.0, 15000.0, step=500.0)
outstanding_balance = st.number_input('Outstanding Balance', 0.0, 50000.0, 5000.0, step=100.0)
days_past_due = st.slider('Days Past Due', 0, 90, 10)
num_calls = st.slider('Number of Calls', 0, 20, 2)
response_rate = st.slider('Response Rate', 0.0, 1.0, 0.5, step=0.01)
credit_score = st.slider('Credit Score', 300, 850, 650)

# Binary inputs
payment_made = st.checkbox('Payment Made Last 30 Days?')
complaint_flag = st.checkbox('Complaint Flag?')

# Categorical inputs (need one-hot encoding)
bank_code = st.selectbox('Bank Code', ['BankA', 'BankB', 'BankC', 'BankD'])
occupation = st.selectbox('Occupation', ['Engineer', 'Doctor', 'Teacher', 'Artist', 'Sales', 'Manager', 'Analyst', 'Clerk', 'Retired'])
last_contact_channel = st.selectbox('Last Contact Channel', ['Phone', 'Email', 'SMS', 'Mail', 'App'])
region = st.selectbox('Region', ['North', 'South', 'East', 'West', 'Central', 'Metropolitan'])

# --- 4. Preprocess User Input for Model Prediction ---
def preprocess_input(user_data):
    # Create a DataFrame from user input
    input_df = pd.DataFrame([user_data])

    # One-hot encode categorical features
    categorical_cols = ['Bank_Code', 'Occupation', 'Last_Contact_Channel', 'Region']
    input_df_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=False)

    # Convert boolean columns to int
    input_df_encoded['Payment_Made_Last_30_Days'] = input_df_encoded['Payment_Made_Last_30_Days'].astype(int)
    input_df_encoded['Complaint_Flag'] = input_df_encoded['Complaint_Flag'].astype(int)

    # Calculate engineered features BEFORE scaling (if they rely on original values)
    input_df_encoded['Debt_to_Income_Ratio'] = input_df_encoded['Outstanding_Balance'] / input_df_encoded['Income']
    input_df_encoded['Loan_to_Outstanding_Ratio'] = input_df_encoded['Loan_Amount'] / (input_df_encoded['Outstanding_Balance'] + 1e-6)
    
    # Age_Group feature creation (based on original age)
    age_bins = [18, 30, 45, 60, 71]
    age_labels = ['Young', 'Adult', 'Middle-aged', 'Senior']
    input_df_encoded['Age_Group'] = pd.cut(input_df['Age'], bins=age_bins, labels=age_labels, right=False)
    input_df_encoded = pd.get_dummies(input_df_encoded, columns=['Age_Group'], drop_first=False)

    # High_Risk_Customer flag (based on original values)
    days_past_due_threshold = 30
    credit_score_threshold = 500
    input_df_encoded['High_Risk_Customer'] = (
        (input_df['Days_Past_Due'] > days_past_due_threshold) &
        (input_df['Credit_Score'] < credit_score_threshold)
    ) | input_df['Complaint_Flag']
    input_df_encoded['High_Risk_Customer'] = input_df_encoded['High_Risk_Customer'].astype(int)

    # Scale numerical features
    numerical_cols = [
        'Age',
        'Income',
        'Loan_Amount',
        'Outstanding_Balance',
        'Days_Past_Due',
        'Number_of_Calls',
        'Response_Rate',
        'Credit_Score'
    ]
    input_df_encoded[numerical_cols] = scaler.transform(input_df_encoded[numerical_cols])
    
    # Align columns with training data to ensure all features are present and in correct order
    # Add missing columns with 0 and reorder
    for col in expected_columns:
        if col not in input_df_encoded.columns:
            input_df_encoded[col] = 0
    input_processed = input_df_encoded[expected_columns]

    # Ensure boolean columns are int (might be object after get_dummies for new columns)
    for col in input_processed.select_dtypes(include='bool').columns:
        input_processed[col] = input_processed[col].astype(int)

    return input_processed

# Collect user input
user_input = {
    'Age': age,
    'Income': income,
    'Loan_Amount': loan_amount,
    'Outstanding_Balance': outstanding_balance,
    'Days_Past_Due': days_past_due,
    'Number_of_Calls': num_calls,
    'Response_Rate': response_rate,
    'Payment_Made_Last_30_Days': payment_made,
    'Credit_Score': credit_score,
    'Complaint_Flag': complaint_flag,
    'Bank_Code': bank_code,
    'Occupation': occupation,
    'Last_Contact_Channel': last_contact_channel,
    'Region': region
}

if st.button('Predict & Recommend'):
    # Preprocess input data
    processed_input = preprocess_input(user_input)

    # Make prediction
    prediction_proba = knn_undersampled_model.predict_proba(processed_input)[:, 1][0]
    prediction_class = knn_undersampled_model.predict(processed_input)[0]

    st.subheader('Prediction Result:')
    if prediction_class == 1:
        st.error(f"**Predicted Outcome: Likely Not to Repay** (Probability: {prediction_proba:.2f})")
    else:
        st.success(f"**Predicted Outcome: Likely to Repay** (Probability: {prediction_proba:.2f})")

    # Get contact strategy recommendation
    # For the recommendation function, we need the original (unscaled) user data.
    # We'll create a pandas Series from user_input for this.
    original_customer_data = pd.Series(user_input)
    recommendation = recommend_contact_strategy(original_customer_data, prediction_class)

    st.subheader('Contact Strategy Recommendation:')
    st.info(f"**Recommended Channel:** {recommendation['Channel']}")
    st.info(f"**Recommended Strategy:** {recommendation['Strategy']}")
