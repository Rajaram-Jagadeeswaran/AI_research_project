import streamlit as st
import boto3
import pandas as pd
import numpy as np
from io import StringIO
import json
import time

# Set AWS credentials explicitly (not recommended for production)
aws_access_key_id = 'AKIAXYKJT4MM2U42NENU'
aws_secret_access_key = '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'
region_name = 'us-west-2'

# Initialize the SageMaker runtime client with explicit credentials
sagemaker_runtime = boto3.client(
    'sagemaker-runtime',
    region_name=region_name,
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
)

# Initialize the Bedrock client
bedrock_client = boto3.client(
    'bedrock-runtime',
    region_name=region_name,
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
)

# Define the SageMaker endpoint name
sagemaker_endpoint_name = 'xgboost-2024-08-11-23-52-29-777'

# Define the Bedrock model ID
bedrock_model_id = 'arn:aws:bedrock:us-west-2:533267211033:provisioned-model/19x5a29wnwbx'

# Function to call SageMaker endpoint
def predict_sagemaker(data):
    csv_buffer = StringIO()
    pd.DataFrame([data]).to_csv(csv_buffer, index=False, header=False)
    csv_data = csv_buffer.getvalue()
    
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=sagemaker_endpoint_name,
        ContentType='text/csv',
        Body=csv_data
    )
    result = response['Body'].read().decode()
    return result

# Function to call Bedrock model
def generate_report_bedrock(input_data):
    prompt = (
        f"Age: {input_data['Age']}, Gender: {'Male' if input_data['Gender'] == 1 else 'Female'}, "
        f"Hypertension: {'Yes' if input_data['Hypertension'] == 1 else 'No'}, Heart Disease: {'Yes' if input_data['Heart Disease'] == 1 else 'No'}, "
        f"Marital Status: {'Married' if input_data['Marital Status'] == 1 else 'Single'}, Work Type: {input_data['Work Type']}, "
        f"Residence Type: {'Urban' if input_data['Residence Type'] == 1 else 'Rural'}, Average Glucose Level: {input_data['Average Glucose Level']}, "
        f"Body Mass Index (BMI): {input_data['Body Mass Index (BMI)']}, Smoking Status: {input_data['Smoking Status']}, "
        f"Alcohol Intake: {'Yes' if input_data['Alcohol Intake'] == 1 else 'No'}, Physical Activity: {'Active' if input_data['Physical Activity'] == 1 else 'Inactive'}, "
        f"Stroke History: {'Yes' if input_data['Stroke History'] == 1 else 'No'}, Family History of Stroke: {'Yes' if input_data['Family History of Stroke'] == 1 else 'No'}, "
        f"Dietary Habits: {'Healthy' if input_data['Dietary Habits'] == 1 else 'Unhealthy'}, Stress Levels: {input_data['Stress Levels']}, "
        f"Systolic: {input_data['Systolic']}, Diastolic: {input_data['Diastolic']}, HDL: {input_data['HDL']}, LDL: {input_data['LDL']}. "
        f"This is a person’s details, I need to check if the person is at risk of stroke or not?"
    )
    
    payload = {
        "inputText": prompt,
        "textGenerationConfig": {
            "maxTokenCount": 512,
            "temperature": 0,
            "topP": 0.9
        }
    }

    response = bedrock_client.invoke_model(
        modelId=bedrock_model_id,
        contentType='application/json',
        accept='application/json',
        body=json.dumps(payload)
    )
    
    response_body = response['body'].read().decode('utf-8')
    return json.loads(response_body)

# Function to evaluate models
def evaluate_models(test_data):
    sagemaker_predictions = []
    bedrock_reports = []
    
    for data in test_data:
        start_time = time.time()
        sagemaker_prediction = predict_sagemaker(data)
        sagemaker_time = time.time() - start_time
        
        start_time = time.time()
        data['Stroke Risk'] = float(sagemaker_prediction)
        try:
            bedrock_report = generate_report_bedrock(data)
        except Exception as e:
            st.error(f"Error generating report from Bedrock: {str(e)}")
            bedrock_report = {"error": str(e)}
        bedrock_time = time.time() - start_time
        
        sagemaker_predictions.append({
            'input': data,
            'prediction': sagemaker_prediction,
            'inference_time': sagemaker_time
        })
        
        bedrock_reports.append({
            'input': data,
            'report': bedrock_report,
            'inference_time': bedrock_time
        })
    
    return sagemaker_predictions, bedrock_reports

# Define the features and corresponding dropdown options
features = {
    'Gender': ['Male', 'Female', 'Other'],
    'Age': list(range(0, 101)),
    'Hypertension': ['No', 'Yes'],
    'Heart Disease': ['No', 'Yes'],
    'Marital Status': ['No', 'Yes'],
    'Work Type': ['Private', 'Self-employed', 'Govt_job', 'Children', 'Never_worked'],
    'Residence Type': ['Urban', 'Rural'],
    'Average Glucose Level': list(np.round(np.arange(50, 300, 0.1), 1)),
    'Body Mass Index (BMI)': list(np.round(np.arange(10, 50, 0.1), 1)),
    'Smoking Status': ['formerly smoked', 'never smoked', 'smokes', 'Unknown'],
    'Alcohol Intake': ['No', 'Yes'],
    'Physical Activity': ['No', 'Yes'],
    'Stroke History': ['No', 'Yes'],
    'Family History of Stroke': ['No', 'Yes'],
    'Dietary Habits': ['Poor', 'Average', 'Good'],
    'Stress Levels': list(np.round(np.arange(0, 100.1, 0.1), 1)),
    'Symptoms': ['No', 'Yes'],
    'Systolic': list(range(50, 201)),
    'Diastolic': list(range(30, 121)),
    'HDL': list(range(10, 101)),
    'LDL': list(range(50, 201))
}

# Mapping for categorical features
categorical_mappings = {
    'Gender': {'Male': 0, 'Female': 1, 'Other': 2},
    'Hypertension': {'No': 0, 'Yes': 1},
    'Heart Disease': {'No': 0, 'Yes': 1},
    'Marital Status': {'No': 0, 'Yes': 1},
    'Work Type': {'Private': 0, 'Self-employed': 1, 'Govt_job': 2, 'Children': 3, 'Never_worked': 4},
    'Residence Type': {'Urban': 0, 'Rural': 1},
    'Smoking Status': {'formerly smoked': 0, 'never smoked': 1, 'smokes': 2, 'Unknown': 3},
    'Alcohol Intake': {'No': 0, 'Yes': 1},
    'Physical Activity': {'No': 0, 'Yes': 1},
    'Stroke History': {'No': 0, 'Yes': 1},
    'Family History of Stroke': {'No': 0, 'Yes': 1},
    'Dietary Habits': {'Poor': 0, 'Average': 1, 'Good': 2},
    'Symptoms': {'No': 0, 'Yes': 1}
}

# Streamlit app
st.title('Healthcare Optimization: Comparison Framework on Stroke Prediction')
st.write('Enter patient data to predict stroke probability and generate report.')

# Collect user input
user_input = {}
user_input['Gender'] = st.selectbox('Gender', features['Gender'])
user_input['Age'] = st.number_input('Age', min_value=0, max_value=100)
user_input['Hypertension'] = st.selectbox('Hypertension', features['Hypertension'])
user_input['Heart Disease'] = st.selectbox('Heart Disease', features['Heart Disease'])
user_input['Marital Status'] = st.selectbox('Marital Status', features['Marital Status'])
user_input['Work Type'] = st.selectbox('Work Type', features['Work Type'])
user_input['Residence Type'] = st.selectbox('Residence Type', features['Residence Type'])
user_input['Average Glucose Level'] = st.number_input('Average Glucose Level', min_value=50.0, max_value=300.0, step=0.1)
user_input['Body Mass Index (BMI)'] = st.number_input('Body Mass Index (BMI)', min_value=10.0, max_value=50.0, step=0.1)
user_input['Smoking Status'] = st.selectbox('Smoking Status', features['Smoking Status'])
user_input['Alcohol Intake'] = st.selectbox('Alcohol Intake', features['Alcohol Intake'])
user_input['Physical Activity'] = st.selectbox('Physical Activity', features['Physical Activity'])
user_input['Stroke History'] = st.selectbox('Stroke History', features['Stroke History'])
user_input['Family History of Stroke'] = st.selectbox('Family History of Stroke', features['Family History of Stroke'])
user_input['Dietary Habits'] = st.selectbox('Dietary Habits', features['Dietary Habits'])
user_input['Stress Levels'] = st.number_input('Stress Levels', min_value=0.0, max_value=100.0, step=0.1)
user_input['Symptoms'] = st.selectbox('Symptoms', features['Symptoms'])
user_input['Systolic'] = st.number_input('Systolic', min_value=50, max_value=200)
user_input['Diastolic'] = st.number_input('Diastolic', min_value=30, max_value=120)
user_input['HDL'] = st.number_input('HDL', min_value=10, max_value=100)
user_input['LDL'] = st.number_input('LDL', min_value=50, max_value=200)

# Convert categorical features to numerical values
for feature, mapping in categorical_mappings.items():
    user_input[feature] = mapping[user_input[feature]]

# Convert input to model format
input_data = [
    user_input['Age'],
    user_input['Gender'],
    user_input['Hypertension'],
    user_input['Heart Disease'],
    user_input['Marital Status'],
    user_input['Work Type'],
    user_input['Residence Type'],
    user_input['Average Glucose Level'],
    user_input['Body Mass Index (BMI)'],
    user_input['Smoking Status'],
    user_input['Alcohol Intake'],
    user_input['Physical Activity'],
    user_input['Stroke History'],
    user_input['Family History of Stroke'],
    user_input['Dietary Habits'],
    user_input['Stress Levels'],
    user_input['Symptoms'],
    user_input['Systolic'],
    user_input['Diastolic'],
    user_input['HDL'],
    user_input['LDL']
]

# Ensure data is in the correct format
input_data = np.array(input_data).reshape(1, -1)

# Make prediction
if st.button('Predict and Compare'):
    try:
        # Run evaluation on a single input for demonstration
        test_data = [user_input]
        sagemaker_predictions, bedrock_reports = evaluate_models(test_data)

        # Display predictions and reports
        st.subheader('SageMaker Predictions')
        st.write(sagemaker_predictions)

        st.subheader('Bedrock Reports')
        st.write(bedrock_reports)

        # Comparison metrics
        sagemaker_inference_times = [pred['inference_time'] for pred in sagemaker_predictions]
        bedrock_inference_times = [report['inference_time'] for report in bedrock_reports]

        st.subheader('Performance Comparison')
        st.write(f"SageMaker average inference time: {sum(sagemaker_inference_times) / len(sagemaker_inference_times):.2f} seconds")
        st.write(f"Bedrock average inference time: {sum(bedrock_inference_times) / len(bedrock_inference_times):.2f} seconds")

        # Bar chart for inference times
        inference_times_df = pd.DataFrame({
            'Model': ['SageMaker', 'Bedrock'],
            'Average Inference Time (s)': [
                sum(sagemaker_inference_times) / len(sagemaker_inference_times),
                sum(bedrock_inference_times) / len(bedrock_inference_times)
            ]
        })
        st.bar_chart(inference_times_df.set_index('Model'))

        # Display risk prediction
        probability_score = float(sagemaker_predictions[0]['prediction'])
        if probability_score >= 0.5:
            st.success('The model predicts that the person is at risk of a stroke.')
        else:
            st.info('The model predicts that the person is not at risk of a stroke.')
        st.write(f'Probability Score: {probability_score}')

    except Exception as e:
        st.write("An error occurred while predicting:", str(e))

# About section
st.sidebar.title("About")
st.sidebar.info(
    """
    This app predicts the risk of a stroke based on various health and lifestyle factors.
    It uses a machine learning model trained on historical data.
    """
)

# Instructions section
st.sidebar.title("Instructions")
st.sidebar.info(
    """
    Please select the appropriate options for each feature from the dropdown menus.
    Once all selections are made, click on the 'Predict and Compare' button to see the prediction result and confidence score.
    """
)
