import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from keras import layers

# Load All Files
with open('pipeline.pkl', 'rb') as file_1:
  model_pipeline = pickle.load(file_1)

model_churn = load_model('./churn_model.h5')



# Run function stelah loading

def run():
  with st.form(key='Customer Data Input'):
        # define the user inputs
        user_id = st.text_input('User ID')
        age = st.number_input('Age', value=18)
        gender = st.radio('Gender', ['M', 'F'])
        region_category = st.selectbox('Region Category', ['City', 'Village', 'Town'])
        membership_category = st.selectbox('Membership Category', ['No Membership', 'Basic Membership', 'Silver Membership', 'Premium Membership', 'Gold Membership', 'Platinum Membership'])
        day = st.number_input('Joining Day', value=1, min_value=1, max_value=31)
        month = st.number_input('Joining Month', value=1, min_value=1, max_value=12)
        year = st.number_input('Joining Year', value=1, min_value=1, max_value=31)
        last_visit_time_hh = st.number_input('Last Time Visit (Hour)', value=0, min_value=00, max_value=23)
        last_visit_time_mm = st.number_input('Last Time Visit (Minutes)', value=0, min_value=00, max_value=60)
        joined_through_referral = st.radio('Joined through referral?', ['Yes', 'No'])
        preferred_offer_types = st.selectbox('Preferred Offer Types', ['Without Offers', 'Credit/Debit Card Offers', 'Gift Vouchers/Coupons'])
        medium_of_operation = st.selectbox('Medium of Operation', ['Desktop', 'Smartphone', 'Both'])
        internet_option = st.selectbox('Internet Option', ['Wi-Fi', 'Fiber_Optic', 'Mobile_Data'])
        days_since_last_login = st.number_input('Days since Last Login', value=0)
        avg_time_spent = st.number_input('Average Time Spent', value=0.0, format='%.2f')
        avg_transaction_value = st.number_input('Average Transaction Value', value=0.0, format='%.2f')
        avg_frequency_login_days = st.number_input('Average Frequency Login Days', value=0.0, format='%.2f')
        points_in_wallet = st.number_input('Points in Wallet', value=0.0, format='%.2f')
        used_special_discount = st.radio('Used Special Discount?', ['Yes', 'No'])
        offer_application_preference = st.radio('Offer Application Preference?', ['Yes', 'No'])
        past_complaint = st.radio('Past Complaint?', ['Yes', 'No'])
        complaint_status = st.selectbox('Complaint Status', ['No Information Available', 'Not Applicable', 'Unsolved', 'Solved', 'Solved in Follow-up'])
        feedback = st.selectbox('Feedback', ['Poor Website', 'Poor Customer Service', 'Too many ads', 'Poor Product Quality', 'No reason specified', 'Products always in Stock', 'Reasonable Price', 'Quality Customer Care', 'User Friendly Website'])

        submitted = st.form_submit_button('Predict')
        
        # create the dataframe

  data = {'user_id': user_id,
          'age': age, 
          'gender': gender, 
          'region_category': region_category, 
          'membership_category': membership_category, 
          'joined_through_referral': joined_through_referral, 
          'preferred_offer_types': preferred_offer_types, 
          'medium_of_operation': medium_of_operation, 
          'internet_option': internet_option, 
          'day' : day,
          'month' : month,
          'year' : year,
          'last_visit_time_hh' : last_visit_time_hh,
          'last_visit_time_mm' : last_visit_time_mm,
          'days_since_last_login': days_since_last_login, 
          'avg_time_spent': avg_time_spent, 
          'avg_transaction_value': avg_transaction_value, 
          'avg_frequency_login_days': avg_frequency_login_days, 
          'points_in_wallet': points_in_wallet, 
          'used_special_discount': used_special_discount, 
          'offer_application_preference': offer_application_preference, 
          'past_complaint': past_complaint, 
          'complaint_status': complaint_status, 
          'feedback': feedback
          }
    
  data_inf = pd.DataFrame([data])
  st.dataframe(data_inf)

  if submitted:
        data_inf_final = data_inf.drop(['user_id','avg_transaction_value', 'points_in_wallet'], axis=1)
        data_inf_transform = model_pipeline.transform(data_inf_final)
        data_inf_transform

        y_pred_inf = model_churn.predict(data_inf_transform)
        y_pred_inf = np.where(y_pred_inf >= 0.5, 1, 0)

        if y_pred_inf.any() == 1:
            st.write('## The Customer probably will CHURN')
        else:
            st.write('## The Customer probably will NOT Churn')

# calling function
if __name__ == '__main__':
   run()
