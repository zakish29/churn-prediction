import streamlit as st
import eda
import prediction
import tensorflow as tf


nav = st.sidebar.selectbox('Pilih halaman:', ('EDA', 'Predict Churn'))

if nav == 'EDA':
    eda.run()
else:
    prediction.run()

