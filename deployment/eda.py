import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image
import json

st.set_page_config(
    page_title=' Customer Churn ',
    layout= 'wide',
    initial_sidebar_state= 'expanded'
)


# Create Run Function
def run():
    # Membuat Title
    st.title('Churn Prediction')

    # Subheader
    st.subheader('Churn')

    # Menambahkan gambar
    image = Image.open('./growth.jpg')
    st.image(image)

    # Menambahkan deskripsi
    st.write('## Introduction')
    st.write(
    '''
    Customer churn, or the rate at which customers stop doing business with a company, is a significant problem for businesses across various industries. Losing customers can have a substantial negative impact on a company's revenue, market share, and reputation. Therefore, predicting and preventing customer churn is a crucial business goal. In recent years, Artificial Neural Networks (ANN) have become a popular choice for building predictive models due to their ability to identify complex patterns and relationships in data.

The primary objective of this project is `to build an ANN model to predict customer churn and achieve high recall to decrease false negatives`. Recall, also known as sensitivity, is a critical metric for evaluating a predictive model's performance in identifying positive cases. High recall means that the model can correctly identify a large proportion of actual churn cases, reducing the number of customers who leave without being detected. In contrast, false negatives occur when the model fails to identify a positive case, leading to missed opportunities to retain customers.

By building an accurate ANN model, we can identify the customers most likely to churn and take appropriate actions to retain them. This project's outcome can provide valuable insights into customer behavior and assist businesses in developing effective strategies to reduce customer churn rates, enhance customer loyalty, and increase profitability.
    
    '''
    )
    # Membuat garis lurus
    st.markdown('-'*42)

    st.write('## Table')
    # Show DF
    df = pd.read_csv('./churn.csv')
    st.dataframe(df)
    
    st.write('## Churn Trend in Year and Month ')
    # create new column
    temp = df.copy()
    temp['year'] = pd.DatetimeIndex(temp['joining_date']).year
    temp['month'] = pd.DatetimeIndex(temp['joining_date']).month
    temp['day'] = pd.DatetimeIndex(temp['joining_date']).day

    # plotting
    fig, (ax1,ax2) = plt.subplots(1,2, figsize=(25,5))

    sns.countplot(temp, x='year', hue='churn_risk_score', fill=True,ax=ax1, palette="crest")
    sns.countplot(temp, x='month', hue='churn_risk_score', fill=True,ax=ax2, palette="crest")
    ax1.set_title('Join Year Customer')
    ax2.set_title('Join Month Customer')

    # Display the plot using Streamlit
    st.pyplot(fig)


    st.write('## Feedback Towards Churn')
    # Create a dropdown menu to select a specific feedback value
    selected_feedback = st.selectbox("Select Feedback Value", df['feedback'].unique())

    # Filter the dataframe based on the selected feedback value
    filtered_df = df[df['feedback'] == selected_feedback]

    # Use Plotly Express to create a countplot
    fig = px.histogram(filtered_df, x='feedback', color='churn_risk_score', color_discrete_sequence=["#58508d", "#bc5090"], nbins=20)

    # Configure the layout of the plot
    fig.update_layout(
        title='Distribution of Churn/Not Based on Feedbacks',
        xaxis_title='Feedback',
        yaxis_title='Count',
        legend_title='Churn Risk Score',
        font=dict(size=12),
        margin=dict(l=50, r=50, t=50, b=50),
        plot_bgcolor='white'
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig)

    # Filter the dataframe based on the selected feedback value
    filtered_df = df[df['feedback'] == selected_feedback]

    # Count the number of customers with a churn risk score of 1
    churn_count = len(filtered_df[filtered_df['churn_risk_score'] == 1])

    # Display the number of customers in a text box
    st.text(f"Number of customers with feedback {selected_feedback}: {len(filtered_df)}")
    st.text(f"Number of customers with feedback {selected_feedback} and total churn customer: {churn_count}")



# calling function
if __name__ == '__main__':
   run()