import pickle
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

title = 'Predict Amount of Commuter Passenger 🚉'
subtitle = 'Predict Amount of Commuter Passenger using machine learning 🚄🚄 '

def main():
    st.set_page_config(layout="centered", page_icon='🚉', page_title='Lets Predict Amount of Commuter Passenger!')
    st.title(title)
    st.write(subtitle)

    form = st.form("Data Input")
    Region = form.selectbox('Region', ['Jabodetabek', 'Non Jabodetabek (Jawa)', 'Jawa (Jabodetabek+Non Jabodetabek)', 'Sumatera'])
    start_date = form.date_input('Start Date')
    end_date = form.date_input('End Date')

    submit = form.form_submit_button("Predict")  # Add a submit button

    if submit:
        data = {
            'Kode Wilayah': Region,
            'Tanggal Relatif': pd.date_range(start=start_date, end=end_date).to_list()
        }
        data = pd.DataFrame(data)

        data['Kode Wilayah'] = data['Kode Wilayah'].replace({'Jabodetabek': 0, 'Non Jabodetabek (Jawa)': 2, 'Jawa (Jabodetabek+Non Jabodetabek)': 1, 'Sumatera': 3})

        # Convert Tanggal column to datetime and calculate the difference from the reference date
        reference_date = pd.to_datetime('2006-01-01')
        data['Tanggal Relatif'] = pd.to_datetime(data['Tanggal Relatif']).sub(reference_date).dt.days

        # Load the model from the pickle file
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Make prediction using the loaded model
        predictions = model.predict(data)

        # Create a DataFrame to store the results
        results = pd.DataFrame({'Date': data['Tanggal Relatif'], 'Predicted Passenger': predictions})

        # Visualize the results using matplotlib
        plt.plot(results['Date'], results['Predicted Passenger'])
        plt.xlabel('Date')
        plt.ylabel('Predicted Passenger')
        plt.title('Predicted Amount of Commuter Passenger over Time')
        st.pyplot(plt)

        # Optionally, you can also show the raw data in a table
        st.dataframe(results)

if __name__ == '__main__':
    main()
