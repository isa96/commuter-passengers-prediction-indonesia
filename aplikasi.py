import pickle
import streamlit as st
import pandas as pd

title = 'Predict Amount of Commuter Passenger 🚉'
subtitle = 'Predict Amount of Commuter Passenger using machine learning 🚄🚄 '

def main():
    st.set_page_config(layout="centered", page_icon='🚉', page_title='Lets Predict Amount of Commuter Passenger!')
    st.title(title)
    st.write(subtitle)

    form = st.form("Data Input")
    Region = form.selectbox('Region', ['Jabodetabek', 'Non Jabodetabek (Jawa)', 'Jawa (Jabodetabek+Non Jabodetabek)', 'Sumatera'])
    Date = form.date_input('Date')

    submit = form.form_submit_button("Predict")  # Add a submit button

    if submit:
        data = {
            'Kode Wilayah': Region,
            'Tanggal Relatif': Date,
        }
        data = pd.Series(data).to_frame(name=0).T
        data['Kode Wilayah'] = data['Kode Wilayah'].replace({'Jabodetabek': 0, 'Non Jabodetabek (Jawa)': 2, 'Jawa (Jabodetabek+Non Jabodetabek)': 1, 'Sumatera': 3})

        # Convert Tanggal column to datetime and calculate the difference from the reference date
        reference_date = pd.to_datetime('2006-01-01')
        data['Tanggal Relatif'] = pd.to_datetime(data['Tanggal Relatif']).sub(reference_date).dt.days

        # Load the model from the pickle file
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Make prediction using the loaded model
        prediction = model.predict(data)[0]
        rounded_prediction = round(prediction)  # Round the prediction to the nearest integer
        st.success('Your predicted amount of commuter passenger: ' + str(rounded_prediction))

if __name__ == '__main__':
    main()
