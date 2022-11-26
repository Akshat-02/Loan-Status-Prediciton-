import numpy as np
import pandas as pd
import pickle
import streamlit as st
import os

#Getting the absolute path forrespective file to avoid streamlit error
try:
    #st.info(os.getcwd())
    for i in os.listdir():
        if i == 'trained_model.sav':
            trained_model_path = os.path.abspath(i)
        
        if i == 'Loan_status_Dataset':
            loan_dataset_path = os.path.abspath(i)

except FileNotFoundError:
    st.warning("Some file(s) are/is missing, maybe you have deleted the trained_model.sav or Loan_status_Dataset file from the directory. Kindly Reclone the repository!")
    
    
#Loading the serialized model object using pickle load() method
prediction_model = pickle.load(open(trained_model_path, 'rb'))


#Getting the dataset
loan_df = pd.read_csv(loan_dataset_path)
loan_df.dropna(inplace= True)

#Getting columns from dataset
loan_df.drop(columns=['Loan_ID', 'Loan_Status'], inplace=True)

#Getting the column names
all_columns = loan_df.columns


#Creating dictionary which stores the value of each column which has 
#categorical data
dict = {}
for i in all_columns:
    if loan_df[i].dtype == np.object:   
        dict[i] = list(loan_df[i].unique())
    else:
        dict[i] = None
    
print(dict)



def make_pred(input_data):
    
    #Creating the array out the input_data. Note that the asarray() method 
    #doesn't make a copy to the passsed iterable and directly modifies it 
    #in camparison to the np.array() which makes a copy of that iterable.
    input_data_array = np.asarray(input_data)  
    
    #Reshaping the array to full row
    reshaped_input_data = input_data_array.reshape(1, -1)
    
    
    #Making prediction below
    #It returns an ndarray 
    prediction= prediction_model.predict(reshaped_input_data)
    
    if prediction[0] == 1:
        return 'There are high chances that your loan will be approved'
    
    else:
        return 'Sorry, your chances of getting loan is low'
    


#Creating streamlit web app

def main():
    
    st.title('Loan Status Prediction')      #Set webapp title
    
    input_data_list = []
    description_list = ['Gender', 'Marriage Status', 'Number of Dependents',
                        'Education Qualification', 'Self Employment Status',
                        'Applicant Income', 'Co-Applicant Income', 
                        'Loan Amount (Value strictly in Thousands)',  
                        'Term of Loan (Value in months)', 'Credit History',
                        'Property Area']
    
    #for loop with column name and its description in a pair in tuple
    for i,j in zip(all_columns, description_list):
        
        #if column is object dtype then ask use to choose from categorical 
        #data value we created as dict.
        if loan_df[i].dtype == np.object: 
            input_data_list.append(st.selectbox(label=i, options= dict[i]))   
        
        #Creating slider speacially for Credit_History column
        elif i == 'Credit_History':
                input_data_list.append(st.radio(label= j, options= [0, 1])) 
        
        #Text input for other columns
        else:
            input_data_list.append(st.text_input(label= j))  
            
    
    #Creating pandas series object from user input list.
    series_input_data = pd.Series(input_data_list)
    
    #Replacing values to numerical from categorical user input data to feed the
    #model for prediction as model is trained on numerical data.
    series_input_data.replace({'Male': 1, 'Female':0, 
                'Yes': 1, 'No': 0, 'Graduate': 1, 'Not Graduate': 0, 
                'Yes': 1, 'No': 0, 'Rural': 0, 'Semiurban': 1, 'Urban': 2,
                '0': 0, '1': 1, '2': 2, '3+': 4, '0.0': 0.0, '1.0': 1.0}, inplace= True )
    
    
    #Creatng a submit button
    if st.button(label= "Check Loan Approval Status"):
        #Calling make_pred() function to make prediciton
        loan_status= make_pred(series_input_data)
        
        st.info(loan_status)
    

if __name__ == '__main__':
    main()
    
    
    
    