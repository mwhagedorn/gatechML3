import numpy as np
import pandas as pd
from sklearn import preprocessing



def load_churn():
    data = pd.read_csv('customer_churn.csv', header=0)

    def convert_column(col):
        encoder = preprocessing.LabelEncoder()
        data[col] = encoder.fit_transform(data[col])

    data['Churn'] = np.where(data['Churn'] == 'No', 0, 1)  # Change the Class representation
    data['gender'] = np.where(data['gender'] == 'Male', 0, 1)
    data['Partner'] = np.where(data['Partner'] == 'No', 0, 1)
    data['PhoneService'] = np.where(data['PhoneService'] == 'No', 0, 1)
    lines_e = preprocessing.LabelEncoder()
    data['MultipleLines'] = lines_e.fit_transform(data['MultipleLines'])
    is_e = preprocessing.LabelEncoder()
    data['InternetService'] = is_e.fit_transform(data['InternetService'])
    os_e = preprocessing.LabelEncoder()
    data['OnlineSecurity'] = os_e.fit_transform(data['OnlineSecurity'])
    convert_column('OnlineBackup')
    convert_column('DeviceProtection')
    convert_column('TechSupport')
    convert_column('StreamingTV')
    convert_column('StreamingMovies')
    convert_column('Contract')
    convert_column('PaperlessBilling')
    convert_column('PaymentMethod')
    convert_column('Dependents')

    # shuffle the data rows
   # data = data.reindex(np.random.RandomState(seed=42).permutation(data.index))
    data = data.drop('customerID', axis=1)

    data = data.replace(" ", 0)

    data[["TotalCharges"]].apply(pd.to_numeric)

    X = data[data.columns[0:-1]]
    y = data['Churn']

    return X,y

load_churn()