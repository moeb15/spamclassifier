import helpers
import pandas as pd
import joblib
import sys
import os
import uuid

model, ref_col, target, vectorizer, tfid = joblib.load('models/model.pkl')
filepath = sys.argv[1]

if not os.path.exists('figures'):
    os.makedirs('figures')

def make_predictions(filename=None):
    """
    Classifies emails as spam or ham
    Parameters
    filename: String, file path of csv file containing emails, must contain EmailText Column,
    if Label included, classification report will also be created
    """
    emails = pd.read_csv(filename)
    X = emails[ref_col]
    
    extracted_features = vectorizer.transform(X)
    pred = model.predict(extracted_features)

    if target in list(emails.columns):
        y = emails[target]
        helpers.get_report(model, X, y, vectorizer)
        helpers.generate_cnf_mtx(model, X, y, vectorizer,f'{uuid.uuid1()}')
    
    return pred
    
if __name__ == '__main__':
    pred = make_predictions(filepath)
    print(pred)