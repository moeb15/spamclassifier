import helpers
import pandas as pd
import joblib
import sys

model, ref_col, target, vectorizer, tfid = joblib.load('models/model.pkl')
filepath = sys.argv[1]

def make_predictions(filename=None):
    """
    Classifies emails as spam or ham
    Parameters
    filename: String, file path of csv file containing emails, must contain EmailText Column,
    if Label included, classification report will also be created
    """
    cl_report = None
    emails = pd.read_csv(filename)
    X = emails[ref_col]
    
    extracted_features = vectorizer.transform(X)
    pred = model.predict(extracted_features)

    if target in list(emails.columns):
        y = emails[target]
        cl_report = helpers.get_report(model, X, y, vectorizer)
    
    return pred, cl_report
    
if __name__ == '__main__':
    make_predictions(filepath)