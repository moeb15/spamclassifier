import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report


def get_report(model, X, y, vectorizer):
    """
    Displays classification report for model
    Parameters
    X: numpy array,
    y: numpy array,
    Return Value
    None
    """
    extracted_features = vectorizer.transform(X)

    pred = model.predict(extracted_features)

    print(classification_report(y,pred))

