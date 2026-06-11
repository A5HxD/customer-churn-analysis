import joblib       # Library for saving and loading machine learning models

from sklearn.pipeline import Pipeline   # Library that chains multiple data processing steps and an estimator into a single, unified object.

from sklearn.linear_model import LogisticRegression     # Import the LogisticRegression Model

from sklearn.metrics import (   # Import the required evaluation metrics
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from preprocess import (    # Import the functions from the preprocess file
    load_data,
    clean_data,
    split_data,
    train_test_data,
    build_preprocessor
)

from config import MODEL_PATH   # Import the model path from the config file

def train():    # Create a function to initiate model training pipeline

    df=load_data()  # Load Data

    df=clean_data(df)   # Clean data

    X,y=split_data(df)  # Split data into features and class

    X_train, X_test, y_train, y_test = train_test_data(X,y) # Split data into testing and training data

    preprocessor = build_preprocessor(  # Process the training data (normalize)
        X_train
    )

    model = Pipeline(   #   
        ""
    )