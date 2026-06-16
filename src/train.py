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

    # Create a pipeline
    model = Pipeline(       # Pipeline() offers same preprocessing during training and prediction which prevents deployment bugs

        steps=[     # Define the sequence of steps to be executed
            (
                "preprocessor", preprocessor    # Preprocessing step: Apply scaling and encoding
            ),
            (
                "classifier", LogisticRegression(max_iter=1000) # Specify the model for training: Logistic regression used for prediction
            )
        ]
    )

    model.fit(      # Train the pipeline
        X_train, y_train    # Feature variables and target variables used for training
    )

    predictions = model.predict(    # Generate class predictions on the test dataset
        X_test      # Feature vraibles for test dataset
    )

    probabilities = model.predict_proba(    # Generate probability scores for each class predictiom
        X_test
    )[:,1]      # Extract only the class


    # Display the evaluation metrics

    print(
        "Accuracy:", accuracy_score(    # Calculate the model accuracy
            y_test, predictions     # Proportions of correct predictions
        )
    )

    print(
        "Precision:", precision_score(  # Calculate the model precisiom
            y_test, predictions     # percentage of True prediction that were actually true
        )
    )

    print(
        "Recall:", recall_score(    # Calculate the model recall
            y_test, predictions     # percentage of actual churn customers correctly identified
        )
    )

    print(
        "F1:", f1_score(    # Calculate the F1 Score
            y_test, predictions     # Calculates harmonic mean of precision and recall
        )
    )

    print(
        "ROC-AUC:", roc_auc_score(      # Calculate the ROC-AUC score
            y_test, probabilities   # Measures the model's ability to distinguish between churn and non-churn classes
        )
    )

    joblib.dump(    # save the trained data
        model,MODEL_PATH    # Define the model object and the model path
    )

    print(
        "\nModel saved..."  # Confirm saving
    )

if __name__ == "__main__":  # Execute the training function only when this file is run directly
    train()     # Start the model training process