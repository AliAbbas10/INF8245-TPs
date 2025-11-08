import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from q1 import data_preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix


def data_splits(X, y):
    """
    Split the 'features' and 'labels' data into training and testing sets.
    Input(s): X: features (pd.DataFrame), y: labels (pd.DataFrame)
    Output(s): X_train, X_test, y_train, y_test
    """
    # Use random_state = 0 in the train_test_split
    # TODO write data split here
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True)

    return X_train, X_test, y_train, y_test


def normalize_features(X_train, X_test):
    """
    Take the input data and normalize the features.
    Input: X_train: features for train,  X_test: features for test (pd.DataFrame)
    Output: X_train_scaled, X_test_scaled (pd.DataFrame) the same shape of X_train and X_test
    """
    # TODO write normalization here
    # Hint: Use MinMaxScaler, fit on training data, transform both train and test
    
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    
    return X_train_scaled, X_test_scaled


def train_model(model_name, X_train_scaled, y_train):
    '''
    inputs:
       - model_name: the name of learning algorithm to be trained
       - X_train: features training set
       - y_train: label training set
    output: cls: the trained model
    '''
    if model_name == 'Decision Tree':
        # TODO call classifier here
        cls = DecisionTreeClassifier(random_state=0)
    elif model_name == 'Random Forest':
        # TODO call classifier here
        cls = RandomForestClassifier(random_state=0)
    elif model_name == 'SVM':
        # TODO call classifier here
        cls = SVC(random_state=0)

    # TODO train the model
    # y_train_flat = y_train.values.ravel() if hasattr(y_train, 'values') else y_train
    cls.fit(X_train_scaled, y_train)

    return cls


def eval_model(trained_models, X_train, X_test, y_train, y_test):
    '''
    inputs:
       - trained_models: a dictionary of the trained models,
       - X_train: features training set
       - X_test: features test set
       - y_train: label training set
       - y_test: label test set
    outputs:
        - y_train_pred_dict: a dictionary of label predicted for train set of each model
        - y_test_pred_dict: a dictionary of label predicted for test set of each model
        - a dict of accuracy and f1_score of train and test sets for each model
    '''
    evaluation_results = {}
    y_train_pred_dict = {
        'Decision Tree': None,
        'Random Forest': None,
        'SVM': None}
    y_test_pred_dict = {
        'Decision Tree': None,
        'Random Forest': None,
        'SVM': None}

    # Loop through each trained model
    for model_name, model in tqdm(trained_models.items()):
        # Predictions for training and testing sets
        # TODO predict y
        y_train_pred = model.predict(X_train)
        # TODO predict y
        y_test_pred = model.predict(X_test)
        # Calculate accuracy
        # TODO find accuracy
        train_accuracy = accuracy_score(y_train, y_train_pred)
        # TODO find accuracy
        test_accuracy = accuracy_score(y_test, y_test_pred)
        # Calculate F1-score
        # TODO find f1_score
        train_f1 = f1_score(y_train, y_train_pred, average='weighted')
        # TODO find f1_score
        test_f1 = f1_score(y_test, y_test_pred, average='weighted')
        # Store predictions
        # TODO
        y_train_pred_dict[model_name] = y_train_pred
        # TODO
        y_test_pred_dict[model_name] = y_test_pred  
        # Store the evaluation metrics
        evaluation_results[model_name] = {
            'Train Accuracy': train_accuracy,
            'Test Accuracy': test_accuracy,
            'Train F1 Score': train_f1,
            'Test F1 Score': test_f1
        }
    # Return the evaluation results
    return y_train_pred_dict, y_test_pred_dict, evaluation_results


def report_model(y_train, y_test, y_train_pred_dict, y_test_pred_dict):
    '''
    inputs:
        - y_train: label training set
        - y_test: label test set
        - y_train_pred_dict: a dictionary of label predicted for train set of each model, len(y_train_pred_dict.keys)=3
        - y_test_pred_dict: a dictionary of label predicted for test set of each model, len(y_train_pred_dict.keys)=3
    '''

    # Loop through each trained model
    for model_name in y_train_pred_dict.keys():
        print(f"\nModel: {model_name}")

        # Predictions for training and testing sets
        # TODO complete it
        y_train_pred = y_train_pred_dict[model_name]
        # TODO complete it
        y_test_pred = y_test_pred_dict[model_name]
        # Print classification report for training set
        print("\nTraining Set Classification Report:")
        # TODO write Classification Report train
        print(classification_report(y_train, y_train_pred))

        # Print confusion matrix for training set
        print("Training Set Confusion Matrix:")
        # TODO write Confusion Matrix train
        print(confusion_matrix(y_train, y_train_pred))
        # Print classification report for testing set
        print("\nTesting Set Classification Report:")
        # TODO write Classification Report test
        print(classification_report(y_test, y_test_pred))
        # Print confusion matrix for testing set
        print("Testing Set Confusion Matrix:")
        # TODO write Confusion Matrix test
        print(confusion_matrix(y_test, y_test_pred))    

if __name__ == "__main__":
    # TODO call data preprocessing from q1
    X, y = data_preprocessing()
    X_train, X_test, y_train, y_test = data_splits(X, y)
    X_train_scaled, X_test_scaled = normalize_features(X_train, X_test)

    cls_decision_tree = train_model('Decision Tree', X_train_scaled, y_train)
    cls_randomforest = train_model('Random Forest', X_train_scaled, y_train)
    cls_svm = train_model('SVM', X_train_scaled, y_train)

    # Define a dictionary of model name and their trained model
    trained_models = {
            'Decision Tree': cls_decision_tree,
            'Random Forest': cls_randomforest,
            'SVM': cls_svm }

    # predict labels and calculate accuracy and F1score
    y_train_pred_dict, y_test_pred_dict, evaluation_results = eval_model(trained_models, X_train_scaled, X_test_scaled, y_train, y_test)

    print("results:")
    for model_name, metrics in evaluation_results.items():
        print(f"\nModel: {model_name}")
        print(f"Train Accuracy: {metrics['Train Accuracy']}")
        print(f"Test Accuracy:  {metrics['Test Accuracy']}")
        print(f"Train F1 Score: {metrics['Train F1 Score']}")
        print(f"Test F1 Score:  {metrics['Test F1 Score']}")

    # classification report and calculate confusion matrix
    report_model(y_train, y_test, y_train_pred_dict, y_test_pred_dict)

    import seaborn as sns
    import matplotlib.pyplot as plt

    for model_name in trained_models.keys():
        y_test_pred = y_test_pred_dict[model_name]
        conf_matrix = confusion_matrix(y_test, y_test_pred)
        
        fig, ax = plt.subplots(figsize=(5,5))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['≤50K', '>50K'], yticklabels=['≤50K', '>50K'], ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'Confusion Matrix: {model_name}')
        filename = f'q2_confusion_matrix_{model_name.replace(" ", "_")}.png'
        fig.savefig(filename)
        plt.close(fig)

    




