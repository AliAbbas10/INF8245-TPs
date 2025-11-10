import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from q1 import data_preprocessing
from q2 import data_splits, normalize_features

# Step 1: Create hyperparameter grids for each model
# TODO fill out below dictionaries with reasonable values

# Picked the values that do not freeze the code and completes in reasonable time
param_grid_decision_tree = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_leaf': [1, 2, 5, 10],
    'max_leaf_nodes': [None, 10, 20, 50, 100]
}

param_grid_random_forest = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'bootstrap': [True, False]
}

param_grid_svm = {
    'kernel': ['linear', 'rbf'],
    'C': [0.1, 1, 10],
    'tol': [1e-3, 1e-4],
    'gamma': ['scale', 'auto']
}

# would like to try the below configs but the random forest search seems to freeze/take too long, same for svm
# param_grid_decision_tree = {
#     'criterion': ['gini', 'entropy'],
#     'max_depth': [None, 5, 10, 15, 20],
#     'min_samples_leaf': [1,2, 5, 10],
#     'max_leaf_nodes': [None, 10, 20, 50, 100, 200],
# }

# param_grid_random_forest = {
#     'n_estimators': [50, 100, 200, 500],
#     'max_depth': [5, 10, 15, None],
#     'bootstrap': [True, False]
# }

# param_grid_svm = {
#     'kernel': ['linear', 'rbf'],
#     'C': [0.01, 0.1, 1, 10, 100],
#     'tol': [1e-3, 1e-4],
#     'gamma': ['scale', 'auto', 1e-3, 1e-2, 1e-1],
# }

# Step 2: Initialize classifiers with random_state=0
decision_tree = DecisionTreeClassifier(random_state=0)
random_forest = RandomForestClassifier(random_state=0)
svm = SVC(random_state=0)

# Step 3: Assign scorer to 'accuracy'
scorer = 'accuracy'

# changed to 10-fold as in report
# Step 4: Perform grid search for each model using 10-fold StratifiedKFold cross-validation
def perform_grid_search(model, X_train, y_train, params):
    # Define the cross-validation strategy
    strat_kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

    # Grid search for the model
    # Using n_jobs=1 for autograder else it fails
    grid_search = GridSearchCV(estimator=model, param_grid=params, scoring=scorer, cv=strat_kfold, n_jobs=-1, verbose=2)
    
    # Handle both DataFrame/Series and ndarray inputs
    if hasattr(y_train, 'values'):
        y_train_flat = y_train.values.ravel()
    else:
        y_train_flat = np.ravel(y_train)
    
    grid_search.fit(X_train, y_train_flat)

    best_param = grid_search.best_params_
    best_score = grid_search.best_score_
    print("Best parameters are:", best_param)
    print("Best score is:", best_score)

    # Return the fitted grid search objects
    return grid_search, best_param, best_score



if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import accuracy_score
    
    X, y = data_preprocessing()
    X_train, X_test, y_train, y_test = data_splits(X, y)
    X_train_scaled, X_test_scaled = normalize_features(X_train, X_test)

    # Do Grid search for Decision Tree
    print("Grid Search for Decision Tree started")
    grid_decision_tree, best_params_decision_tree, best_score_decision_tree = perform_grid_search(decision_tree, X_train_scaled, y_train, param_grid_decision_tree)

    # Do Grid search for Random Forest
    print("Grid Search for Random Forest started")
    grid_random_forest, best_params_random_forest, best_score_random_forest = perform_grid_search(random_forest, X_train_scaled, y_train, param_grid_random_forest)

    # Do Grid search for SVM
    print("Grid Search for SVM started")
    grid_svm, best_params_svm, best_score_svm = perform_grid_search(svm, X_train_scaled, y_train, param_grid_svm)
    

    
    y_train_flat = y_train.values.ravel() if hasattr(y_train, 'values') else np.ravel(y_train)
    y_test_flat = y_test.values.ravel() if hasattr(y_test, 'values') else np.ravel(y_test)

    max_depth_values = param_grid_decision_tree['max_depth']
    dt_accuracies = []
    for depth in max_depth_values:
        dt_temp = DecisionTreeClassifier(random_state=0, max_depth=depth)
        dt_temp.fit(X_train_scaled, y_train_flat)
        acc = accuracy_score(y_train_flat, dt_temp.predict(X_train_scaled))
        dt_accuracies.append(acc)
    
    plt.figure(figsize=(10, 5))
    x_labels = [str(v) if v is not None else 'None' for v in max_depth_values]
    plt.plot(range(len(max_depth_values)), dt_accuracies, marker='o', linewidth=2, markersize=8)
    plt.xticks(range(len(max_depth_values)), x_labels, rotation=45)
    plt.xlabel('max_depth', fontsize=10)
    plt.ylabel('Training Accuracy', fontsize=10)
    plt.title('Effect of max_depth on training accuracy (Decision Tree)')
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.savefig('q3_dt_max_depth.png')
    plt.close()
    
    n_estimators_values = param_grid_random_forest['n_estimators']
    rf_accuracies = []
    for n_est in n_estimators_values:
        rf_temp = RandomForestClassifier(random_state=0, n_estimators=n_est)
        rf_temp.fit(X_train_scaled, y_train_flat)
        acc = accuracy_score(y_train_flat, rf_temp.predict(X_train_scaled))
        rf_accuracies.append(acc)
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(n_estimators_values)), rf_accuracies, marker='o', linewidth=2, markersize=8)
    plt.xticks(range(len(n_estimators_values)), n_estimators_values, rotation=45)
    plt.xlabel('n_estimators', fontsize=10)
    plt.ylabel('Training Accuracy', fontsize=10)
    plt.title('Effect of n_estimators on training accuracy (Random Forest)')
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.savefig('q3_rf_n_estimators.png')
    plt.close()
    
    kernel_values = param_grid_svm['kernel']
    svm_accuracies = []
    for kern in kernel_values:
        svm_temp = SVC(random_state=0, kernel=kern, C=1)
        svm_temp.fit(X_train_scaled, y_train_flat)
        acc = accuracy_score(y_train_flat, svm_temp.predict(X_train_scaled))
        svm_accuracies.append(acc)
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(kernel_values)), svm_accuracies, marker='o', linewidth=2, markersize=8)
    plt.xticks(range(len(kernel_values)), kernel_values, rotation=45)
    plt.xlabel('kernel', fontsize=10)
    plt.ylabel('Training Accuracy', fontsize=10)
    plt.title('Effect of kernel on training accuracy (SVM)')
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.savefig('q3_svm_kernel.png')
    plt.close()
    
    
    best_dt = DecisionTreeClassifier(random_state=0, **best_params_decision_tree)
    best_dt.fit(X_train_scaled, y_train_flat)
    dt_test_acc = accuracy_score(y_test_flat, best_dt.predict(X_test_scaled))
    
    best_rf = RandomForestClassifier(random_state=0, **best_params_random_forest)
    best_rf.fit(X_train_scaled, y_train_flat)
    rf_test_acc = accuracy_score(y_test_flat, best_rf.predict(X_test_scaled))
    
    best_svm = SVC(random_state=0, **best_params_svm)
    best_svm.fit(X_train_scaled, y_train_flat)
    svm_test_acc = accuracy_score(y_test_flat, best_svm.predict(X_test_scaled))
    
    model_names = ['Decision Tree', 'Random Forest', 'SVM']
    test_accuracies = [dt_test_acc]
    test_accuracies = [dt_test_acc, rf_test_acc, svm_test_acc]
    
    plt.figure(figsize=(10, 5))
    bars = plt.bar(model_names, test_accuracies, color=["#0D2B7E", "#db4231", "#15944a"])
    for bar, acc in zip(bars, test_accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, f'{acc:.5f}', ha='center', va='bottom')
    plt.xlabel('Model', fontsize=10)
    plt.ylabel('Test Accuracy', fontsize=10)
    plt.title('Test Accuracy across all three models', fontsize=10)
    plt.ylim([0, 1.0])
    plt.grid(True, alpha=0.5, axis='y')
    plt.tight_layout()
    plt.savefig('q3_test_accuracy_comparison.png')
    plt.close()
    
    
    grid_results = pd.DataFrame({
        'Model': ['Decision Tree', 'Random Forest', 'SVM'],
        'Best_CV_Score': [best_score_decision_tree, best_score_random_forest, best_score_svm],
        'Best_Parameters': [str(best_params_decision_tree), str(best_params_random_forest), str(best_params_svm)],
        # 'Best_CV_Score': [best_score_decision_tree, best_score_random_forest, best_score_svm],
        # 'Best_Parameters': [str(best_params_decision_tree), str(best_params_random_forest), str(best_params_svm)]
    })
    grid_results.to_csv('q3_grid_search_results.csv', index=False)
    
    dt_hp_df = pd.DataFrame({
        'max_depth': [str(v) for v in max_depth_values],
        'Training_Accuracy': dt_accuracies
    })
    dt_hp_df.to_csv('q3_dt_hyperparameter_analysis.csv', index=False)
    
    rf_hp_df = pd.DataFrame({
        'n_estimators': n_estimators_values,
        'Training_Accuracy': rf_accuracies
    })
    rf_hp_df.to_csv('q3_rf_hyperparameter_analysis.csv', index=False)
    
    svm_hp_df = pd.DataFrame({
        'kernel': kernel_values,
        'Training_Accuracy': svm_accuracies
    })
    svm_hp_df.to_csv('q3_svm_hyperparameter_analysis.csv', index=False)
    
    test_results = pd.DataFrame({
        'Model': model_names,
        'Test_Accuracy': test_accuracies
    })
    test_results.to_csv('q3_test_accuracy_results.csv', index=False)
    
    best_idx = np.argmax(test_accuracies)
    best_model_name = model_names[best_idx]
    best_test_acc = test_accuracies[best_idx]
    
    summary = {
        'Best_Model': [best_model_name],
        'Best_Test_Accuracy': [best_test_acc],
        'DT_Test_Accuracy': [dt_test_acc],
        'RF_Test_Accuracy': [rf_test_acc],
        'SVM_Test_Accuracy': [svm_test_acc],
        'DT_Best_Params': [str(best_params_decision_tree)],
        'RF_Best_Params': [str(best_params_random_forest)],
        'SVM_Best_Params': [str(best_params_svm)]
    }
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv('q3_summary_results.csv', index=False)