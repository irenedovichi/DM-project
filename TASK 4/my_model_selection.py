from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report, roc_curve


def grid_search_kfold_cv(X_train, y_train, X_test, y_test, model_class, param_grid: dict, metric='f1', cv=3, eval=True, random_state=42,verbose=3):
    """
    Grid search and evaluation of the model
    :param X_train: Training set
    :param y_train: Training labels
    :param X_test: Test set
    :param y_test: Test labels
    :param model: model to evaluate (must be a sklearn model, or a model with methods fit and predict)
    :param param_grid: dict, parameter grid
    :param metric: metric to evaluate (default f1)
    :param cv: number of folds (default 3)
    :param random_state: seed (default 42)
    :param eval: whether to display and return statistics or not.
    False is useful if you don't want to train a model on the full training set, and you want to perform a second grid search
    :return: best_params if evaluate_model ==False, else best_params, best_model and results on the text set (dictionary with fields
    accuracy, confusion_matrix and classification_report)
    """
    grid = GridSearchCV(param_grid=param_grid, scoring=metric, cv=cv, estimator=model_class(random_state=random_state), verbose=verbose)
    grid.fit(X_train, y_train)
    best_params = grid.best_params_

    if eval:
        best_model = model_class(**best_params)
        best_model.fit(X_train, y_train)
        result = evaluate_model(best_model, X_train, y_train, X_test, y_test)
        return best_params, best_model, result
    else:  # just output the parameters (useful if you want to do another finer grid search)
        return best_params


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Evaluate the model on the test set
    :return: dictionary with fields accuracy, confusion_matrix and classification_report
    """
    # train model on whole training set
    # output statistics
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print('Accuracy on training set: ', accuracy_score(y_train, y_pred_train))
    print('confusion_matrix on training set: \n', confusion_matrix(y_train, y_pred_train))
    print('Accuracy on validation set: ', accuracy)
    print('Confusion matrix: \n', conf_matrix)
    print('Classification report: \n', class_report)

    result = {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report
    }
    return result

def plot_roc_curve(y_test, y_pred):
    """
    Plots the ROC curve
    :param y_test: test labels
    :param y_pred: predicted labels
    """
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    plt.plot(fpr, tpr)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.show()

from itertools import product

def hold_out_gs( X_train, y_train, X_val, y_val, param_grid, model, cv=3, scoring='f1', verbose=1):
    """
    Performs grid search on X_train, y_train, and evaluates the best model on X_val, y_val
    """
    params = list[(product(*param_grid.values()))]
    print(params)
    best_score = 0
    best_params = None
    for param in params:
        score_fold=np.zeros(cv)
        for i in range(cv):
            param = dict(zip(param_grid.keys(), param))
            model.set_params(**param)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            score_fold[i] = f1_score(y_val, y_pred)
            if verbose:
                print(f'params: {param}, score: {score_fold[i]}')
        score = score_fold.mean()
        if score > best_score:
            best_score = score
            best_params = param