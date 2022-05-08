import numpy as np

from sklearn.model_selection import cross_validate, train_test_split, KFold
from sklearn.metrics import r2_score, log_loss, mean_squared_error,\
    confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

class ModelMetrics:
    """
    --------------------------------------------------------------------------
    This module reads your sklearn model and dataset(has to be preprocessed, 
    scaled before pass) and display metrics.
    --------------------------------------------------------------------------
    For regression models display:
    -R2 score
    -RMSE
    --------------------------------------------------------------------------
    For classification models display:
    - log loss
    - confusion matrix
    - classification report
    """
    def __init__(self, model, type: str, X_train,
                 y_train, X_test, y_test) -> None:
        self._model = model
        self._type = type
        self._X_train, self._y_train, self._X_test, self._y_test =\
            X_train, y_train, X_test, y_test
    
    def model_performance(self):
        kf = KFold(shuffle=True)
        self._model.fit(self._X_train, self._y_train)
        if self._type == 'regression':
            cross_score = cross_validate(self._model, self._X_train,
             self._y_train,cv=kf, scoring='neg_log_loss')
            print(f"""Cross validation mean score is\
{-1*np.mean(cross_score['test_score']):.2f} with standard deviation\
{np.std(cross_score):.2f}""")
            predicted = self._model.predict(self._X_test)
            rmse = np.sqrt(mean_squared_error(self._y_test, predicted))
            print(f'Root Mean Squared Error for test set is {rmse:.2f}')
            r2 = r2_score(self._y_test, predicted)
            print(f'R2 score for test set is {r2:.2f}')
        else:
            cross_score = cross_validate(self._model, self._X_train,
             self._y_train,cv=kf, scoring='neg_mean_squared_error')
            mean_sc = -1*np.mean(cross_score['test_score'])
            print(f'Cross validation mean score is {mean_sc}')
            predicted = self._model.predict(self._X_test)
            log_pred = self._model.predict_proba(self._X_test)
            log = log_loss(self._y_test, log_pred)
            print(f'Log Loss for test set is {log:.4f}')
            print(f'Confusion matrix for test set')
            print(confusion_matrix(self._y_test, predicted))
            print(classification_report(self._y_test, predicted))
    
if __name__ == '__main__':
    from sklearn.datasets import load_iris
    from xgboost.sklearn import XGBClassifier
    from sklearn.pipeline import make_pipeline
    iris = load_iris()
    X=iris['data']
    y=iris['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y,
     test_size=0.3, random_state=42)
    print(X_test.shape, y_test.shape, X_train.shape, y_train.shape)
    model = make_pipeline(StandardScaler(), XGBClassifier())
    metrics = ModelMetrics(model, 'classification', X_train, y_train, X_test, y_test)
    metrics.model_performance()
