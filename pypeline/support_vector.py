"""pypeline support vector classification and/or regression."""
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import preprocessing, svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,f1_score, jaccard_score, root_mean_squared_error
from sklearn.svm import SVR, SVC
from pypeline import utils as plu

# Create a function that conducts Support Vector Classification or Regression
def models_svm(x,y, type: str = "SVC"):
  """ Function to conduct cv hypergrid of support vector classification model
  x=independent variables
  y=dependent variable(s)
  type = SVC for classification or SVR for regression
  """
  y_val = []
  x_val = []
  best_i=1
  b_p=[]
  b_pk=[]

  best_y=0
  for i in range(1,10):
    X_train, X_test, y_train, y_test = train_test_split(x,y, random_state=42,test_size=i/10, shuffle=True)


    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_test)

# Set up the parameter grid
    param_grid = {'C': [0.1, 1, 10, 100, 1000], 
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['linear', 'rbf'],
    }
    if type in ['SVC','svc','classification','classify', 'class']:
        svm = svm.SVC(probability=True)
        score_method='accuracy'
    else:
        svm = SVR()
        score_method = 'neg_root_mean_squared_error'

# Create a GridSearchCV object
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring=score_method, n_jobs=-1)

# Fit the GridSearchCV object to the training data
    grid_search.fit(X_train_scaled, y_train)

# Get the best hyperparameters and score
    best_params = grid_search.best_params_
    b_p.append(best_params)
    best_score = grid_search.best_score_
    if type in ['SVC','svc','classification','classify', 'class']:
        svm_best = SVC(**best_params, probability=True)
    else:
        svm_best = SVR(**best_params)

    svm_best.fit(X_train_scaled, y_train)
    y_pred= svm_best.predict(X_test)

    x_val.append(i)
    if type in ['SVC','svc','classification','classify', 'class']:
        y_val.append(accuracy_score(y_test,y_pred)*100)
    else:
        y_val.append(root_mean_squared_error(y_test, y_pred))

    if i>1:
      if y_val[i-1]> best_y:
        best_y=y_val[i-1]
        best_i=i
        b_pk=best_params['kernel']
        if type in ['SVC','svc','classification','classify', 'class']:
            y_pred_bestp=svm_best.predict_proba(X_test)[:, 1]
        y_pred_best=svm_best.predict(X_test)
        y_act=y_test
    k=str(b_pk)
  if type in ['SVC','svc','classification','classify', 'class']:
    fig, ax= plt.subplots(2,2,figsize=(40,15))
  else:
    fig, ax= plt.subplots(1,2, figsize=(40,8))
  mpl.rcParams['font.size'] = 28

  if type in ['SVC','svc','classification','classify', 'class']:
    if k=='linear':
        ax[0,0].set_title('Learning Curve')
        plu.learn_curve(X_train, y_train, b_p[best_i-1], 'svm',ax[0,0])
        ax[0,1].set_title('Test Accuracy vs. Train-Test Split')
        plu.accuracy_plot(x_val,y_val,ax[0,1])
        ax[1,0].set_title('Features Importance')
        plu.feature_importance(x,svm_best,ax[1,0])
        ax[1,1].set_title("ROC Curve")
        plu.roc_plot(y_act,y_pred_bestp,ax[1,1])
    else:
        ax[0,0].set_title('Learning Curve')
        plu.learn_curve(X_train, y_train, b_p[best_i-1],'svm',ax[0,0])
        ax[0,1].set_title('Test Accuracy vs. Train-Test Split')
        plu.accuracy_plot(x_val,y_val,ax[0,1])
        ax[1,0].set_title('ROC Curve')
        plu.roc_plot(y_act,y_pred_bestp,ax[1,0])
        ax[1, 1].axis('off')

    plt.show()
    Ac=max(y_val)
    F1=f1_score(y_act, y_pred_best, average='weighted')
    Jac=jaccard_score(y_act, y_pred_best, average='micro')
    print("\n\nMax Accuracy:",Ac)
    print("F1: ",F1 )
    print("Jaccard:",Jac )
    metrics = [Ac, F1, Jac]

  else:
    ax[0].set_title('Learning Curve')
    plu.learn_curve(X_train, y_train, b_p[best_i-1],'svr',ax[0])
    ax[1].set_title('')
    rmse =  abs(min(y_val))
    metrics = [rmse]


  print("\n\nParams: ",b_p[best_i-1])
  return(metrics)