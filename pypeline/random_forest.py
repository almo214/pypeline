"""Random Forest model"""

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, jaccard_score
import matplotlib as mpl
import matplotlib.pyplot as plt
from pypeline import utils as plu

#Define a function that conducts Random Forest classification and/or features selection
def models_rf(x,y,imft=10,rt_imp=False):
  """Function to conduct cv hypergrid of Random Forest model, can also conduct feature selection for other models.
  x=independent variables
  y=dependent variable(s)
  imft=Number of important features to report, default 10
  rt_imp=Report important features? For feature selection 
      prior to other using other ML methods
  """
  y_val = []
  x_val = []
  best_i=1
  b_p=[]
  best_y=0
  #x=preprocessing.normalize(x)
  for i in range(1,10):
    X_train, X_test, y_train, y_test = train_test_split(x,y, random_state=42,test_size=i/10, shuffle=True)

# Set up the parameter grid
    rf= RandomForestClassifier()
    parameters = dict(criterion=['gini', 'entropy'] ,
                      max_depth=range(2,20),
                      n_estimators = range(1,30))

# Create a GridSearchCV object
    grid_search = GridSearchCV(rf, parameters, cv=5, scoring='accuracy', n_jobs=-1)

# Fit the GridSearchCV object to the training data
    grid_search.fit(X_train, y_train)

# Get the best hyperparameters and score
    best_params = grid_search.best_params_
    b_p.append(best_params)
    best_score = grid_search.best_score_
    rf_best = RandomForestClassifier(**best_params)
    rf_best.fit(X_train, y_train)
    y_pred= rf_best.predict(X_test)
    x_val.append(i)
    y_val.append(accuracy_score(y_test,y_pred)*100)
    if i>1:
      if y_val[i-1]> best_y:
        best_y=y_val[i-1]
        best_i=i
        y_pred_bestp=rf_best.predict_proba(X_test)[:, 1]
        y_pred_best=rf_best.predict(X_test)
        y_act=y_test
        rf_imp=rf_best.feature_importances_
        sorted_idx = rf_imp.argsort()[::-1][:imft]
  if rt_imp==False:
    fig, ax= plt.subplots(2,2,figsize=(40,15))
    mpl.rcParams['font.size'] = 28
    ax[0,0].set_title('Learning Curve')
    plu.learn_curve(X_train, y_train, b_p[best_i-1],'rf',ax[0,0])
    ax[0,1].set_title('Test Accuracy vs. Train-Test Split')
    plu.accuracy_plot(x_val,y_val,ax[0,1])
    ax[1,0].set_title('Features Importance')
    ax[1,0].bar(x.columns[sorted_idx], rf_imp[sorted_idx], color="#400000")
    ax[1,0].set_xticks(range(len(x.columns[sorted_idx])), x.columns[sorted_idx], rotation=90, fontsize=20)
    ax[1,0].set_ylabel('Feature Importance', fontsize=22)
    ax[1,1].set_title("ROC Curve")
    plu.roc_plot(y_act,y_pred_bestp,ax[1,1])
    plt.show()

#Output Accuracy metrics
  Ac=max(y_val)
  F1=f1_score(y_act, y_pred_best, average='weighted')
  Jac=jaccard_score(y_act, y_pred_best, average='micro')
  print("\n\nMax Accuracy:",Ac)
  print("F1: ",F1 )
  print("Jaccard:",Jac )
  print("\n\nParams: ",b_p[best_i-1])
  if not rt_imp:
    return(Ac,F1,Jac)
  if rt_imp:
    return(x.columns[sorted_idx], rf_imp[sorted_idx])