"""Pruned Classification Tree"""
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, jaccard_score

from pypeline import utils as plu

#Define a function that conducts a classification tree and pruning
def models_ctreep(x,y,imft=10):
  """ Function to conduct cv hypergrid of Classification tree & prune model
  x=independent variables
  y=dependent variable(s)
  imft=Number of important features to report, default 10
      used for features importance plot
  """
  y_val = []
  x_val = []
  best_i=1
  b_p=[]
  best_y=0
  le = LabelEncoder()
  y = le.fit_transform(y)
  for i in range(1,10):
    X_train, X_test, y_train, y_test = train_test_split(x,y, random_state=42,test_size=i/10, shuffle=True)

# Set up the parameter grid
    ctree = DecisionTreeClassifier()
    parameters = dict(criterion=['gini','entropy'],
                      max_depth=range(1,20),
                      min_samples_split=range(2,10))

# Create a GridSearchCV object
    grid_search = GridSearchCV(ctree, parameters, cv=5, scoring='accuracy', n_jobs=-1)

# Fit the GridSearchCV object to the training data
    grid_search.fit(X_train, y_train)

# Get the best hyperparameters and score
    best_params = grid_search.best_params_
    b_p.append(best_params)
    best_score = grid_search.best_score_
    ctree_best = DecisionTreeClassifier(**best_params)
    ctree_best.fit(X_train, y_train)
    y_pred= ctree_best.predict(X_test)
    x_val.append(i)
    y_val.append(accuracy_score(y_test,y_pred)*100)
    if i!=1:
      if y_val[i-1]> best_y:
        best_y=y_val[i-1]
        best_i=i
        y_pred_bestp=ctree_best.predict_proba(X_test)[:, 1]
        y_pred_best=ctree_best.predict(X_test)
        y_act=y_test
        ctree_imp=ctree_best.feature_importances_
        sorted_idx = ctree_imp.argsort()[::-1][:imft]

  fig, ax= plt.subplots(2,2,figsize=(40,15))
  mpl.rcParams['font.size'] = 28
  ax[0,0].set_title('Learning Curve')
  plu.learn_curve(X_train, y_train, b_p[best_i-1],'ctree',ax[0,0])
  ax[0,1].set_title('Test Accuracy vs. Train-Test Split')
  plu.accuracy_plot(x_val,y_val,ax[0,1])
  ax[1,0].set_title('Features Importance')
  ax[1,0].bar(x.columns[sorted_idx], ctree_imp[sorted_idx], color="#400000")
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
  return(Ac,F1,Jac)