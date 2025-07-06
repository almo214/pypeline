"""K Nearest Neighbors"""
#Import all necessary libraries

import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, jaccard_score 
from pypeline import utils as plu
import warnings
warnings.filterwarnings("ignore")




#Define a function that conducts KNN
def models_knn(x,y):
  """ Function to conduct cv hypergrid of KNN model
  x=independent variables
  y=dependent variable(s)
  """
  y_val = []
  x_val = []
  best_i=1
  b_p=[]
  best_y=0
  s=StandardScaler()
  x=s.fit_transform(x)
  for i in range(1,10): 
    X_train, X_test, y_train, y_test = train_test_split(x,y, random_state=42,test_size=i/10, shuffle=True)
    
# Set up the parameter grid
    knn=KNeighborsClassifier()
    parameters = dict(n_neighbors=list(range(1, 31)),
                      weights=['uniform', 'distance'],
                      algorithm=['auto','ball_tree','kd_tree','brute'])

# Create a GridSearchCV object
    grid_search = GridSearchCV(knn, parameters, cv=5, scoring='accuracy', n_jobs=-1)

# Fit the GridSearchCV object to the training data
    grid_search.fit(X_train, y_train)

# Get the best hyperparameters and score
    best_params = grid_search.best_params_
    b_p.append(best_params)
    best_score = grid_search.best_score_
    knn_best = KNeighborsClassifier(**best_params)
    knn_best.fit(X_train, y_train)
    y_pred= knn_best.predict(X_test)
    x_val.append(i)
    y_val.append(accuracy_score(y_test,y_pred)*100)
    if i!=1:
      if y_val[i-1]> best_y:
        best_y=y_val[i-1]
        best_i=i
        y_pred_bestp=knn_best.predict_proba(X_test)[:, 1]
        y_pred_best=knn_best.predict(X_test)
        y_act=y_test

  fig, ax= plt.subplots(2,2,figsize=(40,15))
  mpl.rcParams['font.size'] = 28
  ax[0,0].set_title('Learning Curve')
  plu.learn_curve(X_train, y_train, b_p[best_i-1],'knn',ax[0,0])
  ax[0,1].set_title('Test Accuracy vs. Train-Test Split')
  plu.accuracy_plot(x_val,y_val,ax[0,1])
  ax[1,0].set_title("ROC Curve")
  plu.roc_plot(y_act,y_pred_bestp,ax[1,0])
  ax[1, 1].axis('off')

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


  