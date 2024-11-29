import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import learning_curve, ShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


#Define a function to plot the Learning Curve
def learn_curve(x,y,params,model,ax):

  """ Function to plot the learning curve for ML models
  x= independent variables
  y=response/ dependent variable(s)
  params= best parameters identified in cv grid search
  model= the model for which the user wants to plot a learning curve
  ax= Axes plot location information
  """

  train_sizes = np.linspace(0.1, 1.0, 5)
  if model=='svc':
    mod=svm.SVC(**params)
  elif model=='logr':
    mod=LogisticRegression()
  elif model=='rf':
    mod=RandomForestClassifier()
  elif model=='knn':
    mod=KNeighborsClassifier()
  elif model=='ctree':
    mod=DecisionTreeClassifier()
  else:
    raise ValueError("No Model Detected")

# Define the shuffle split cross-validator
  cv = ShuffleSplit(n_splits=10, random_state=0)
  train_sizes, train_scores, test_scores = learning_curve(
      mod, x, y, cv=cv, n_jobs=-1,scoring = 'accuracy',
      train_sizes=train_sizes)


# Compute the learning curve scores
  train_mean = np.mean(train_scores, axis=1)

  test_mean = np.mean(test_scores, axis=1)

#Plot Curves
  ax.plot(train_sizes, train_mean, '--', color="Black",  label="Training score")
  ax.plot(train_sizes, test_mean, color="#400000", label="Cross-validation score")
  
  ax.set_xlabel("Training Set Size", fontsize=22), ax.set_ylabel("Accuracy", fontsize=22), ax.legend(loc="best", fontsize=22)
  plt.tight_layout()
  

  #Define a function to plot variable importance for linear type kernels
def imp(x, clf,ax):
    """ Function to plot feature importance for applicable models
    x= independent variables
    clf= best performing model
    ax= Axes plot location information
    """
    coef = clf.coef_.flatten()
    feature_names = x.columns.values
    importance = np.abs(coef)
    sorted_importance = importance.argsort()[::-1]
    sorted_features = feature_names[sorted_importance][:10]
    sorted_importance_values = importance[sorted_importance][:10]


    ax.bar(range(len(sorted_features)), sorted_importance_values, width=.3, edgecolor='black',color="#400000")
    ax.set_xticks(range(len(sorted_features)), sorted_features, rotation=90, fontsize=20)
    ax.set_ylabel('Feature Importance', fontsize=22)

   #Define a function to plot Accuracy vs. Test size
def accuracy_plot(x_val,y_val,ax):
  """ Function to plot model accuracy vs. test size
  x_val= integers 1-10
  y_val= Accuracy for each test size
  ax= Axes plot location information
  """
  ax.plot(x_val, y_val, color="#400000")
  ax.set_ylabel("Estimated Test Accuracy", fontsize=22)
  ax.set_xlabel("Test Size (x/10)%", fontsize=22)
  ax.set_xticks(np.arange(min(x_val), max(x_val)+1, 1),x_val, fontsize=20)

#Define a function to plot the ROC curve
def roc_plot(y1,y2,ax):
  """
  Function to plot the ROC 
  y1=Actual y values
  y2=Predicted y values
  ax= Axes plot location information
  """
  threshold = np.linspace(0,1,50)
  fpr, tpr, thresholds = roc_curve(y1, y2)
  roc_auc = auc(fpr, tpr)

  ax.plot(fpr, tpr, color="#400000", lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
  ax.plot([0, 1], [0, 1],  lw=2, linestyle='--', color='black')
  ax.set_xlabel('False Positive Rate')
  ax.set_ylabel('True Positive Rate')
  ax.legend(loc="lower right")
