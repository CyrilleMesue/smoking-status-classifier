# Impprt Utilities
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import LinearSVC, SVC
from scipy.sparse import csr_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier, StackingClassifier, RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.svm import LinearSVR, SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV


def get_model(model_type):
    """
    Method returns the model objective respective to the given model name
    """
    if model_type == "SVC":
        clf = SVC(probability=True, random_state = 42)
        return clf
    elif model_type == "SGDClassifier":
        clf = SGDClassifier(random_state = 42)
        return clf
    elif model_type == "LogisticRegression":
        clf= LogisticRegression(random_state = 42)
        return clf
    elif model_type == "XGBClassifier":
        clf= XGBClassifier(random_state = 42)
        return clf
    elif model_type == "MLPClassifier":
        clf= MLPClassifier(random_state = 42)
        return clf
    elif model_type == "KNeighborsClassifier":
        clf= KNeighborsClassifier()
        return clf
    elif model_type == "GradientBoostingClassifier":
        clf=GradientBoostingClassifier(random_state = 42)
        return clf
    elif model_type == "RandomForestClassifier":
        clf=RandomForestClassifier(random_state=42)
        return clf
    elif model_type == "AdaBoostClassifier":
        clf = AdaBoostClassifier(random_state = 42)
        return clf
    elif model_type == "VotingClassifier":
        clf1 = MLPClassifier(random_state = 42)
        clf2 = SVC(random_state = 42)
        clf3 = XGBClassifier(random_state = 42)
        clf = VotingClassifier(estimators=[('clf1', clf1), ('clf2', clf2), ('clf3', clf3)], voting='hard')
        return clf

