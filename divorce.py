'''
    Aidan Melchior
    Data Mining Final Project

    processes data pertaining to predictors for divorced marriages

    source:
        http://archive.ics.uci.edu/ml/datasets/Divorce+Predictors+data+set
'''

import csv
import numpy as np
import math
import statistics
import scipy
import re
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#import sklearn
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from sklearn import decomposition
from sklearn import datasets
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
#from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler 
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF


'''~~~~~~~~~~~
    functions
   ~~~~~~~~~~~''' 

def get_data(data, filename):
    """opens data file and copies data to runtime
        data object"""
    with open(filename) as file:
        reader = csv.reader(file, delimiter=',', quotechar='|')
        for row in reader:
            data.append(row)
    return data


def format_row(data):
    """splits string into 1d array of float"""
    data_split = re.split(';',data)
    for index in range(0, len(data_split)):
        data_split[index] = float(data_split[index])
    return data_split


def format_row_to_string(data):
    """splits string into 1d array of string"""
    data_split = re.split(';',data)
    for index in range(0, len(data_split)):
        data_split[index] = data_split[index]
    return data_split


def format_data(data):
    """formats data from 1d array of string
        to 2d array of floats"""
    for index in range(0, len(data)):
        data[index] = format_row(data[index][0])
    return data


def get_target(data, index_of_class):
    '''returns array containing class attribute'''
    target = []
    for point in data:
        target.append(int(point[index_of_class]))
    return target


def shuffle(data):
    np.random.seed(0)
    np.random.shuffle(data)
    return data
    
    

def scale(data):
    scaler = MinMaxScaler(feature_range=(0,1))
    return scaler.fit_transform(data)


def simple_sample(data, interval):
    new_data = []
    for i in range(0, len(data)):
        if i % interval == 0:
            new_data.append(data[i])
    return new_data


def split(data, target, train_ratio):
    index = int(len(data) * train_ratio)
    return data[0:index], data[index:], target[0:index], target[index:]
    

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


def kmeans(data, target):

    np.random.seed(5)

    iris = datasets.load_iris()
    X = np.array(data)#iris.data
    y = target#iris.target

    estimators = [('k_means_divorce_8', KMeans(n_clusters=8)),
              ('k_means_divorce_3', KMeans(n_clusters=3)),
              ('k_means_fivorce_bad_init', KMeans(n_clusters=3, n_init=1,
                                               init='random'))]

    fignum = 1  
    titles = ['8 clusters', '3 clusters', '3 clusters, bad initialization']
    for name, est in estimators:
        fig = plt.figure(fignum, figsize=(4, 3))
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
        est.fit(X)
        labels = est.labels_

        ax.scatter(X[:, 3], X[:, 0], X[:, 2],
               c=labels.astype(np.float), edgecolor='k')

        ax.w_xaxis.set_ticklabels([])
        ax.w_yaxis.set_ticklabels([])
        ax.w_zaxis.set_ticklabels([])
        ax.set_xlabel('Petal width')
        ax.set_ylabel('Sepal length')
        ax.set_zlabel('Petal length')
        ax.set_title(titles[fignum - 1])
        ax.dist = 12
        fignum = fignum + 1

    # Plot the ground truth
    fig = plt.figure(fignum, figsize=(4, 3))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    '''
    for name, label in [('Setosa', 0),
                    ('Versicolour', 1),
                    ('Virginica', 2)]:
        ax.text3D(X[y == label, 3].mean(),
              X[y == label, 0].mean(),
              X[y == label, 2].mean() + 2, name,
              horizontalalignment='center',
              bbox=dict(alpha=.2, edgecolor='w', facecolor='w'))'''
    # Reorder the labels to have colors matching the cluster results
    y = np.choose(y, [1, 0]).astype(np.float)
    ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=y, edgecolor='k')

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('Petal width')
    ax.set_ylabel('Sepal length')
    ax.set_zlabel('Petal length')
    ax.set_title('Ground Truth')
    ax.dist = 12

    fig.show()

    
def classifier0(data, target, target_names):
    iris = datasets.load_iris()
    X = np.array(data)
    y = target 
    n_features = X.shape[1]
    C = 10
    kernel = 1.0 * RBF([1.0, 1.0])  # for GPC
    # Create different classifiers.
    classifiers = {
        'L1 logistic': LogisticRegression(C=C, penalty='l1',
                                      solver='saga',
                                      multi_class='multinomial',
                                      max_iter=10000),
        'L2 logistic (Multinomial)': LogisticRegression(C=C, penalty='l2',
                                                    solver='lbfgs',
                                                    multi_class='multinomial',
                                                    max_iter=10000),
        'L2 logistic (OvR)': LogisticRegression(C=C, penalty='l2',
                                            solver='saga',
                                            multi_class='ovr',
                                            max_iter=10000),
        'Linear SVC': SVC(kernel='linear', C=C, probability=True,
                      random_state=0)#,
        #'GPC': GaussianProcessClassifier(kernel)
    }

    n_classifiers = len(classifiers)

    plt.figure(figsize=(3 * 2, n_classifiers * 2))
    plt.subplots_adjust(bottom=.2, top=.95)

    xx = np.linspace(0, 4, 100)
    print("X",X)
    print("XX",xx)
    yy = np.linspace(0, 1, 100).T
    print("yy",yy)
    xx, yy = np.meshgrid(xx, yy)
    Xfull = np.c_[xx.ravel(), yy.ravel()]
    print("xfull",Xfull)
    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(X, y)

        y_pred = classifier.predict(X)
        accuracy = accuracy_score(y, y_pred)
        print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))

        # View probabilities:
        probas = classifier.predict_proba(X)
        n_classes = np.unique(y_pred).size
        for k in range(n_classes):
            plt.subplot(n_classifiers, n_classes, index * n_classes + k + 1)
            plt.title(target_names[k])
            if k == 0:
                plt.ylabel(name)
            #print("k=",k)
           # print(probas)
            print(probas.shape)
            imshow_handle = plt.imshow(probas,#[:, k].reshape((100, 100)),
                                   extent=(0, 1, 0, 1), origin='lower')
            plt.xticks(())
            plt.yticks(())
            idx = (y_pred == k)
            if idx.any():
                plt.scatter(X[idx, 0], X[idx, 1], marker='o', c='w', edgecolor='k')

    ax = plt.axes([0.15, 0.04, 0.7, 0.05])
    plt.title("Probability")
    plt.colorbar(imshow_handle, cax=ax, orientation='horizontal')

    plt.show()






'''-------------
    script
   -------------'''

print("started")
data_raw = []
model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
get_data(data_raw, "divorce.csv")
labels = format_row_to_string(data_raw[0][0])
target_names = ['married','divorced']
data_raw = data_raw[1:]
data = format_data(data_raw)
target = get_target(data, 54)

data_array= np.array(data) 

#data_panda= pd.array(data)
#print(data_panda.head())

print("------PREPROCESSING------")

#print(len(data))
data = shuffle(scale(data))  #normalize and shuffle data
data_train, data_test, target_train, target_test = split(data, target, .80)
print("data randomized")
print("len data\t",len(data))
print("len train\t",len(data_train))
print("len test\t",len(data_test))
sample = simple_sample(data,10)
sample_target = get_target(sample,54)
print("sample created\n\tsize\t", len(sample))

#print(sample_target)

print("~~~~~~DENDOGRAM~~~~~~")
model = model.fit(data)
#plt.title('Hierarchical Clustering Dendrogram')
#plot_dendrogram(model, truncate_mode='level', p=3)
#plt.xlabel("Number of points in node (or index of point if no parenthesis).")
#plt.show()

print("~~~~~~K-MEANS~~~~~~")
#kmeans(data[:53], target)
#kmeans(sample[:53], sample_target)


print("~~~~~~PROBABILITY CLASSIFIER~~~~~~")
#classifier0(sample[:53], sample_target, target_names)
#classifier0(data, target, target_names)

#print_data(data[:53])

#print_data(data_raw)
print("finished")

