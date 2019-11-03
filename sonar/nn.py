from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

from sonar.load_data import load_sonar
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_samples, accuracy_score
from matplotlib import cm
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
import pandas as pd
from sklearn.random_projection import johnson_lindenstrauss_min_dim, SparseRandomProjection
from yellowbrick.cluster import InterclusterDistance
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import learning_curve
from sklearn.mixture import GaussianMixture

# https://www.analyticsvidhya.com/blog/2018/08/dimensionality-reduction-techniques-python/

X, y = load_sonar()

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, stratify=y)

RAND = 23

def plot_confusion_matrix(y_test,y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_test,y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(2.5, 3.5))
    ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x=j, y=i,s = cm[i, j], va = 'center', ha = 'center')

    plt.ylabel('true mine')
    plt.xlabel('\nmine\n\naccuracy={:0.4f}\n precision={:0.4f}'.format(accuracy, precision))

    plt.text(0.5, 1.25, title,
             horizontalalignment='center',
             fontsize=12,
             transform=ax.transAxes)


    plt.show()


def run_analysis(X,y, classifier, title):
    train_sizes, train_scores, test_scores = learning_curve( classifier,
                                                        X,
                                                        y,
                                                        # Number of folds in cross-validation
                                                        cv=10,
                                                        # Evaluation metric
                                                        scoring='accuracy',
                                                        # Use all computer cores
                                                        n_jobs=-1,
                                                        # 10 different sizes of the training set
                                                        train_sizes=np.linspace(0.10,1.0, 20),
                                                        shuffle=True,
                                                        random_state=23)

    #create means and standard deviations of training set scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    # Create means and standard deviations of test set scores
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Draw lines
    plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")
    plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")

    # Draw bands
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")

    # Create plot
    plt.title(title)
    plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
    plt.tight_layout()

    plt.show()


def pca(train, test, y_train, y_test):
    pca = PCA(n_components=16)
    X_train = pca.fit_transform(train)
    X_test = pca.transform(test)

    # run NN
    # 70
    # hidden
    # units in a
    # single
    # layer, learning
    # rate = 0.15, regularization(L2) = 0.4
    # had
    # an
    # accuracy
    # of
    # 0.78
    clf = MLPClassifier(solver='sgd', hidden_layer_sizes=(70,), random_state=23, shuffle=True, activation='relu',
                        learning_rate_init=0.15, alpha=0.45)
    run_analysis(X_train, y_train, clf, "NN with lrate=0.15, 70 units in hidden layer, alpha 0.45, PCA(16)")

    clf = MLPClassifier(solver='sgd', hidden_layer_sizes=(70,), random_state=23, shuffle=True, activation='relu',
                        learning_rate_init=0.15, alpha=0.45)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    plot_confusion_matrix(y_test, y_pred, title="NN (70,) lrate=0.15, alpha 0.45")

def ica(train, test, y_train, y_test):
    ica = FastICA(n_components=10, random_state=RAND, max_iter=1000)
    X_train = ica.fit_transform(train)
    X_test = ica.transform(test)

    clf = MLPClassifier(solver='sgd', hidden_layer_sizes=(70,), random_state=23, shuffle=True, activation='relu',
                        learning_rate_init=0.15, alpha=0.45)
    run_analysis(X_train, y_train, clf, "NN with lrate=0.15, 70 units in hidden layer, alpha 0.45, ICA(10)")

    clf = MLPClassifier(solver='sgd', hidden_layer_sizes=(70,), random_state=23, shuffle=True, activation='relu',
                        learning_rate_init=0.15, alpha=0.45)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    plot_confusion_matrix(y_test, y_pred, title="NN (70,) lrate=0.15, ICA(10)")

def rp(train, test, y_train, y_test):
    sp = SparseRandomProjection(n_components=12)
    X_train = sp.fit_transform(train)
    X_test = sp.transform(test)

    clf = MLPClassifier(solver='sgd', hidden_layer_sizes=(70,), random_state=23, shuffle=True, activation='relu',
                        learning_rate_init=0.15, alpha=0.45)
    run_analysis(X_train, y_train, clf, "NN with lrate=0.15, 70 units in hidden layer, alpha 0.45, RP(12)")

    clf = MLPClassifier(solver='sgd', hidden_layer_sizes=(70,), random_state=23, shuffle=True, activation='relu',
                        learning_rate_init=0.15, alpha=0.45)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    plot_confusion_matrix(y_test, y_pred, title="NN (70,) lrate=0.15, RP(12)")

def tree(train, test, y_train, y_test):
    clf = ExtraTreesClassifier(n_estimators=50)
    clf = clf.fit(train, y_train)
    print("Original Features Size:", train.shape[1])
    model = SelectFromModel(clf, prefit=True, max_features=3)
    X_train = model.transform(train)
    X_test = model.transform(test)

    clf = MLPClassifier(solver='sgd', hidden_layer_sizes=(70,), random_state=23, shuffle=True, activation='relu',
                        learning_rate_init=0.15, alpha=0.45)
    run_analysis(X_train, y_train, clf, "NN with lrate=0.15, 70 units in hidden layer, alpha 0.45, Tree(3)")

    clf = MLPClassifier(solver='sgd', hidden_layer_sizes=(70,), random_state=23, shuffle=True, activation='relu',
                        learning_rate_init=0.15, alpha=0.45)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    plot_confusion_matrix(y_test, y_pred, title="NN (70,) lrate=0.15, Tree(3)")

def km(train, test, y_train, y_test):
    km = KMeans(n_clusters=2,
                init='k-means++',
                n_init=10,
                max_iter=300,
                random_state=RAND)
    cluster_labels = km.fit_predict(train).reshape(-1, 1)
    cluster_lables_test = km.predict(test).reshape(-1, 1)

    clf = MLPClassifier(solver='sgd', hidden_layer_sizes=(70,), random_state=23, shuffle=True, activation='relu',
                        learning_rate_init=0.15, alpha=0.45)
    clf.fit(cluster_labels, y_train)
    y_pred = clf.predict(cluster_lables_test)
    print(confusion_matrix(y_test, y_pred))
    plot_confusion_matrix(y_test, y_pred, title="NN (70,) KM cluster as feature")


    # so y_km is the predicted cluster

def em(train, test, y_train, y_test):
    gmm = GaussianMixture(2, covariance_type='full', random_state=RAND)
    cluster_labels = gmm.fit_predict(train).reshape(-1, 1)
    cluster_lables_test = gmm.predict(test).reshape(-1, 1)

    clf = MLPClassifier(solver='sgd', hidden_layer_sizes=(70,), random_state=23, shuffle=True, activation='relu',
                        learning_rate_init=0.15, alpha=0.45)
    clf.fit(cluster_labels, y_train)
    y_pred = clf.predict(cluster_lables_test)
    print(confusion_matrix(y_test, y_pred))
    plot_confusion_matrix(y_test, y_pred, title="NN (70,) EM cluster as feature")

pca(X_train, X_test, y_train, y_test)
ica(X_train, X_test, y_train, y_test)
rp(X_train, X_test, y_train, y_test)
tree(X_train, X_test, y_train, y_test)
km(X_train, X_test, y_train, y_test)
em(X_train, X_test, y_train, y_test)



