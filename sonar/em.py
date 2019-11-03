# see GaussianMixture

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
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

from sklearn import mixture, metrics


from sonar.load_data import load_sonar
from sklearn.model_selection import train_test_split

X, y = load_sonar()

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

RAND = 23

gm = GaussianMixture(n_components=2, verbose=100, random_state=RAND)
gm.fit(X_train, y_train)


# EM PCA
# EM ICA
# EM RP
# EM Other

def plot_silhouette(km, training_data,  title='Silhoutte for EM'):
    y_km = km.fit_predict(training_data)

    cluster_labels = np.unique(y_km)
    n_clusters = cluster_labels.shape[0]
    silhouette_vals = silhouette_samples(training_data, y_km, metric='euclidean')
    y_ax_lower, y_ax_upper = 0, 0
    yticks = []
    for i, c in enumerate(cluster_labels):
        c_silhouette_vals = silhouette_vals[y_km == c]
        c_silhouette_vals.sort()
        y_ax_upper += len(c_silhouette_vals)
        color = cm.jet(float(i) / n_clusters)
        plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0,
                 edgecolor='none', color=color)

        yticks.append((y_ax_lower + y_ax_upper) / 2.)
        y_ax_lower += len(c_silhouette_vals)

    silhouette_avg = np.mean(silhouette_vals)
    print("Silohouette Avg", silhouette_avg, title)
    plt.axvline(silhouette_avg, color="red", linestyle="--")

    plt.yticks(yticks, cluster_labels + 1)
    plt.ylabel('Cluster')
    plt.xlabel('Silhouette coefficient')

    plt.tight_layout()
    plt.title(title)
    plt.show()

def pca_cum_variance(pca):
    plt.rcParams["figure.figsize"] = (12, 6)

    fig, ax = plt.subplots()
    xi = np.arange(1, 21, step=1)
    y = np.cumsum(pca.explained_variance_ratio_)[:20]

    plt.ylim(0.0, 1.1)
    plt.plot(xi, y, marker='o', linestyle='--', color='b')

    plt.xlabel('Number of Components')
    plt.xticks(np.arange(0, 31, step=1))  # change from 0-based array index to 1-based human-readable label
    plt.ylabel('Cumulative variance (%)')
    plt.title('The number of components needed to explain variance')

    plt.axhline(y=0.95, color='r', linestyle='-')
    plt.text(0.5, 0.85, '95% cut-off threshold', color='red', fontsize=16)

    ax.grid(axis='x')
    plt.show()



def pca(training_set, test_set):
    pca = PCA()

    # from KM analysis, #components = 16
    # use PCA projection training data
    pca = PCA(n_components=16)
    X_train = pca.fit_transform(training_set)
    X_test = pca.fit_transform(test_set)

    bics = []
    aics = []
    max_clusters = 5
    for i in range(1, max_clusters):
        gmm = mixture.GaussianMixture(i, covariance_type='full', random_state=RAND)
        gmm.fit(X_train)
        bics.append(gmm.bic(X_train))
        aics.append(gmm.aic(X_train))

    plt.plot(range(1, max_clusters), bics, marker='o')
    plt.plot(range(1, max_clusters), aics, marker='o')
    plt.xlabel('Number of clusters')
    plt.xlabel("No. of GMM components")
    plt.legend(loc="best")
    plt.ylabel('BIC')
    plt.title("bic, aic vs # components GMM, PCA w/16 comp")

    plt.tight_layout()
    plt.show()

    gmm = mixture.GaussianMixture(2, covariance_type='full', random_state=RAND)
    gmm.fit(X_train)
    plot_silhouette(gmm, X_train)

    gmm = mixture.GaussianMixture(3, covariance_type='full', random_state=RAND)
    gmm.fit(X_train)
    plot_silhouette(gmm, X_train, title="K=3")

    gmm = mixture.GaussianMixture(4, covariance_type='full', random_state=RAND)
    gmm.fit(X_train)
    plot_silhouette(gmm, X_train, title="K=4")

def ica(training_set, test_set):
    # 10 is the optimum number of EM components
    ica = FastICA(n_components=10, random_state=RAND, max_iter=1000)
    X_train = ica.fit_transform(training_set)

    gmm = mixture.GaussianMixture(2, covariance_type='full', random_state=RAND)
    gmm.fit(X_train)
    plot_silhouette(gmm, X_train, title="ICA(10), GMM(2)")

    gmm = mixture.GaussianMixture(3, covariance_type='full', random_state=RAND)
    gmm.fit(X_train)
    plot_silhouette(gmm, X_train, title="ICA(10), GMM(3)")

    gmm = mixture.GaussianMixture(4, covariance_type='full', random_state=RAND)
    gmm.fit(X_train)
    plot_silhouette(gmm, X_train, title="ICA(10), GMM(4)")

    gmm = mixture.GaussianMixture(5, covariance_type='full', random_state=RAND)
    gmm.fit(X_train)
    plot_silhouette(gmm, X_train, title="ICA(10), GMM(5)")

def rp(train, test, y_train,  y_test):
    model = LinearSVC()
    model.fit(train, y_train)
    baseline = metrics.accuracy_score(model.predict(X_test), y_test)

    accuracies = []
    components = np.int32(np.linspace(2, 60, 20))

    # loop over the projection sizes
    for comp in components:
        # create the random projection
        sp = SparseRandomProjection(n_components=comp)
        X = sp.fit_transform(train)

        # train a classifier on the sparse random projection
        model = LinearSVC()
        model.fit(X, y_train)

        # evaluate the model and update the list of accuracies
        test = sp.transform(X_test)
        accuracies.append(metrics.accuracy_score(model.predict(test), y_test))

    # create the figure
    plt.figure()
    plt.title("Accuracy of Sparse Rand Projection on Sonar (EM, GMM)")
    plt.xlabel("# of Components")
    plt.ylabel("Accuracy")
    plt.xlim([2, 64])
    plt.ylim([0, 1.0])

    # plot the baseline and random projection accuracies
    plt.plot(components, [baseline] * len(accuracies), color="r")
    plt.plot(components, accuracies)
    plt.show()

    #random pick 30 as the best number of Random components
    sp = SparseRandomProjection(n_components=30)
    X_train = sp.fit_transform(train)

    gmm = mixture.GaussianMixture(2, covariance_type='full', random_state=RAND)
    gmm.fit(X_train)
    plot_silhouette(gmm, X_train, title="RP(30), GMM(2)")

    gmm = mixture.GaussianMixture(3, covariance_type='full', random_state=RAND)
    gmm.fit(X_train)
    plot_silhouette(gmm, X_train, title="RP(30), GMM(3)")

    gmm = mixture.GaussianMixture(4, covariance_type='full', random_state=RAND)
    gmm.fit(X_train)
    plot_silhouette(gmm, X_train, title="RP(30), GMM(4)")

def tree(X_train, X_test):
    # 7 features equals the original accuracy

    clf = ExtraTreesClassifier(n_estimators=50, random_state=RAND)
    clf = clf.fit(X_train, y_train)
    print("Original Features Size:", X_train.shape[1])
    model = SelectFromModel(clf, prefit=True, max_features=7)
    X_new = model.transform(X)

    gmm = mixture.GaussianMixture(2, covariance_type='full', random_state=RAND)
    gmm.fit(X_new)
    plot_silhouette(gmm, X_train, title="Tree(7), GMM(2)")

    gmm = mixture.GaussianMixture(3, covariance_type='full', random_state=RAND)
    gmm.fit(X_new)
    plot_silhouette(gmm, X_train, title="Tree(7), GMM(3)")

    gmm = mixture.GaussianMixture(4, covariance_type='full', random_state=RAND)
    gmm.fit(X_new)
    plot_silhouette(gmm, X_train, title="Tree(7), GMM(4)")






pca(X_train, X_test)
ica(X_train, X_test)
rp(X_train, X_test,y_train, y_test)
tree(X_train, X_test)

