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

# https://www.analyticsvidhya.com/blog/2018/08/dimensionality-reduction-techniques-python/

X, y = load_sonar()

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, stratify=y)

RAND = 23

# KM PCA
# KM ICA
# KM RP
# KM Other

def ica_avg_kurtosis_curve(training_set):
    plt.rcParams["figure.figsize"] = (12, 6)

    upper_bound = 20
    fig, ax = plt.subplots()
    xi = np.arange(1, upper_bound, step=1)
    avg_k = []
    for x in xi:
        ica = FastICA(n_components=x, random_state=RAND, max_iter=5000)

        X_train = ica.fit_transform(training_set)

        df_train = pd.DataFrame(X_train)
        kt = df_train.kurtosis()
        avg_k.append(np.mean(kt))

    y = avg_k

    plt.ylim(0.0, 10.0)
    plt.plot(xi, y, marker='o', linestyle='--', color='b')

    plt.xlabel('Number of Components')
    plt.xticks(np.arange(0, upper_bound, step=1))  # change from 0-based array index to 1-based human-readable label
    plt.ylabel('Average Kurtosis')
    plt.title('The average kurtosis for component set of size n')


    ax.grid(axis='x')
    plt.show()


def pca_cum_variance(pca):
    plt.rcParams["figure.figsize"] = (12, 6)

    fig, ax = plt.subplots()
    xi = np.arange(1, 20, step=1)
    y = np.cumsum(pca.explained_variance_ratio_)[:19]

    plt.ylim(0.0, 1.1)
    plt.plot(xi, y, marker='o', linestyle='--', color='b')

    plt.xlabel('Number of Components')
    plt.xticks(np.arange(0, 20, step=1))  # change from 0-based array index to 1-based human-readable label
    plt.ylabel('Cumulative variance (%)')
    plt.title('The number of components needed to explain variance')

    plt.axhline(y=0.95, color='r', linestyle='-')
    plt.text(0.5, 0.85, '95% cut-off threshold', color='red', fontsize=16)

    ax.grid(axis='x')
    plt.show()

def plot_silhouette(km, training_data,  title='Silhoutte for KM'):
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

def plain_clustering():
    distortions = []
    for i in range(1, 11):
        km = KMeans(n_clusters=i,
                    init='k-means++',
                    n_init=10,
                    max_iter=300,
                    random_state=RAND)
        km.fit(X_train)
        distortions.append(km.inertia_)

    plt.plot(range(1, 11), distortions, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.tight_layout()
    plt.show()

    km = KMeans(n_clusters=3,
                init='k-means++',
                n_init=10,
                max_iter=300,
                random_state=RAND)
    y_km = km.fit_predict(X_train)

    visualizer = InterclusterDistance(km)

    visualizer.fit(X_train)  # Fit the data to the visualizer
    visualizer.show()  # Finalize and render the figure

    # cluster_labels = np.unique(y_km)
    # n_clusters = cluster_labels.shape[0]
    # silhouette_vals = silhouette_samples(X, y_km, metric='euclidean')
    # y_ax_lower, y_ax_upper = 0, 0
    # yticks = []
    # for i, c in enumerate(cluster_labels):
    #     c_silhouette_vals = silhouette_vals[y_km == c]
    #     c_silhouette_vals.sort()
    #     y_ax_upper += len(c_silhouette_vals)
    #     color = cm.jet(float(i) / n_clusters)
    #     plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0,
    #              edgecolor='none', color=color)
    #
    #     yticks.append((y_ax_lower + y_ax_upper) / 2.)
    #     y_ax_lower += len(c_silhouette_vals)
    #
    # silhouette_avg = np.mean(silhouette_vals)
    # plt.axvline(silhouette_avg, color="red", linestyle="--")
    #
    # plt.yticks(yticks, cluster_labels + 1)
    # plt.ylabel('Cluster')
    # plt.xlabel('Silhouette coefficient')
    #
    # plt.tight_layout()
    # # plt.savefig('images/11_04.png', dpi=300)
    # plt.show()


    km = KMeans(n_clusters=2,
                init='k-means++',
                n_init=10,
                max_iter=300,
                random_state=RAND)
    y_km = km.fit_predict(X_train)
    visualizer = InterclusterDistance(km)
    visualizer.fit(X_train)  # Fit the data to the visualizer
    visualizer.show()  # F

    cluster_labels = np.unique(y_km)
    n_clusters = cluster_labels.shape[0]
    silhouette_vals = silhouette_samples(X, y_km, metric='euclidean')
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
    plt.axvline(silhouette_avg, color="red", linestyle="--")

    plt.yticks(yticks, cluster_labels + 1)
    plt.ylabel('Cluster')
    plt.xlabel('Silhouette coefficient')

    plt.tight_layout()
    # plt.savefig('images/11_04.png', dpi=300)
    plt.show()

def pca(training_set, test_set):
    pca = PCA()

    pca.fit_transform(training_set)
    pca.transform(test_set)

    explained_variance = pca.explained_variance_ratio_
    components = 16
    print("for "+str(components)+" components")
    top_n = explained_variance[:components]
    print(top_n)
    print("captures ")
    print(np.sum(top_n))
    print("percent")

    pca_cum_variance(pca)

    pca = PCA(n_components=16)
    X_train = pca.fit_transform(training_set)
    X_test = pca.transform(test_set)

    distortions = []
    for i in range(1, 11):
        km = KMeans(n_clusters=i,
                    init='k-means++',
                    n_init=10,
                    max_iter=300,
                    random_state=RAND)
        km.fit(X_train)
        distortions.append(km.inertia_)

    plt.plot(range(1, 11), distortions, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.title("Distortion vs # Clusters PCA-20")

    plt.tight_layout()
    plt.show()

    km = KMeans(n_clusters=3,
                init='k-means++',
                n_init=10,
                max_iter=300,
                random_state=RAND)

    plot_silhouette(km, X_train, title="PCA20, K=3")

    visualizer = InterclusterDistance(km)
    visualizer.fit(X_train)  # Fit the data to the visualizer
    visualizer.show()  # F



    km = KMeans(n_clusters=2,
                init='k-means++',
                n_init=10,
                max_iter=300,
                random_state=RAND)

    plot_silhouette(km, X_train, title="PCA20, K=2")

    visualizer = InterclusterDistance(km)
    visualizer.fit(X_train)  # Fit the data to the visualizer
    visualizer.show()  # F

    km = KMeans(n_clusters=4,
                init='k-means++',
                n_init=10,
                max_iter=300,
                random_state=RAND)

    plot_silhouette(km, X_train, title="PCA20, K=4")

    km = KMeans(n_clusters=5,
                init='k-means++',
                n_init=10,
                max_iter=300,
                random_state=RAND)

    plot_silhouette(km, X_train, title="PCA20, K=5")

def ica(training_set, test_set, y_train):
    # https://www.ritchieng.com/machine-learning-dimensionality-reduction-feature-transform/
    ica_avg_kurtosis_curve(training_set)

    km = KMeans(n_clusters=2,
                init='k-means++',
                n_init=10,
                max_iter=300,
                random_state=RAND)

    ica = FastICA(n_components=10, random_state=RAND, max_iter=1000)

    X_train = ica.fit_transform(training_set)

    plot_silhouette(km, X_train, title="ICA(10), K=2")
    visualizer = InterclusterDistance(km)
    visualizer.fit(X_train)  # Fit the data to the visualizer
    visualizer.show()  # F

    km = KMeans(n_clusters=5,
                init='k-means++',
                n_init=10,
                max_iter=300,
                random_state=RAND)

    ica = FastICA(n_components=10, random_state=RAND, max_iter=1000)

    X_train = ica.fit_transform(training_set)

    plot_silhouette(km, X_train, title="ICA(10), K=5")

    km = KMeans(n_clusters=3,
                init='k-means++',
                n_init=10,
                max_iter=300,
                random_state=RAND)

    ica = FastICA(n_components=10, random_state=RAND, max_iter=1000)

    X_train = ica.fit_transform(training_set)

    plot_silhouette(km, X_train, title="ICA(10)"
                                       ", K=3")

    km = KMeans(n_clusters=2,
                init='k-means++',
                n_init=10,
                max_iter=300,
                random_state=RAND)
    km.fit(X_train)

    hs=metrics.homogeneity_score(y_train, km.labels_)
    print("homogenatity score for K=2:",hs)

    y_train_inverse = (~y_train.astype(bool)).astype(int)

    hs=metrics.homogeneity_score(y_train_inverse, km.labels_)
    print("homogenatity score for K=2: (inverse)",hs)


def rp(X_train, X_test):
    num_components = johnson_lindenstrauss_min_dim(n_samples=X_train.shape[0], eps=0.1)
    print(num_components)
    print("# features: ", X_train.shape[1], " JL min dim:", num_components)
    print("JL number > #features so cant make any JL guarentees")
    # Of course not! It simply means that we can’t make any assumptions regarding the preservation of pairwise distances between data points.

    accuracies = []
    components = np.int32(np.linspace(2, 64, 20))

    model = LinearSVC()
    model.fit(X_train, y_train)
    baseline = metrics.accuracy_score(model.predict(X_test), y_test)

    # loop over the projection sizes
    for comp in components:
        # create the random projection
        sp = SparseRandomProjection(n_components=comp)
        X = sp.fit_transform(X_train)

        # train a classifier on the sparse random projection
        model = LinearSVC()
        model.fit(X, y_train)

        # evaluate the model and update the list of accuracies
        test = sp.transform(X_test)
        accuracies.append(metrics.accuracy_score(model.predict(test), y_test))

    # create the figure
    plt.figure()
    plt.title("Accuracy of Sparse Projection on Sonar")
    plt.xlabel("# of Components")
    plt.ylabel("Accuracy")
    plt.xlim([2, 64])
    plt.ylim([0, 1.0])

    # plot the baseline and random projection accuracies
    plt.plot(components, [baseline] * len(accuracies), color="r")
    plt.plot(components, accuracies)

    plt.show()
    #sp = SparseRandomProjection(n_components = comp)

def tree(X_train, X_test):
    clf = ExtraTreesClassifier(n_estimators=50, random_state=RAND)
    clf = clf.fit(X_train, y_train)
    print("Original Features Size:", X_train.shape[1])
    model = SelectFromModel(clf, prefit=True)
    X_new = model.transform(X)
    print("Number of features from feature importance:", X_new.shape[1])

    thresholds = np.sort(clf.feature_importances_)
    num_features = []
    accuracies = []

    for thresh in thresholds:
        # select features using threshold
        selection = SelectFromModel(clf, threshold=thresh, prefit=True)
        select_X_train = selection.transform(X_train)
        select_X_test = selection.transform(X_test)
        # train model
        selection_model = ExtraTreesClassifier(n_estimators=50, random_state=RAND)
        selection_model.fit(select_X_train, y_train)
        select_X_test = selection.transform(X_test)
        y_pred = selection_model.predict(select_X_test)
        predictions = [round(value) for value in y_pred]
        accuracy = accuracy_score(y_test, predictions)
        num_features.append(select_X_train.shape[1])
        accuracies.append(accuracy)
        print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy * 100.0))
        #print(classification_report(y_test, predictions))

    plt.plot(num_features, accuracies, '--', color="#111111", label="Accuracy")
    plt.plot(num_features, [accuracies[0]] * len(accuracies), color="r")
    # Create plot
    plt.title("Accuracy vs num features (Decision Tree)")
    plt.xlabel("Num Features"), plt.ylabel("Accuracy"), plt.legend(loc="best")
    plt.tight_layout()

    plt.show()

    clf = ExtraTreesClassifier(n_estimators=50)
    clf = clf.fit(X_train, y_train)
    print("Original Features Size:", X_train.shape[1])
    model = SelectFromModel(clf, prefit=True, max_features=3)
    X_new = model.transform(X)


        #TODO graph accuracy

# https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/decomposition/base.py

plain_clustering()
pca(X_train, X_test)
ica(X_train, X_test, y_train)
rp(X_train, X_test)
tree(X_train, X_test)



