import numpy as np
from sklearn import mixture, metrics
import itertools
from matplotlib import cm

from sklearn.decomposition import PCA, FastICA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import silhouette_samples
from sklearn.mixture import GaussianMixture
from sklearn.random_projection import SparseRandomProjection
from sklearn.svm import LinearSVC

from churn.load_data import load_churn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import linalg

RAND = 23

X, y = load_churn()

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, stratify=y)


def plot_silhouette(km, training_data, title='Silhoutte for KM'):
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
    print("Silhouette score mean:", silhouette_avg, title)
    plt.axvline(silhouette_avg, color="red", linestyle="--")

    plt.yticks(yticks, cluster_labels + 1)
    plt.ylabel('Cluster')
    plt.xlabel('Silhouette coefficient')

    plt.tight_layout()
    plt.title(title)
    plt.show()


def SelBest(arr, X):
    '''
    returns the set of X configurations with shorter distance
    '''
    dx = np.argsort(arr)[:X]
    return arr[dx]

#Courtesy of https://stackoverflow.com/questions/26079881/kl-divergence-of-two-gmms. Here the difference is that we take the squared root, so it's a proper metric

def gmm_js(gmm_p, gmm_q, n_samples=10**5):
    X = gmm_p.sample(n_samples)[0]
    log_p_X = gmm_p.score_samples(X)
    log_q_X = gmm_q.score_samples(X)
    log_mix_X = np.logaddexp(log_p_X, log_q_X)

    Y = gmm_q.sample(n_samples)[0]
    log_p_Y = gmm_p.score_samples(Y)
    log_q_Y = gmm_q.score_samples(Y)
    log_mix_Y = np.logaddexp(log_p_Y, log_q_Y)

    return np.sqrt((log_p_X.mean() - (log_mix_X.mean() - np.log(2))
            + log_q_Y.mean() - (log_mix_Y.mean() - np.log(2))) / 2)


def plain_em(training_data):
    n_components = np.arange(1, 21)
    models = [mixture.GaussianMixture(n, covariance_type='full', random_state=RAND).fit(training_data)
              for n in n_components]

    plt.plot(n_components, [m.bic(training_data) for m in models], label='BIC')
    plt.legend(loc='best')
    plt.xlabel('n_components')
    plt.show()

    print("Best Components Score is 17")

    # https://towardsdatascience.com/gaussian-mixture-model-clusterization-how-to-select-the-number-of-components-clusters-553bef45f6e4
    n_clusters = np.arange(2, 10)
    sils = []
    sils_err = []
    iterations = 20
    for n in n_clusters:
        tmp_sil = []
        for _ in range(iterations):
            print(n, len(tmp_sil))
            gmm = GaussianMixture(n, n_init=2)
            labels = gmm.fit_predict(X_train)
            sil = metrics.silhouette_score(X_train, labels, metric='euclidean')
            tmp_sil.append(sil)
        val = np.mean(np.array(tmp_sil))
        err = np.std(tmp_sil)
        sils.append(val)
        sils_err.append(err)
    plt.errorbar(n_clusters, sils, yerr=sils_err)
    plt.title("Silhouette Scores EM Churn", fontsize=20)
    plt.xticks(n_clusters)
    plt.xlabel("N. of clusters")
    plt.ylabel("Score")
    plt.show()

    n_clusters = np.arange(2, 10)
    iterations = 20
    results = []
    res_sigs = []
    for n in n_clusters:
        dist = []

        for iteration in range(iterations):
            train, test =train_test_split(X_train, test_size=0.5)

            gmm_train = GaussianMixture(n, n_init=2).fit(train)
            gmm_test = GaussianMixture(n, n_init=2).fit(test)
            dist.append(gmm_js(gmm_train, gmm_test))
        selec = SelBest(np.array(dist), int(iterations / 5))
        result = np.mean(selec)
        res_sig = np.std(selec)
        results.append(result)
        res_sigs.append(res_sig)

    plt.errorbar(n_clusters, results, yerr=res_sigs)
    plt.title("Distance between Train and Test GMMs", fontsize=20)
    plt.xticks(n_clusters)
    plt.xlabel("N. of clusters")
    plt.ylabel("Distance")
    plt.show()

    # 16 components

    # lowest_bic = np.infty
    # bic = []
    # n_components_range = range(1, 10)
    # cv_types = ['spherical', 'tied', 'diag', 'full']
    # for cv_type in cv_types:
    #     for n_components in n_components_range:
    #         # Fit a Gaussian mixture with EM
    #         gmm = mixture.GaussianMixture(n_components=n_components,
    #                                       covariance_type=cv_type, random_state=23)
    #         gmm.fit(training_data)
    #         bic.append(gmm.bic(training_data))
    #         if bic[-1] < lowest_bic:
    #             lowest_bic = bic[-1]
    #             best_gmm = gmm
    #
    # bic = np.array(bic)
    # color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
    #                               'darkorange'])
    # clf = best_gmm
    # bars = []
    # print("lowest bic:",lowest_bic)
    # # Plot the BIC scores
    # plt.figure(figsize=(8, 6))
    # spl = plt.subplot(2, 1, 1)
    # for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
    #     xpos = np.array(n_components_range) + .2 * (i - 2)
    #     bars.append(plt.bar(xpos, bic[i * len(n_components_range):
    #                                   (i + 1) * len(n_components_range)],
    #                         width=.2, color=color))
    # plt.xticks(n_components_range)
    # plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
    # plt.title('BIC score per model')
    # xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 + \
    #        .2 * np.floor(bic.argmin() / len(n_components_range))
    # plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
    # spl.set_xlabel('Number of components')
    # spl.legend([b[0] for b in bars], cv_types)
    # plt.show()
    # exit(1)
    # # Plot the winner
    # splot = plt.subplot(2, 1, 2)
    # Y_ = clf.predict(training_data)
    # for i, (mean, cov, color) in enumerate(zip(clf.means_, clf.covariances_,
    #                                            color_iter)):
    #     v, w = linalg.eigh(cov)
    #     if not np.any(Y_ == i):
    #         continue
    #     plt.scatter(training_data[Y_ == i, 0], training_data[Y_ == i, 1], .8, color=color)
    #
    #     # Plot an ellipse to show the Gaussian component
    #     angle = np.arctan2(w[0][1], w[0][0])
    #     angle = 180. * angle / np.pi  # convert to degrees
    #     v = 2. * np.sqrt(2.) * np.sqrt(v)
    #     ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
    #     ell.set_clip_box(splot.bbox)
    #     ell.set_alpha(.5)
    #     splot.add_artist(ell)
    #
    # plt.xticks(())
    # plt.yticks(())
    # plt.title('Selected GMM: full model, 2 components')
    # plt.subplots_adjust(hspace=.35, bottom=.02)
    # plt.show()


def pca_cum_variance(pca):
    plt.rcParams["figure.figsize"] = (12, 6)

    fig, ax = plt.subplots()
    xi = np.arange(1, 20, step=1)
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


def pca(training_set):
    pca = PCA()

    pca.fit_transform(training_set)

    explained_variance = pca.explained_variance_ratio_
    components = 1
    print("for " + str(components) + " components")
    top_n = explained_variance[:components]
    print(top_n)
    print("captures ")
    print(np.sum(top_n))
    print("percent")

    pca_cum_variance(pca)

    pca = PCA(n_components=1)
    X_train = pca.fit_transform(training_set)

    distortions = []
    for i in range(1, 20):
        gmm = mixture.GaussianMixture(i, covariance_type='full', random_state=RAND)
        gmm.fit(X_train)
        distortions.append(gmm.bic(X_train))

    plt.plot(range(1, 20), distortions, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.title("bic vs # components GMM-1, PCA w/1")

    plt.tight_layout()
    plt.show()

    n_clusters = np.arange(2, 10)
    iterations = 20
    results = []
    res_sigs = []
    for n in n_clusters:
        dist = []

        for iteration in range(iterations):
            train, test = train_test_split(X_train, test_size=0.5)

            gmm_train = GaussianMixture(n, n_init=2).fit(train)
            gmm_test = GaussianMixture(n, n_init=2).fit(test)
            dist.append(gmm_js(gmm_train, gmm_test))
        selec = SelBest(np.array(dist), int(iterations / 5))
        result = np.mean(selec)
        res_sig = np.std(selec)
        results.append(result)
        res_sigs.append(res_sig)

    plt.errorbar(n_clusters, results, yerr=res_sigs)
    plt.title("Distance between Train and Test GMMs", fontsize=20)
    plt.xticks(n_clusters)
    plt.xlabel("N. of clusters")
    plt.ylabel("Distance")
    plt.show()


def ica(training_set):
    gmm = mixture.GaussianMixture(4, covariance_type='full', random_state=RAND)

    ica = FastICA(n_components=3, random_state=RAND, max_iter=1000)
    X_train = ica.fit_transform(training_set)

    plot_silhouette(gmm, X_train, title="GMM ICA, K=4")

    gmm = mixture.GaussianMixture(5, covariance_type='full', random_state=RAND)

    ica = FastICA(n_components=3, random_state=RAND, max_iter=1000)

    X_train = ica.fit_transform(training_set)

    plot_silhouette(gmm, X_train, title="GMM ICA, K=5")

    gmm = mixture.GaussianMixture(3, covariance_type='full', random_state=RAND)

    ica = FastICA(n_components=3, random_state=RAND, max_iter=1000)

    X_train = ica.fit_transform(training_set)

    plot_silhouette(gmm, X_train, title="GMM ICA, K=3")

    gmm = mixture.GaussianMixture(2, covariance_type='full', random_state=RAND)

    ica = FastICA(n_components=3, random_state=RAND, max_iter=1000)

    X_train = ica.fit_transform(training_set)

    plot_silhouette(gmm, X_train, title="ICA, K=2")


def rp(X_train, y_train, X_test, y_test):
    accuracies = []
    components = np.int32(np.linspace(2, 64, 20))

    model = LinearSVC()
    model.fit(X_train, y_train)
    baseline = metrics.accuracy_score(model.predict(X_test), y_test)

    # loop over the projection sizes
    # for comp in components:
    #     # create the random projection
    #     sp = SparseRandomProjection(n_components=comp, random_state=RAND)
    #     X = sp.fit_transform(X_train)
    #
    #     # train a classifier on the sparse random projection
    #     model = LinearSVC()
    #     model.fit(X, y_train)
    #
    #     # evaluate the model and update the list of accuracies
    #     test = sp.transform(X_test)
    #     accuracies.append(metrics.accuracy_score(model.predict(test), y_test))
    #
    # # create the figure
    # plt.figure()
    # plt.title("Accuracy of Sparse Projection on Churn")
    # plt.xlabel("# of Components")
    # plt.ylabel("Accuracy")
    # plt.xlim([2, 64])
    # plt.ylim([0, 1.0])
    #
    # # plot the baseline and random projection accuracies
    # plt.plot(components, [baseline] * len(accuracies), color="r")
    # plt.plot(components, accuracies)
    #
    # print("Average of 4 runs, first better than baseline ave 12 components")
    #
    # plt.show()

    sp = SparseRandomProjection(n_components=12, random_state=RAND)
    X = sp.fit_transform(X_train)

    em = mixture.GaussianMixture(2, covariance_type='full', random_state=RAND)

    plot_silhouette(em, X, title="RP, K=2, 12 RC")

    em = mixture.GaussianMixture(3, covariance_type='full', random_state=RAND)

    plot_silhouette(em, X, title="RP, K=3, 12 RC")

    em = mixture.GaussianMixture(4, covariance_type='full', random_state=RAND)

    plot_silhouette(em, X, title="RP, K=4, 12 RC")


def tree(x_train, y_train):
    clf = ExtraTreesClassifier(n_estimators=50)
    clf = clf.fit(X_train, y_train)
    print("Original Features Size:", X_train.shape[1])
    model = SelectFromModel(clf, prefit=True, max_features=3)
    X_new = model.transform(X_train)

    em = mixture.GaussianMixture(2, covariance_type='full', random_state=RAND)

    plot_silhouette(em, X_new, title="EMM, K=2, tree(3)")

    em = mixture.GaussianMixture(3, covariance_type='full', random_state=RAND)

    plot_silhouette(em, X_new, title="RP, K=3, tree")

    em = mixture.GaussianMixture(4, covariance_type='full', random_state=RAND)

    plot_silhouette(em, X_new, title="RP, K=4, tree")


plain_em(X_train)
pca(X_train)
ica(X_train)
rp(X_train, y_train, X_test, y_test)
tree(X_train, y_train)
