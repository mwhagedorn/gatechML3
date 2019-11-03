try https://www.scikit-yb.org/en/latest/api/cluster/icdm.html

 believe its:
K means PCA Dataset1        K means PCA dataset2
K means ICA dataset1        K means ICA dataset2
K means RP dataset1         K means RP datsaet2
K means Tree dataset1       K means Tree dataset2
EM PCA dataset 1            EM PCA dataset 2
EM ICA dataset 1            EM ICA dataset 2
EM RP dataset 1             EM RP dataset 2
EM Tree dataset 1           EM Tree dataset 2

Basically (2x clustering algo) X (4x dim reduction) X (2x dataset)

First 3 parts are 2 datasets
Part 1 kmeans and em on original datasets
Part 2 dimension reduction/transformation on both datasets
Part 3 clustering on feature reduction/transformations from part 2
Part 4 nn on feature reduction/ transformation for one dataset
Part 5 same dataset as part 4, but use original data and concatenate clustering info (once for em, once for kmeans) (edited)


Tala 1:10 AM
Part 1 should have 4 results
Part 2 8
Part 3 16 (I think it tells you that in the instructions)
Part 4 : 4 results with the baseline from a1 as a 5th
Part 5 : 2

homogenaity?
 metrics.homogeneity_score
# do a kmeans.fit -> estimator, estimator.labels_.. compared to y
