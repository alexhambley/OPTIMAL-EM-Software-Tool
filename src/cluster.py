import logging

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler

import complexity
import constant
import feature_extraction
import interface
import statistics
from files import Files, SoupedFile
from src import samplesize


def run_dbscan_cluster_functions():
    """
    This function runs some other functions in this module to produce a matrix. This matrix, termed Y, is a set of
    labels that are then passed into TSNE to form coordinates. These coordinates are passed into DBSCAN and clustered.
    The results are then set in the SoupedFile class, and references to these classes are returned.
    :return: List[SoupedFile]: A list of SoupedFiles with their cluster_value variables set.
    """
    token_matrix, references_to_souped_files = produce_token_matrix()
    print(token_matrix)
    write_dbscan_labels(plot_dbscan(fit_tsne(token_matrix)), references_to_souped_files)
    return references_to_souped_files


def get_files_sorted_by_cluster(references_to_souped_files):
    list_y = []
    for i in references_to_souped_files:
        list_y.append([i.get_label(), i.get_cluster_value()])
    return sorted(list_y, key=lambda x: x[1])


def write_dbscan_labels(dbscan_labels, references_to_souped_files):
    for i, f in enumerate(references_to_souped_files):
        f.set_cluster_value(dbscan_labels[i])


def check_if_k_means_required(files, weights, cluster_variance, cluster_sizes):
    """
    Iterates through the clusters provided by DBSCAN. For each cluster, we obtain the files that belong to that
    cluster. Then, we measure if we need to conduct additional clustering via k-means. We take the second element
    of clusters_ordered_by_complexity as the threshold value. This means the top two clusters (in terms of complexity
    variance) are checked. If these clusters have a number of pages > threshold_size, then it will be further
    clustered by k-means, so that we have more representative samples. We assign threshold_complexity to
    cluster_variance[K_MEANS_COMPLEXITY_THRESHOLD]. The logic here is that this takes the
    K_MEANS_COMPLEXITY_THRESHOLD (default: 2) most complex clusters.
    :param files:
    :param weights:
    :param cluster_variance:
    :param cluster_sizes:
    :return df: a DataFrame object. We have a representative (sample) HTML file, followed by all the files the sample
                represents.
    """

    complexity_ordered_by_cluster = sorted(statistics.sort_list_of_clusters_by_mean_complexity(cluster_variance),
                                           key=lambda x: x[0])
    complexity_list = sorted([item[1] for item in statistics.
                             sort_list_of_clusters_by_mean_complexity(cluster_variance)], reverse=True)
    threshold_complexity = complexity_list[constant.K_MEANS_COMPLEXITY_THRESHOLD]
    print(complexity_list)

    # If we need to run k_means, we input the results of k_means into k_means_labels. The other lists are used for
    # calculating the size of the clusters, which (in turn) is used to calculate the sample size:
    counter = 0
    k_means_cluster_size = []
    sample_df = samplesize.create_df()

    # Here we are iterating through the clusters provided by DBSCAN:
    for i, c in enumerate(cluster_sizes):
        temporary_files = []
        for f in files:

            # Check if the DBSCAN cluster assigned to the HTML file is the same as the current cluster:
            if f.get_cluster_value() == c[1]:
                temporary_files.append(f)

        # Here, we check if the current cluster size is above a threshold value, and if the complexity of the current
        # cluster is greater than the threshold. (The logging is helpful for debugging.)
        logging.info("c[0] = " + str(c[0]) + " and constant.K_MEANS_THRESHOLD = " + str(constant.K_MEANS_THRESHOLD))
        logging.info("complexity_ordered_by_cluster[i][1] = " + str(complexity_ordered_by_cluster[i][1])
                     + " and threshold_complexity = " + str(threshold_complexity))
        print(str(c[0]) + " >= " + str(constant.K_MEANS_THRESHOLD) + "?")
        print(str(complexity_ordered_by_cluster[i][1]) + " >= " + str(threshold_complexity) + "?")
        if c[0] >= constant.K_MEANS_THRESHOLD and complexity_ordered_by_cluster[i][1] >= threshold_complexity:
            print("YES")
            # Now we run k-means on cluster c, and the cluster labels are returned to us:
            logging.info("Match: K-Means required:")
            k_means_labels = [plot_k_means(fit_tsne(CountVectorizer().fit_transform(
                [f.get_soup_data() for f in temporary_files]))).tolist()]
            print(np.unique(k_means_labels))
            unique_clusters = np.unique(k_means_labels)

            for cluster_number in unique_clusters:
                temp_at_k = []

                for index, label in enumerate(k_means_labels[0]):
                    if label == cluster_number:
                        temp_at_k.append(temporary_files[index])
                        counter = counter + 1

                k_means_cluster_size.append(counter)
                counter = 0

                # Now we calculate the sample size required for these "sub-clusters".
                # Here, we use a margin of error of 0.2. This is higher than the margin of error for the standard
                # sample size calculation. The justification is that as the sample is smaller (and we have already
                # clustered the pages) it matters less. If you want to change it, just change the sample size variable:
                k_means_margin_of_error = 0.2
                z_score = constant.SAMPLE_Z_SCORES[constant.Z_SCORE_X][constant.Z_SCORE_Y]

                k_means_sample_size = \
                    ((((z_score ** 2) * (constant.SAMPLE_POPULATION_PROPORTION *
                                         (1 - constant.SAMPLE_POPULATION_PROPORTION))) / (k_means_margin_of_error ** 2))
                        / (1 + (z_score ** 2) * (constant.SAMPLE_POPULATION_PROPORTION *
                                                 (1 - constant.SAMPLE_POPULATION_PROPORTION)) /
                           ((k_means_margin_of_error ** 2) * len(temporary_files))))

                for k in samplesize.get_weighted_sample(cluster_size=k_means_cluster_size,
                                                        population_size=len(temporary_files),
                                                        sample_size=k_means_sample_size):
                    samplesize.random_sample(sample_weights=k, files=temp_at_k, dbscan_cluster_number=i,
                                             from_k_means=cluster_number, sample_df=sample_df)
                k_means_cluster_size = []
        else:

            # Otherwise, we just sample what we have. For more information please see the random_sample() function in
            # the samplesize.py module:
            logging.info("No match: K-Means not required:")
            samplesize.random_sample(sample_weights=weights[i], files=temporary_files, dbscan_cluster_number=i,
                                     from_k_means=False, sample_df=sample_df)
    representative_files = sample_df.get_samples_df()
    representative_files.to_csv(constant.OUTPUT_FILENAME)
    return representative_files


def produce_token_matrix():
    """
    This function returns a document-term matrix and a list containing data regarding the HTML files. The document-term
    matrix is an array of shape (n_samples, n_features). n_samples should be the same as the number
    of files (length_of_files()) and n_features should be the number of unique 'words'. 'Words' are automatically
    produced through the functions present in the feature_extraction module. Examples include HTML tags or words in
    the page content.
    :return: Tuple[X, List[SoupedFile]]:
        X: An array of shape (n_samples, n_features).
        List[SoupedFile]: List of references to the HTML file.
    """
    files = Files()
    length_of_files = files.length_of_files()
    references_to_files = []
    for i in range(0, length_of_files):
        soup = files.get_soup(index=i)
        temp = feature_extraction.get_html_block_structure(soup=soup)
        # print(temp)
        souped_file = SoupedFile(
            label=files.get_label(index=i),
            # soup_data=feature_extraction.get_html_structure(soup=soup),
            soup_data=temp,
            complexity_data=complexity.run_standard_complexity(soup))
        references_to_files.append(souped_file)
        interface.print_progress_bar(i, length_of_files, 'Running dimensionality reduction:')
    return CountVectorizer().fit_transform([f.get_soup_data() for f in references_to_files]), references_to_files


def fit_tsne(X):
    """Runs TSNE on input document-term matrix X, and returns X_New, a ndarray of shape (n_samples, n_components).
    TSNE attributes are constants from the constant module.
    :param X: document-term matrix
    :return X_new: ndarray of shape (n_samples, n_components)
        Embedding of the training data in low-dimensional space.
    """
    return TSNE(n_components=constant.TSNE_NUM_COMPONENTS, perplexity=constant.TSNE_PERPLEXITY).fit_transform(X)


def plot_dbscan(X):
    """
    This function is used to compute and plot DBSCAN (density-based spatial clustering of applications with noise.)
    Citation here: https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.121.9220
    Code sourced and adapted from: https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html
    :param X: token_matrix
        Also referred to as a document-term matrix. Of shape (n_samples, n_features).
        Convention in ML to use uppercase X which breaks PEP 8 conventions, sorry!
    :return: labels: List
        These are the labels of the cluster. In other words, the cluster value results.
    """
    db = DBSCAN(eps=constant.DBSCAN_EPS, min_samples=constant.DBSCAN_MIN_SAMPLES).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    unique_labels = set(labels)
    c_map = plt.cm.get_cmap("Spectral")
    colors = c_map(np.linspace(0, 1, 12))

    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]
        class_member_mask = (labels == k)
        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=14)
        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=6)
    plt.title('Estimated number of clusters: %d' % n_clusters_)
    # plt.show()
    plt.savefig('clusters.png')

    """The following code is useful for debugging, if needed:"""
    logger = logging.getLogger('debugging')
    logger.setLevel(logging.INFO)
    logging.info('Estimated number of clusters: %d' % n_clusters_)
    logging.info('Estimated number of noise points: %d' % n_noise_)
    logging.info('Labels:' + str(labels))
    return labels


def plot_k_means(X):
    """
    Runs k means on the input variable X
    :param X: Matrix
        TSNE token matrix
    :return:
    """
    best_k, results = choose_best_k_for_k_means(MinMaxScaler().fit_transform(X), constant.K_RANGE)
    sns.set_style("darkgrid")
    model = KMeans(n_clusters=best_k)
    model.fit(X)
    # sns.scatterplot(x=X[:, 0], y=X[:, 1], c=model.labels_, cmap='cool')
    # sns.scatterplot(x=model.cluster_centers_[:, 0], y=model.cluster_centers_[:, 1], c=['black'])
    # logging.info("k_means labels:" + str(model.labels_))
    plt.show()
    return model.labels_


def choose_best_k_for_k_means(scaled_data, k_range):
    """
    Returns best_k, an estimated value for k
    :param scaled_data:
    :param k_range:
    :return: best_K:
    :return results:
    """
    ans = []
    for k in k_range:
        scaled_inertia = k_means_res(scaled_data, k)
        ans.append((k, scaled_inertia))
    results = pd.DataFrame(ans, columns=['k', 'Scaled Inertia']).set_index('k')
    best_k = results.idxmin()[0]
    return best_k, results


# From: https://towardsdatascience.com/an-approach-for-choosing-number-of-clusters-for-k-means-c28e614ecb2c
def k_means_res(scaled_data, k, alpha_k=0.02):
    """
    Returns scaled_inertia. A float used for estimating k value
    :param scaled_data: matrix
        Scaled data. Rows are samples and columns are features for clustering
    :param k: int
        current k for applying KMeans
    :param alpha_k: float
        manually tuned factor that gives penalty to the number of clusters
    :return: scaled_inertia: float
        scaled inertia value for current k
    """
    inertia_o = np.square((scaled_data - scaled_data.mean(axis=0))).sum()
    kmeans = KMeans(n_clusters=k, random_state=0).fit(scaled_data)
    scaled_inertia = kmeans.inertia_ / inertia_o + alpha_k * k
    return scaled_inertia
