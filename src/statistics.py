import numpy as np

import interface


def get_complexity_variance_per_cluster(references_to_souped_files):
    """
    This function is an algorithm that calculates the complexity variance, per cluster, of the HTML files.
    Each HTML file has a complexity associated with it, calculated in the complexity module. This function groups each
    page by the DBSCAN cluster and measures the complexity - and the variance of that complexity.
    Complexity (simply) demonstrates the complexity of that cluster, but the variance demonstrates the disparity of said
    cluster. The algorithm is annotated throughout, and pseudocode is present in the paper. See the README.md for more
    detail.
    :param references_to_souped_files: List[SoupedFile]
    :returns:
        List[Float]: List[Mean complexity per cluster],
        ndarray: (List[Float]): List[Mean internal complexity per cluster],
        List[List[int]]: List[[Cluster size], [Cluster number]]
    """
    # We declare the lists here (just for readability):
    complexity_per_cluster_list = []
    cluster_list = []
    cluster_sizes_list = []
    complexity_list = []
    # complexity_list = []

    internal_cluster_complexity_variance = np.array([])
    mean_complexity_list = []

    # First, we create two lists. This is a list of the cluster values from the HTML files in our target population, and
    # a list of the complexity.
    for f in references_to_souped_files:
        cluster_list.append(f.get_cluster_value())
        complexity_list.append([f.get_cluster_value(), f.get_complexity_data()])

        # complexity_list.append([f.get_cluster_value(), f.get_complexity_data()])

    print("complexity_list")
    print(complexity_list)

    # We then calculate the number of unique clusters that DBSCAN(TSNE) produced:
    unique_clusters = np.unique(cluster_list)

    # Now, we iterate through this unique_clusters list.
    for i, cluster_number in enumerate(unique_clusters):
        print("cluster: " + str(cluster_number))
        interface.print_progress_bar(i, len(unique_clusters), 'Calculating complexity variance:')
        for j in complexity_list:
            if j[0] == cluster_number:
                print(j)
                # At each index we find the complexity of each HTML file that belongs to that cluster:
                complexity_per_cluster_list.append(j[1])

        # We also calculate the size of each cluster to later calculate the mean:
        cluster_sizes_list.append([len(complexity_per_cluster_list), cluster_number])

        # Next, we calculate the complexity of each cluster:
        mean_complexity_list.append(sum(complexity_per_cluster_list) / len(complexity_per_cluster_list))
        internal_cluster_complexity_variance = np.append(
            internal_cluster_complexity_variance,
            [np.var(np.array(complexity_per_cluster_list), ddof=1, axis=0)], axis=0)

        # Reset the values for the next iteration:
        complexity_per_cluster_list = []

    print("mean_complexity_list")
    print(mean_complexity_list)

    print("internal_cluster_complexity_variance")
    print(internal_cluster_complexity_variance)

    return mean_complexity_list, internal_cluster_complexity_variance.tolist(), cluster_sizes_list, len(cluster_list)


def sort_list_of_clusters_by_mean_complexity(list_of_cluster_complexity):
    """This is a simple helper function that will sort a list of clusters by its complexity or complexity variance.
    It takes a list of complexity (or variance) and returns a list of a list.
    e.g. [list_of_cluster_complexity] -> [[cluster number], [complexity]]
         [list_of_cluster_complexity_variance] -> [[cluster number], [complexity_variance]]
    :param list_of_cluster_complexity: List[Float]
    :return Y: List[List[]]:
    """
    temp = []
    for i in range(-1, len(list_of_cluster_complexity) - 1):
        temp.append([i, list_of_cluster_complexity[i]])
    return sorted(temp, key=lambda x: x[1])

