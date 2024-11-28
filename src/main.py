import cluster
import constant
import interface
import samplesize
import statistics

if __name__ == '__main__':
    """
    First, we calculate the document-term matrix of a list of HTML files. We input that into t-SNE, and then input the 
    t-SNE matrix to DBSCAN:
    """
    references_to_souped_files = cluster.run_dbscan_cluster_functions()

    """Next, we calculate the complexity variance of each cluster of HTML files:"""
    mean_complexity_list, internal_cluster_complexity_variance, cluster_sizes_list_all_clusters, population_size \
        = statistics.get_complexity_variance_per_cluster(references_to_souped_files)

    """
    We calculate the sample size weights for each cluster. This is the number of pages that should be manually reviewed 
    by an auditor:
    """
    weights = samplesize.get_weighted_sample(
        cluster_size=[item[0] for item in cluster_sizes_list_all_clusters],
        population_size=len(references_to_souped_files),
        sample_size=samplesize.get_sample_size_from_z_score(z_score_x=constant.Z_SCORE_X, z_score_y=constant.Z_SCORE_Y,
                                                            population_size=population_size))

    """
    Now, we check if we need to run k-means on any large and complex clusters. K-means is used in this stage as it is 
    deterministic and will introduce "clusters of clusters". This (hopefully) reduces the number of pages we need to 
    review from the large clusters.
    """
    # print(cluster_sizes_list_all_clusters)
    sample_pages = cluster.check_if_k_means_required(
        files=references_to_souped_files, weights=weights, cluster_variance=internal_cluster_complexity_variance,
        cluster_sizes=cluster_sizes_list_all_clusters)

    """
    Finally, we produce a markdown report of our findings:
    """
    print("count: " + str(len(references_to_souped_files)))
    interface.produce_markdown_report(sample_pages, len(references_to_souped_files))


