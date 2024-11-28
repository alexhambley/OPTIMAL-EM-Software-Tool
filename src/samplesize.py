import logging
import math
from random import sample

import pandas as pd

import constant


class Sample_D_F():
    def __init__(self):
        self.samples_df = pd.DataFrame(columns=['Page_Name', 'DBSCAN_Cluster_Number', 'From_K_Means',
                                                'Pages_Representative_Of'])

    def write_to_samples_df(self, page_name, dbscan_cluster_number, from_k_means, page_representative_of):
        # TODO: Change this from df.append() to df.concat()
        self.samples_df = self.samples_df.append({'Page_Name': page_name,
                                                  'DBSCAN_Cluster_Number': dbscan_cluster_number,
                                                  'From_K_Means': from_k_means,
                                                  'Pages_Representative_Of': page_representative_of}, ignore_index=True)

    def get_samples_df(self):
        return self.samples_df


def create_df():
    return Sample_D_F()


def random_sample(sample_weights, files, dbscan_cluster_number, from_k_means, sample_df):
    """

    :param sample_weights:
    :param files:
    :param dbscan_cluster_number:
    :param from_k_means:
    :param sample_df:
    :return:
    """
    samples = sample(files, sample_weights)
    for s in samples:
        sample_df.write_to_samples_df(page_name=s.get_label(),
                                      dbscan_cluster_number=dbscan_cluster_number,
                                      from_k_means=from_k_means,
                                      page_representative_of=[f.get_label() for f in files])


class SampleSize:
    """The SampleSize class is used to hold sample size data."""
    def __init__(self):
        self.sample_size = None

    def set_sample_size(self, x, y, population_size):
        """
        This function takes the desired Z_Score index and runs an equation to form a sample size.
        :param x: The first index
        :param y: The second index. Always 1. Present for potential future wok, changes to the Z_SCORES and readability.
            These are used together to navigate constant.SAMPLE_Z_SCORES. i.e. constant.SAMPLE_Z_SCORES[3][1]
        :param population_size: Int
        """
        z_score = constant.SAMPLE_Z_SCORES[x][y]
        self.sample_size = (
                (((z_score ** 2) * (constant.SAMPLE_POPULATION_PROPORTION * (
                        1 - constant.SAMPLE_POPULATION_PROPORTION))) / (constant.SAMPLE_MARGIN_OF_ERROR ** 2))
                / (1 + (z_score ** 2) * (constant.SAMPLE_POPULATION_PROPORTION
                                         * (1 - constant.SAMPLE_POPULATION_PROPORTION))
                   / ((constant.SAMPLE_MARGIN_OF_ERROR ** 2) * population_size)))

    def get_sample_size(self) -> int or AttributeError:
        """A simple getter method that returns the sample size (or an AttributeError if the sample size has not
        been generated yet)"""
        if self.sample_size is not None:
            return self.sample_size
        else:
            return AttributeError


def get_weighted_sample(cluster_size, population_size, sample_size) -> list:
    """

    :param cluster_size: List
    :param population_size: Int:
    :param sample_size: Float
    :return: weights: List
    """
    weights = []
    for j in cluster_size:
        logging.info("cluster_size: " + str(j))
        weights.append(math.ceil((j / population_size) * sample_size))
    return weights


def get_sample_size_from_z_score(z_score_x, z_score_y, population_size) -> int or AttributeError:
    """
    A utility function that invokes class methods and returns the sample size or an AttributeError.
    :param z_score_x: The first index
    :param z_score_y: The second index. Should always be 1.
    :param population_size: Int
    :return: int: The sample size, or:
             AttributeError: If produce_sample_size() has not been called first.
    """
    s = SampleSize()
    s.set_sample_size(z_score_x, z_score_y, population_size)
    try:
        return s.get_sample_size()
    except AttributeError:
        return AttributeError
