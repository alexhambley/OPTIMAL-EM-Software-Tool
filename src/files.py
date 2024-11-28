import os

from bs4 import BeautifulSoup

import constant


class Files:
    """The Files class is used to hold file data. It provides a number of getter methods for files
    and BeautifulSoup objects"""
    def __init__(self):
        self.list_of_html_files = [f for f in os.listdir(constant.PATH_HTML) if f.endswith('.html')]

    def get_list_of_files(self):
        """Returns a list of the HTML files imported."""
        return self.list_of_html_files

    def get_soup(self, index: int) -> BeautifulSoup:
        """
        Returns a BeautifulSoup object for the file at the specified index.
        :param index: int
            Index for the list_of_html_files list
        :return: BeautifulSoup object
        """
        return BeautifulSoup(open(constant.PATH_HTML + self.list_of_html_files[index]), constant.PARSER)

    def get_label(self, index: int):
        """
        Returns string containing the HTML file content at the specified index.
        :param index: int
        :return: string
        """
        return self.list_of_html_files[index]

    def length_of_files(self):
        """Returns the length of the list. This is the number of pages imported, or the target population."""
        return len(self.list_of_html_files)


class SoupedFile:
    """The SoupedFiles class is used to hold data regarding all the HTML files that have been run through
    BeautifulSoup. For convenience, we also use this class to store the labels, HTML (soup) and complexity data."""
    def __init__(self, label, soup_data, complexity_data):
        self.label = label
        self.soup_data = soup_data
        self.complexity_data = complexity_data
        self.cluster_value = None

    def get_label(self):
        """Returns the HTML label. This is the title of the HTML page."""
        return self.label

    def get_soup_data(self):
        """Returns the soup. The exact content of self.soup_data depends on the function invoked by
        cluster.produce_token_matrix(). Please see the functions in the feature_extraction.py module for more
        information."""
        return self.soup_data

    def get_complexity_data(self):
        """Returns the complexity data. This is a numerical representation of complexity. More information is present
        in the complexity.run_standard_complexity() function."""
        return self.complexity_data

    def set_cluster_value(self, cluster_value):
        self.cluster_value = cluster_value

    def get_cluster_value(self):
        return self.cluster_value
