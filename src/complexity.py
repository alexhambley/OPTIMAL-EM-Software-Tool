from bs4 import BeautifulSoup

import constant


def run_standard_complexity(f: BeautifulSoup):
    """
    Measures the complexity of the web page. In this instance, complexity is defined as the
    page_rich_content_tags / page_regular_tags. Additonal methods, discussed in the paper, can be added as extensions
    or future work.
    :param f: BeautifulSoup
        Specific BeautifulSoup object of an HTML file.
    :return: int
        Complexity value. This is calculated as a fraction of page_rich_content_tags / page_regular_tags.
    """
    f.prettify()
    page_rich_content_tags = len(f.find_all(constant.RICH_CONTENT_TAGS))
    page_regular_tags = len(f.find_all())
    return page_rich_content_tags / page_regular_tags
