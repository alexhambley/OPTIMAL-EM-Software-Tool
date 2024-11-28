from bs4 import BeautifulSoup
from constant import INLINE_TAGS


def get_html_structure(soup):
    """Returns just the HTML tags in a string, e.g. "head meta div div div". Content is stripped. The order of the tags
    is preserved but this does not affect clustering or dimensionality reduction at the moment. Default solution."""
    tags = ""
    for tag in soup.find_all():
        tags = tags + str(tag.name) + " "
    return tags


def get_html_block_structure(soup):
    """
    Returns just the HTML block-level tags in a string, e.g. "head meta div div div". Content is stripped. The order of the tags
    is preserved but this does not affect clustering or dimensionality reduction at the moment.
    """
    tags = ""
    for tag in soup.find_all():
        if tag.name not in INLINE_TAGS:
            tags += str(tag.name) + " "
    return tags


def get_html_tags(soup):
    """Returns the HTML tags. Content is stripped. Order of the HTML tags is not preserved."""
    tags = []
    for tag in soup.find_all():
        tags.append(tag.name)
    return tags


def get_html_structure_tree(soup):
    """Returns the HTML tree structure. No class, id, or other information is returned."""
    for tag in soup.find_all():
        if not tag.find(recursive=False):
            tag.string = ''
        tag.attrs = {}
    return soup.prettify()


def get_html_structure_attrs(soup):
    """Returns a list of tags in a string, and their attributes. For example, 'head div class megaContent class row.'"""
    tags = []
    for tag in soup.find_all():
        tags.append(tag.name)
        total_string = ""
        for key, value in tag.attrs.items():
            total_string += str(key) + " " + str(value)
            if isinstance(value, list):
                value_list = ''.join(map(str, value))
            elif isinstance(value, BeautifulSoup.element.CharsetMetaAttributeValue):
                value_list = ""
            else:
                value_list = str(value)
            total_string += str(key) + " " + value_list
        tags[-1] += total_string
    return tags


def get_html_structure_attrs_body(soup):
    tags = []
    for tag in soup.find_all():
        if tag.name not in INLINE_TAGS:  # Only process block-level tags
            tags.append(tag.name)
            total_string = ""
            for key, value in tag.attrs.items():
                if isinstance(value, list):
                    value_list = ''.join(map(str, value))
                else:
                    value_list = str(value)
                total_string += f" {key} {value_list}"  # Append key and value to total_string
            tags[-1] += total_string  # Append the attribute string to the tag
    return tags


def get_html_content(soup):
    """Returns the text content present on the web page."""
    content = ""
    for tag in soup.find_all():
        if not tag.find(recursive=False):
            cont = ' '.join(map(str, tag.contents))
            content = content + cont + " "
    return content


def get_html_structure_content(soup):
    """Returns a mixture of the HTML structure and content. Classes, etc. are present."""
    return soup.prettify()
