from mdutils import MdUtils

import constant


def print_progress_bar(iteration, total, prefix):
    """
    Call in a loop to create terminal progress bar. Adapted from:
    https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters?noredirect=1&lq=1
    :param iteration: int: current iteration
    :param total: int: total iterations
    :param prefix: str: prefix string
    """
    suffix = 'Complete'
    decimals = 1
    length = 50
    fill = '█'
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='')
    if iteration == total:
        print()


def produce_markdown_report(samples, number_of_population_pages):
    """
    This method creates and formats a markdown (.md) file.
    :return:
    """
    sample_pages = samples.iloc[:, 0]
    cluster_number = samples.iloc[:, 1]
    standard_pages = samples.iloc[:, 3]
    number_of_sample_pages = len(sample_pages)

    """Markdown formatting and file populating:"""
    md_file = MdUtils(file_name=constant.INTERFACE_MD_FILENAME, title=constant.INTERFACE_MD_FILE_HEADING)
    md_file.new_header(level=1, title="Report")

    md_file.new_paragraph(
        "This report details a list of pages that should be reviewed. These pages are selected as they are similar to "
        "the pages within their cluster. The image below shows the clusters of the website. Each cluster contains a "
        "number of pages that are similar."
    )

    """Load in the image we created - see the cluster.plot_dbscan(X) method."""
    md_file.new_paragraph(md_file.new_inline_image("Clusters for website", 'clusters.png'))

    md_file.new_paragraph(
        "Representative pages are selected based on a confidence level. The higher the confidence, the more pages "
        "reviewed, and the lower the confidence, the fewer pages reviewed -  with a greater degree of error. Large "
        "clusters may need more pages to be reviewed, as there may be greater variance in these clusters. We can "
        "avoid reviewing some clusters, reducing the number of pages reviewed, however there will be less coverage, "
        "and key pages may be missed."
    )
    md_file.new_paragraph("***")
    md_file.new_line()
    md_file.new_header(level=2, title="Site Statistics")

    md_file.new_line()
    md_file.new_paragraph(
        "Sample pages: " + str(number_of_sample_pages)
    )
    md_file.new_line()
    md_file.new_paragraph(
        "Total pages in population: " + str(number_of_population_pages)
    )
    md_file.new_line()
    md_file.new_paragraph(
        "Reduction: " + str(round((100 - (number_of_sample_pages / number_of_population_pages) * 100), 3)) + "%"
    )
    md_file.new_line()
    md_file.new_paragraph(
        "Margin of Error: " + str(constant.SAMPLE_MARGIN_OF_ERROR)
    )
    md_file.new_line()
    md_file.new_paragraph(
        "Confidence Level: " + str(constant.SAMPLE_Z_SCORES[constant.Z_SCORE_X][0]) + "%"
    )
    md_file.new_line()
    md_file.new_paragraph("***")
    md_file.new_line()
    md_file.new_header(level=2, title="Sample Pages")
    # md_file.new_header(2, "Pages")
    md_table = ["Sample Page:", "Cluster Number:", "Representing:"]
    for i in range(number_of_sample_pages):
        md_table.extend([str(sample_pages[i]),
                         "Cluster: " + str(cluster_number[i]),
                         str(len(standard_pages[i])) + " pages"])
    md_file.new_table(columns=3, rows=number_of_sample_pages+1, text=md_table, text_align='left')
    md_file.new_line()
    md_file.new_paragraph("***")
    md_file.new_line()
    md_file.new_table_of_contents(table_title='Report Contents', depth=2)
    md_file.create_md_file()

