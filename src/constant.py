BLOCK_TAGS = ['address', 'article', 'aside', 'blockquote', 'canvas', 'dd', 'div', 'dl', 'dt', 'fieldset', 'figcaption',
              'figure', 'footer', 'form', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'header', 'hr', 'li', 'main', 'nav',
              'noscript', 'ol', 'p', 'pre', 'section', 'table', 'tfoot', 'ul', 'video']
DBSCAN_EPS = 4  # Default: 4. The higher this value, the fewer the number of clusters, but also less accurate.
DBSCAN_MIN_SAMPLES = 3
IS_BLOCK = False
IS_INLINE = False
INLINE_TAGS = ['a', 'abbr', 'acronym', 'b', 'bdo', 'big', 'br', 'button', 'cite', 'code', 'dfn', 'em', 'i', 'img',
               'input', 'kbd', 'label', 'map', 'object', 'output', 'q', 'samp', 'script', 'select', 'small', 'span',
               'strong', 'sub', 'sup', 'textarea', 'time', 'tt', 'var']
INTERFACE_MD_FILE_HEADING = "ArsTechnica Report"
INTERFACE_MD_FILENAME = "ArsTechnica_Report"
K_ALPHA = 0.02
K_RANGE = range(1, 5)
K_MEANS_COMPLEXITY_THRESHOLD = 2
K_MEANS_THRESHOLD = 30  # This is the threshold for the size of the *DBSCAN cluster*.
OUTPUT_FILENAME = 'output' + '.csv'
PARSER = "html.parser"
PATH_ACCESSIBILITY_RESULTS = '../res/output_a11y.txt'
PATH_HTML = 'res/res-ars/'
RICH_CONTENT_TAGS = ['a', 'audio', 'button', 'canvas', 'embed', 'iframe', 'img', 'input', 'keygen', 'label', 'math',
                     'object', 'select', 'svg', 'textarea', 'video']
SAMPLE_MARGIN_OF_ERROR = 0.2
SAMPLE_POPULATION_PROPORTION = 0.5
# SAMPLE_POPULATION_SIZE = 1300
# Precomputed Z-critical values: [[Cumulative Probability (%), Z-critical value]]
SAMPLE_Z_SCORES = [[70, 1.04], [75, 1.15], [80, 1.28], [85, 1.44], [90, 1.645], [91, 1.70], [92, 1.75], [93, 1.81],
                   [94, 1.88], [95, 1.96], [96, 2.05], [97, 2.17], [98, 2.33], [99, 2.576], [99.5, 2.807], [99.9, 3.29],
                   [99.99, 3.89]]
TSNE_NUM_COMPONENTS = 2
TSNE_PERPLEXITY = 20    # Default: 10, W3: 50
Z_SCORE_X = 3   # This is used for accessing the index. The higher the value the greater the sample size required.
Z_SCORE_Y = 1   # Should always be 1 as we use these values to navigate SAMPLE_Z_SCORES
