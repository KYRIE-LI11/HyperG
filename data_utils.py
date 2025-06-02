import re
import unicodedata

from data import  CAP_TAG, HEADER_TAG, ROW_TAG, ROW_DESCRIPTION_TAG, COL_DESCRIPTION_TAG, MISSING_CAP_TAG, MISSING_CELL_TAG, MISSING_HEADER_TAG,NEG_CLA_TAG, POS_CLA_TAG
MAX_ROW_LEN = 100
MAX_COL_LEN = 100
MAX_WORD_LEN = 128


def clean_wiki_template(text):
    if re.match(r'^{{.*}}$', text):
        text = text[2:-2].split('|')[-1]  
    else:
        text = re.sub(r'{{.*}}', '', text)

    return text

def sanitize_text(text, entity="cell", replace_missing=True):
    """
    Clean up text in a table to ensure that the text doesn't accidentally
    contain one of the other special table tokens / tags.

    :param text: raw string for one cell in the table
    :return: the cell string after sanitizing
    """
    #breakpoint()
    rval = re.sub(r"\|+", " ", text).strip()
    rval = re.sub(r'\s+', ' ', rval).strip()
    if rval and rval.split()[0] in ['td', 'th', 'TD', 'TH']:
        rval = ' '.join(rval.split()[1:])

    rval = rval.replace(CAP_TAG, "")
    rval = rval.replace(HEADER_TAG, "")
    rval = rval.replace(ROW_TAG, "")

    rval = rval.replace(MISSING_CAP_TAG , "")
    rval = rval.replace(MISSING_CELL_TAG, "")
    rval = rval.replace(MISSING_HEADER_TAG, "")
    rval =  ' '.join(rval.strip().split()[:MAX_WORD_LEN])

    if (rval == "" or rval.lower() == "<missing>" or rval.lower() == "missing") and replace_missing:
        if entity == "cell":
            rval = MISSING_CELL_TAG
        elif entity == "header":
            rval = MISSING_HEADER_TAG
        else:
            rval = MISSING_CAP_TAG
    return rval

def clean_cell_value(cell_val):
    if isinstance(cell_val, list):
        val = ' '.join(cell_val)
    else:
        val = cell_val
    val = unicodedata.normalize('NFKD', val)
    val = val.encode('ascii', errors='ignore')
    val = str(val, encoding='ascii')
    val = clean_wiki_template(val)
    val = re.sub(r'\s+', ' ', val).strip()
    val = sanitize_text(val)
    return val
