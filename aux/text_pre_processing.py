import re
import numpy as np
from nltk.corpus import stopwords, words
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


def build_full_review_col(
    title_col="title_review",
    text_col="text",
):
    df[output_col] = df[title_col].fillna("") + " " + df[text_col].fillna("")
    return df


def pre_process_categories(row, col):
    if isinstance(row[column], str) and row[col].strip():
        categories = row[col]
    elif isinstance(row[col], (list, np.ndarray)):
        categories = " ".join(str(element) for element in row[col])
    else:
        categories = ""
    return categories


def combine_text_features(row, col1="text", col2="features", col3="descriprion"):

    if isinstance(row[col3], str) and row[col3].strip():
        description = row[col3]
    elif isinstance(row[col3], (list, np.ndarray)):
        description = " ".join(str(element) for element in row[col3])
    else:
        description = ""

    if isinstance(row[col2], (list, np.ndarray)) and len(row[col2]) > 0:
        features = " ".join(str(feature) for feature in row[col2])
    else:
        features = ""

    # combine title, description, and features
    combined = f"{row[col1]} {description} {features}".strip()
    return combined


def pre_process_text(df, input_col="text", output_col="pre_processed_text"):
    # remove special characters
    df[output_col] = df[input_col].str.lower().str.replace(r"[^a-z\s]", "", regex=True)
    # remove stop words and lemmatize
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    df[output_col] = df[output_col].apply(
        lambda text: tokenize_and_lemmatize(text, stop_words, lemmatizer)
    )

    return df

def tokenize_and_lemmatize(text, stop_words, lemmatizer):
    tokens = word_tokenize(text)
    filtered_tokens = [
        lemmatizer.lemmatize(word) for word in tokens if word not in stop_words
    ]
    return " ".join(filtered_tokens)


def pre_process_query(query):
    """
    Preprocess a single query string.
    """
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    query = query.lower()
    query = re.sub(r"[^a-z\s]", "", query)

    tokens = word_tokenize(query)

    # remove stopwords and lemmatize
    processed_tokens = [
        lemmatizer.lemmatize(token) for token in tokens if token not in stop_words
    ]

    return " ".join(processed_tokens)
