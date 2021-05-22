import math

import nltk
import sys
import string
import os
nltk.download('stopwords')

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    mapping = dict()
    file_names = sorted([name for name in os.listdir(directory)])
    for file in file_names:
        with open(os.path.join(directory, file)) as file_content:
            mapping[file] = file_content.read()
    return mapping



def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """

    tokens = nltk.word_tokenize(document.lower().translate(str.maketrans('', '', string.punctuation)))
    for token in tokens:
        if not token.isalpha() or len(token) < 1 or token in nltk.corpus.stopwords.words("english"):
            tokens.remove(token)
    return tokens


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    idfs_map = dict()
    total_documents = len(documents)
    for document in documents:
        word_list = documents[document]
        word_set = set(word_list)
        for word in word_set:
            doc_count = 0
            for d in documents:
                if word in documents[d]:
                    doc_count += 1
            idfs_map[word] = math.log(total_documents / doc_count)
    return idfs_map


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the `n` top
    files that match the query, ranked according to tf-idf.
    """
    files_ranking_map = dict()

    for file_name in files.keys():
        file_content = files.get(file_name)
        tf_id = 0
        for word in query:
            tf = file_content.count(word)
            tf_id = tf_id + tf * idfs.get(word)
        files_ranking_map[tf_id] = file_name

    return get_n_highest_keys(files_ranking_map, n)


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    ranking_map = dict()

    for sentence in sentences.keys():
        set_sentence_words = set(sentences.get(sentence))
        density = len(query.intersection(set_sentence_words)) / len(set_sentence_words)
        score = 0
        for word in query.intersection(set_sentence_words):
            if word in idfs.keys():
                score = score + idfs.get(word)
        ranking_map[score, density] = sentence

    return get_n_highest_keys(ranking_map, n)


def get_n_highest_keys(dictionary, n):
    sorted_files_by_descending_rank = sorted(dictionary, reverse=True)
    n_best_files = []

    for i in range(0, n):
        n_best_files.append(dictionary.get(sorted_files_by_descending_rank[i]))
    return n_best_files

if __name__ == "__main__":
    main()
