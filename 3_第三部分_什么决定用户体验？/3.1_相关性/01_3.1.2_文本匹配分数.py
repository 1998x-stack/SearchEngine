# 01_3.1.2_文本匹配分数

"""
Lecture: 3_第三部分_什么决定用户体验？/3.1_相关性
Content: 01_3.1.2_文本匹配分数
"""

import numpy as np
from scipy.sparse import csr_matrix
from typing import List, Tuple

class TfIdfVectorizer:
    """
    TF-IDF Vectorizer using numpy and scipy

    Attributes:
        vocabulary_ (dict): A dictionary where keys are terms and values are term indices.
        idf_ (np.ndarray): Array of inverse document frequencies for terms in the vocabulary.
    """

    def __init__(self) -> None:
        """Initialize the vectorizer with an empty vocabulary and IDF array."""
        self.vocabulary_ = {}
        self.idf_ = None

    def fit(self, documents: List[str]) -> 'TfIdfVectorizer':
        """
        Learn the vocabulary and IDF from the list of documents.

        Args:
            documents (List[str]): List of documents as strings.

        Returns:
            self (TfIdfVectorizer): The fitted vectorizer.
        """
        # 创建词汇表
        self._build_vocabulary(documents)
        
        # 计算IDF
        self._calculate_idf(documents)
        
        return self

    def transform(self, documents: List[str]) -> csr_matrix:
        """
        Transform documents into TF-IDF vectors.

        Args:
            documents (List[str]): List of documents as strings.

        Returns:
            csr_matrix: Sparse matrix of TF-IDF vectors.
        """
        # 创建TF矩阵
        tf_matrix = self._calculate_tf(documents)
        
        # 计算TF-IDF
        tfidf_matrix = tf_matrix.multiply(self.idf_)
        
        return tfidf_matrix

    def fit_transform(self, documents: List[str]) -> csr_matrix:
        """
        Fit the vectorizer to the documents and transform them into TF-IDF vectors.

        Args:
            documents (List[str]): List of documents as strings.

        Returns:
            csr_matrix: Sparse matrix of TF-IDF vectors.
        """
        self.fit(documents)
        return self.transform(documents)

    def _build_vocabulary(self, documents: List[str]) -> None:
        """
        Build the vocabulary from the list of documents.

        Args:
            documents (List[str]): List of documents as strings.
        """
        vocab = set()
        for doc in documents:
            vocab.update(doc.split())
        self.vocabulary_ = {term: idx for idx, term in enumerate(vocab)}
        print(f"Vocabulary built with {len(self.vocabulary_)} terms.")

    def _calculate_idf(self, documents: List[str]) -> None:
        """
        Calculate inverse document frequency (IDF) for each term in the vocabulary.

        Args:
            documents (List[str]): List of documents as strings.
        """
        N = len(documents)
        df = np.zeros(len(self.vocabulary_))
        for doc in documents:
            terms = set(doc.split())
            for term in terms:
                if term in self.vocabulary_:
                    df[self.vocabulary_[term]] += 1
        self.idf_ = np.log((1 + N) / (1 + df)) + 1
        print(f"IDF calculated for {len(self.idf_)} terms.")

    def _calculate_tf(self, documents: List[str]) -> csr_matrix:
        """
        Calculate term frequency (TF) matrix for the list of documents.

        Args:
            documents (List[str]): List of documents as strings.

        Returns:
            csr_matrix: Sparse matrix of term frequencies.
        """
        rows, cols, data = [], [], []
        for row_idx, doc in enumerate(documents):
            term_counts = {}
            for term in doc.split():
                if term in self.vocabulary_:
                    term_idx = self.vocabulary_[term]
                    term_counts[term_idx] = term_counts.get(term_idx, 0) + 1
            length = len(doc.split())
            for term_idx, count in term_counts.items():
                rows.append(row_idx)
                cols.append(term_idx)
                data.append(count / length)
        tf_matrix = csr_matrix((data, (rows, cols)), shape=(len(documents), len(self.vocabulary_)))
        print(f"TF matrix calculated with shape {tf_matrix.shape}.")
        return tf_matrix
    
import numpy as np
from scipy.sparse import csr_matrix
from typing import List, Tuple, Dict

class BM25:
    """
    BM25 Vectorizer using numpy and scipy
    
    Attributes:
        vocabulary_ (Dict[str, int]): A dictionary where keys are terms and values are term indices.
        doc_lengths_ (np.ndarray): Array of document lengths.
        avg_doc_length_ (float): Average document length.
        term_freq_ (csr_matrix): Sparse matrix of term frequencies.
        doc_freq_ (np.ndarray): Array of document frequencies for terms in the vocabulary.
        num_docs_ (int): Number of documents.
        k1 (float): Term frequency saturation parameter.
        b (float): Length normalization parameter.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        """Initialize the vectorizer with default parameters and empty attributes."""
        self.vocabulary_ = {}
        self.doc_lengths_ = None
        self.avg_doc_length_ = None
        self.term_freq_ = None
        self.doc_freq_ = None
        self.num_docs_ = 0
        self.k1 = k1
        self.b = b

    def fit(self, documents: List[str]) -> 'BM25':
        """
        Learn the vocabulary and document frequencies from the list of documents.
        
        Args:
            documents (List[str]): List of documents as strings.
        
        Returns:
            self (BM25): Fitted BM25 object.
        """
        self.num_docs_ = len(documents)
        term_counts = {}
        doc_lengths = []

        for doc in documents:
            words = doc.split()
            doc_lengths.append(len(words))
            unique_words = set(words)
            for word in unique_words:
                if word not in self.vocabulary_:
                    self.vocabulary_[word] = len(self.vocabulary_)
                if word not in term_counts:
                    term_counts[word] = 0
                term_counts[word] += 1

        self.doc_lengths_ = np.array(doc_lengths)
        self.avg_doc_length_ = np.mean(self.doc_lengths_)
        self.doc_freq_ = np.zeros(len(self.vocabulary_))
        
        for term, count in term_counts.items():
            self.doc_freq_[self.vocabulary_[term]] = count

        term_freq_data = []
        term_freq_rows = []
        term_freq_cols = []
        
        for i, doc in enumerate(documents):
            term_freq = {}
            words = doc.split()
            for word in words:
                term_idx = self.vocabulary_[word]
                if term_idx not in term_freq:
                    term_freq[term_idx] = 0
                term_freq[term_idx] += 1
            for term_idx, freq in term_freq.items():
                term_freq_rows.append(i)
                term_freq_cols.append(term_idx)
                term_freq_data.append(freq)
        
        self.term_freq_ = csr_matrix((term_freq_data, (term_freq_rows, term_freq_cols)), shape=(self.num_docs_, len(self.vocabulary_)))
        
        return self

    def transform(self, query: str) -> np.ndarray:
        """
        Transform the query to BM25 scores with respect to the fitted documents.
        
        Args:
            query (str): The query as a string.
        
        Returns:
            scores (np.ndarray): BM25 scores for each document.
        """
        query_terms = query.split()
        scores = np.zeros(self.num_docs_)
        
        for term in query_terms:
            if term in self.vocabulary_:
                term_idx = self.vocabulary_[term]
                idf = np.log((self.num_docs_ - self.doc_freq_[term_idx] + 0.5) / (self.doc_freq_[term_idx] + 0.5) + 1)
                tf = self.term_freq_[:, term_idx].toarray().flatten()
                scores += idf * ((tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * self.doc_lengths_ / self.avg_doc_length_)))
        
        return scores

    def fit_transform(self, documents: List[str], query: str) -> np.ndarray:
        """
        Fit the BM25 model and transform the query in one step.
        
        Args:
            documents (List[str]): List of documents as strings.
            query (str): The query as a string.
        
        Returns:
            scores (np.ndarray): BM25 scores for each document.
        """
        self.fit(documents)
        return self.transform(query)

# Example usage
documents = ["this is a sample document", "this document is another example", "BM25 is a ranking function"]
query = "sample document"

bm25 = BM25()
scores = bm25.fit_transform(documents, query)
print("BM25 Scores:", scores)
import numpy as np
from scipy.sparse import csr_matrix
from typing import List, Tuple, Dict

class BM25:
    """
    BM25 Vectorizer using numpy and scipy
    
    Attributes:
        vocabulary_ (Dict[str, int]): A dictionary where keys are terms and values are term indices.
        doc_lengths_ (np.ndarray): Array of document lengths.
        avg_doc_length_ (float): Average document length.
        term_freq_ (csr_matrix): Sparse matrix of term frequencies.
        doc_freq_ (np.ndarray): Array of document frequencies for terms in the vocabulary.
        num_docs_ (int): Number of documents.
        k1 (float): Term frequency saturation parameter.
        b (float): Length normalization parameter.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        """Initialize the vectorizer with default parameters and empty attributes."""
        self.vocabulary_ = {}
        self.doc_lengths_ = None
        self.avg_doc_length_ = None
        self.term_freq_ = None
        self.doc_freq_ = None
        self.num_docs_ = 0
        self.k1 = k1
        self.b = b

    def fit(self, documents: List[str]) -> 'BM25':
        """
        Learn the vocabulary and document frequencies from the list of documents.
        
        Args:
            documents (List[str]): List of documents as strings.
        
        Returns:
            self (BM25): Fitted BM25 object.
        """
        self.num_docs_ = len(documents)
        term_counts = {}
        doc_lengths = []

        for doc in documents:
            words = doc.split()
            doc_lengths.append(len(words))
            unique_words = set(words)
            for word in unique_words:
                if word not in self.vocabulary_:
                    self.vocabulary_[word] = len(self.vocabulary_)
                if word not in term_counts:
                    term_counts[word] = 0
                term_counts[word] += 1

        self.doc_lengths_ = np.array(doc_lengths)
        self.avg_doc_length_ = np.mean(self.doc_lengths_)
        self.doc_freq_ = np.zeros(len(self.vocabulary_))
        
        for term, count in term_counts.items():
            self.doc_freq_[self.vocabulary_[term]] = count

        term_freq_data = []
        term_freq_rows = []
        term_freq_cols = []
        
        for i, doc in enumerate(documents):
            term_freq = {}
            words = doc.split()
            for word in words:
                term_idx = self.vocabulary_[word]
                if term_idx not in term_freq:
                    term_freq[term_idx] = 0
                term_freq[term_idx] += 1
            for term_idx, freq in term_freq.items():
                term_freq_rows.append(i)
                term_freq_cols.append(term_idx)
                term_freq_data.append(freq)
        
        self.term_freq_ = csr_matrix((term_freq_data, (term_freq_rows, term_freq_cols)), shape=(self.num_docs_, len(self.vocabulary_)))
        
        return self

    def transform(self, query: str) -> np.ndarray:
        """
        Transform the query to BM25 scores with respect to the fitted documents.
        
        Args:
            query (str): The query as a string.
        
        Returns:
            scores (np.ndarray): BM25 scores for each document.
        """
        query_terms = query.split()
        scores = np.zeros(self.num_docs_)
        
        for term in query_terms:
            if term in self.vocabulary_:
                term_idx = self.vocabulary_[term]
                idf = np.log((self.num_docs_ - self.doc_freq_[term_idx] + 0.5) / (self.doc_freq_[term_idx] + 0.5) + 1)
                tf = self.term_freq_[:, term_idx].toarray().flatten()
                scores += idf * ((tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * self.doc_lengths_ / self.avg_doc_length_)))
        
        return scores

    def fit_transform(self, documents: List[str], query: str) -> np.ndarray:
        """
        Fit the BM25 model and transform the query in one step.
        
        Args:
            documents (List[str]): List of documents as strings.
            query (str): The query as a string.
        
        Returns:
            scores (np.ndarray): BM25 scores for each document.
        """
        self.fit(documents)
        return self.transform(query)

# Example usage
documents = ["this is a sample document", "this document is another example", "BM25 is a ranking function"]
query = "sample document"

bm25 = BM25()
scores = bm25.fit_transform(documents, query)
print("BM25 Scores:", scores)


import numpy as np
from scipy.sparse import csr_matrix
from typing import List, Dict, Tuple

class TermProximityScore:
    """
    Term Proximity Score (TPS) using numpy and scipy.
    
    Attributes:
        vocabulary_ (Dict[str, int]): A dictionary where keys are terms and values are term indices.
        doc_term_positions_ (List[Dict[int, List[int]]]): List of dictionaries, each containing term positions for each document.
        num_docs_ (int): Number of documents.
    """

    def __init__(self) -> None:
        """Initialize the TPS with an empty vocabulary and term positions list."""
        self.vocabulary_ = {}
        self.doc_term_positions_ = []
        self.num_docs_ = 0

    def fit(self, documents: List[str]) -> 'TermProximityScore':
        """
        Learn the vocabulary and term positions from the list of documents.
        
        Args:
            documents (List[str]): List of documents as strings.
        
        Returns:
            self (TermProximityScore): Fitted TPS object.
        """
        self.num_docs_ = len(documents)
        
        for doc in documents:
            words = doc.split()
            term_positions = {}
            for pos, word in enumerate(words):
                if word not in self.vocabulary_:
                    self.vocabulary_[word] = len(self.vocabulary_)
                term_idx = self.vocabulary_[word]
                if term_idx not in term_positions:
                    term_positions[term_idx] = []
                term_positions[term_idx].append(pos)
            self.doc_term_positions_.append(term_positions)
        
        return self

    def transform(self, query: str) -> np.ndarray:
        """
        Transform the query to Term Proximity Scores with respect to the fitted documents.
        
        Args:
            query (str): The query as a string.
        
        Returns:
            scores (np.ndarray): Term Proximity Scores for each document.
        """
        query_terms = query.split()
        query_term_indices = [self.vocabulary_[term] for term in query_terms if term in self.vocabulary_]
        scores = np.zeros(self.num_docs_)
        
        for i, term_positions in enumerate(self.doc_term_positions_):
            for j, term_idx in enumerate(query_term_indices):
                if term_idx in term_positions:
                    positions = term_positions[term_idx]
                    for k in range(j + 1, len(query_term_indices)):
                        next_term_idx = query_term_indices[k]
                        if next_term_idx in term_positions:
                            next_positions = term_positions[next_term_idx]
                            for pos in positions:
                                for next_pos in next_positions:
                                    distance = abs(pos - next_pos)
                                    if distance > 0:
                                        scores[i] += 1 / (distance ** 2)
        
        return scores

    def fit_transform(self, documents: List[str], query: str) -> np.ndarray:
        """
        Fit the TPS model and transform the query in one step.
        
        Args:
            documents (List[str]): List of documents as strings.
            query (str): The query as a string.
        
        Returns:
            scores (np.ndarray): Term Proximity Scores for each document.
        """
        self.fit(documents)
        return self.transform(query)

# Example usage
documents = ["this is a sample document", "this document is another example", "term proximity score is useful"]
query = "sample document"

tps = TermProximityScore()
scores = tps.fit_transform(documents, query)
print("Term Proximity Scores:", scores)
