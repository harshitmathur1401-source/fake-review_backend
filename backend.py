from typing import List, Dict, Tuple
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize
from collections import Counter

nltk.download('punkt')

def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()  # normalize spaces
    return text

def is_fake_review(text: str) -> bool:
    """
    Simple heuristics based fake detection:
    - Excessive repetition of exclamation/question marks or all caps words
    - Very short reviews with generic positive or negative words
    - Overuse of promotional or spammy phrases
    - Low lexical diversity
    """
    text = preprocess(text)

    # Too short
    if len(text) < 20:
        return True

    # Count exclamation/question marks
    if text.count('!') > 3 or text.count('?') > 3:
        return True

    # Check all caps words count
    all_caps_words = re.findall(r'\b[A-Z]{2,}\b', text)
    if len(all_caps_words) > 2:
        return True
    
    # Check for spammy phrases (can be customized)
    spammy_phrases = ['best product ever', 'highly recommended', 'five stars', 'buy now', 'must buy']
    for phrase in spammy_phrases:
        if phrase in text:
            return True

    # Lexical diversity
    tokens = text.split()
    if len(tokens) == 0:
        return True
    unique_tokens = set(tokens)
    lexical_diversity = len(unique_tokens) / len(tokens)
    if lexical_diversity < 0.3:
        return True

    return False

def detect_paraphrases(reviews: List[str], threshold: float = 0.75) -> List[Tuple[int, int]]:
    """
    Detect paraphrased reviews by comparing pairwise cosine similarity of TF-IDF vectors.
    Returns list of pairs of indices (i, j) where i < j and reviews are paraphrases.
    """
    cleaned_reviews = [preprocess(r) for r in reviews]
    vectorizer = TfidfVectorizer().fit_transform(cleaned_reviews)
    vectors = vectorizer.toarray()
    similarity_matrix = cosine_similarity(vectors)

    paraphrases = []
    n = len(reviews)
    for i in range(n):
        for j in range(i + 1, n):
            if similarity_matrix[i, j] >= threshold:
                paraphrases.append((i, j))
    return paraphrases

def summarize_reviews(reviews: List[str], max_sentences: int = 3) -> str:
    """
    Extractive summarization using sentence scoring based on word frequency:
    - Tokenize sentences
    - Score each sentence by summing frequency of its words
    - Return top max_sentences sentences joined
    """
    combined_text = ' '.join(reviews)
    sentences = sent_tokenize(combined_text)
    if len(sentences) <= max_sentences:
        return combined_text  # short enough

    # Word frequency
    words = re.findall(r'\w+', combined_text.lower())
    freq = Counter(words)

    sentence_scores = {}
    for sent in sentences:
        sent_words = re.findall(r'\w+', sent.lower())
        score = sum(freq[word] for word in sent_words if word in freq)
        sentence_scores[sent] = score

    # Sort sentences by score descending
    sorted_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)

    # Pick top max_sentences
    summary = ' '.join(sorted_sentences[:max_sentences])
    return summary

def analyze_reviews(reviews: List[str]) -> Dict:
    """
    Main analysis function that:
    - Identifies fake reviews and real reviews
    - Detects paraphrased reviews
    - Summarizes real reviews
    Returns dictionary with categorized outputs.
    """
    results = {
        'fake_reviews': [],
        'real_reviews': [],
        'paraphrased_pairs': [],
        'summary': '',
    }

    # Detect fake
    fake_flags = [is_fake_review(r) for r in reviews]
    for idx, is_fake in enumerate(fake_flags):
        if is_fake:
            results['fake_reviews'].append(reviews[idx])
        else:
            results['real_reviews'].append(reviews[idx])

    # Detect paraphrases among real reviews only
    paraphrases = detect_paraphrases(results['real_reviews']) if results['real_reviews'] else []
    results['paraphrased_pairs'] = paraphrases

    # Summarize real reviews
    if results['real_reviews']:
        results['summary'] = summarize_reviews(results['real_reviews'])
    else:
        results['summary'] = 'No genuine reviews to summarize.'

    return results

# Example usage
if __name__ == '__main__':
    example_reviews = [
        "This product is amazing!!! Highly recommended!!!",
        "I loved the product. It works well and is exactly as described.",
        "Must buy! Best product ever!!!",
        "The product is okay, nothing exceptional.",
        "Loved it! Works well, exactly what I expected.",
        "This product is amazing!!! Highly recommended!!!",  # Paraphrased duplicate of 0
        "Worst product. Totally disappointed."
    ]

    output = analyze_reviews(example_reviews)
    print("Fake Reviews:")
    for fr in output['fake_reviews']:
        print("-", fr)
    print("\nReal Reviews:")
    for rr in output['real_reviews']:
        print("-", rr)
    print("\nParaphrased Review Pairs (indices):", output['paraphrased_pairs'])
    print("\nSummary of Real Reviews:")
    print(output['summary'])
