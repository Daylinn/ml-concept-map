import re
from typing import List, Dict, Tuple
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# Create a class for Concept Extractor
class ConceptExtractor: 
    def __init__(self):
        # Initialize NLTK resources properly
        required_resources = ['punkt', 'stopwords', 'punkt_tab']
        for resource in required_resources:
            try:
                if resource == 'punkt':
                    nltk.data.find('tokenizers/punkt')
                elif resource == 'stopwords':
                    nltk.data.find('corpora/stopwords')
                elif resource == 'punkt_tab':
                    nltk.data.find('tokenizers/punkt_tab')
            except LookupError:
                print(f"Downloading {resource}...")
                nltk.download(resource, quiet=True)

        self.stop_words = set(stopwords.words('english'))

        self.tfidf = TfidfVectorizer(
            ngram_range=(1, 2), 
            stop_words='english',
            max_features=1000
        )

    # Here we are cleaning the text by cleaning the noise 
    def clean_text(self, text: str) -> str:
        # convert letters to lowercase, remove special characters, & extra white space
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    # Extract the key concepts using the TF-IDF algorithm 
    def extract_concepts(self, texts: List[str], top_n: int = 50) -> List[str]:
        # Clean texts
        cleaned_texts = [self.clean_text(text) for text in texts]
        
        # Fit and transform using the initialized TF-IDF vectorizer
        tfidf_matrix = self.tfidf.fit_transform(cleaned_texts)
        
        # Get feature names (terms)
        feature_names = self.tfidf.get_feature_names_out()
        
        # Calculate average scores
        avg_scores = tfidf_matrix.mean(axis=0).A1
        
        # Get top terms
        top_indices = avg_scores.argsort()[::-1][:top_n]
        top_concepts = [feature_names[i] for i in top_indices]

        return top_concepts
    
    # Extract frequently occurring phrases
    def extract_frequent_phrases(self, texts: List[str], top_n: int = 50) -> List[str]:
        # Clean and tokenize text into sentences
        all_sentences = []
        for text in texts:
            # Apply the same cleaning used in extract_concepts
            cleaned_text = self.clean_text(text)
            try:
                sentences = sent_tokenize(cleaned_text)
            except LookupError:
                # Fallback to simple sentence splitting
                sentences = [s.strip() for s in cleaned_text.split('.') if s.strip()]
            all_sentences.extend(sentences)
        
        # Find common phrases using bigrams
        ngrams = []
        for sentence in all_sentences:
            words = [w for w in word_tokenize(sentence) 
                    if w not in self.stop_words and w.isalpha()]
            
            # Extract bigrams (pairs of words)
            for i in range(len(words) - 1):
                ngrams.append(f"{words[i]} {words[i+1]}")
        
        # Count phrase frequencies
        phrase_counts = Counter(ngrams)
        
        # Get most common phrases
        top_phrases = [phrase for phrase, _ in phrase_counts.most_common(top_n)]
        
        return top_phrases
    
    # Combine different extraction methods
    def extract_all_concepts(self, texts: List[str], top_n: int = 50) -> List[str]:
        # Extract using different methods
        tfidf_concepts = self.extract_concepts(texts, top_n=top_n)
        phrases = self.extract_frequent_phrases(texts, top_n=top_n//2)
        
        # Combine results (removing duplicates)
        combined = []
        seen = set()
        
        # Add TF-IDF concepts first
        for concept in tfidf_concepts:
            if concept.lower() not in seen:
                combined.append(concept)
                seen.add(concept.lower())
        
        # Then add phrases not already included
        for phrase in phrases:
            if phrase.lower() not in seen and len(phrase.split()) > 1:
                combined.append(phrase)
                seen.add(phrase.lower())
                
                # Stop once we reach the desired number
                if len(combined) >= top_n:
                    break
        
        return combined[:top_n]


