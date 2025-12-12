"""
NLP Practice Problems - Basic to Advanced
Real-world scenarios to build your NLP skills

Instructions:
- Solve each problem step by step
- Use appropriate NLP libraries (nltk, spacy, sklearn, etc.)
- Test your solutions with the provided sample data
- Progress from basic to advanced problems
"""

import nltk
import spacy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import re
from collections import Counter

# Sample datasets for practice
SAMPLE_REVIEWS = [
    "This movie was absolutely fantastic! Great acting and storyline.",
    "Terrible film, waste of time. Poor acting and boring plot.",
    "Amazing cinematography and excellent performances by all actors.",
    "Not worth watching. Very disappointing and poorly made.",
    "One of the best movies I've ever seen. Highly recommend!",
    "Awful movie, couldn't even finish watching it.",
    "Brilliant direction and outstanding script. Must watch!",
    "Complete disaster. Bad acting, worse plot."
]

SAMPLE_EMAILS = [
    "Congratulations! You've won $1,000,000! Click here to claim your prize now!",
    "Meeting scheduled for tomorrow at 2 PM in conference room A.",
    "URGENT: Your account will be suspended unless you verify immediately!",
    "Please find the quarterly report attached for your review.",
    "FREE MONEY! No strings attached! Act now before it's too late!",
    "Reminder: Project deadline is next Friday. Please submit your work.",
    "You are the lucky winner of our lottery! Send your details to claim!",
    "Team lunch is scheduled for Thursday at the new restaurant downtown."
]

SAMPLE_NEWS = [
    "The stock market reached new highs today as technology companies reported strong earnings.",
    "Local football team wins championship after defeating rivals 3-1 in final match.",
    "New medical breakthrough offers hope for cancer patients worldwide.",
    "Political tensions rise as negotiations between countries continue.",
    "Scientists discover new species in deep ocean exploration mission.",
    "Economic indicators suggest potential recession in coming months.",
    "Celebrity couple announces engagement after two years of dating.",
    "Climate change summit brings together world leaders to discuss solutions."
]

print("=" * 60)
print("NLP PRACTICE PROBLEMS - SOLVE THESE STEP BY STEP")
print("=" * 60)

# =============================================================================
# BASIC LEVEL PROBLEMS (1-5)
# =============================================================================

print("\n🟢 BASIC LEVEL PROBLEMS")
print("-" * 30)

print("""
PROBLEM 1: Text Cleaning and Preprocessing
Task: Clean and preprocess the movie reviews above
Requirements:
- Remove punctuation and special characters
- Convert to lowercase
- Remove extra whitespaces
- Tokenize the text

Sample Input: "This movie was absolutely fantastic! Great acting and storyline."
Expected Output: ['this', 'movie', 'was', 'absolutely', 'fantastic', 'great', 'acting', 'and', 'storyline']

def clean_and_tokenize(text):
    # YOUR CODE HERE
    pass

# Test your function
test_review = SAMPLE_REVIEWS[0]
print(f"Original: {test_review}")
print(f"Cleaned: {clean_and_tokenize(test_review)}")
""")

print("""
PROBLEM 2: Word Frequency Analysis
Task: Find the most common words in all movie reviews
Requirements:
- Combine all reviews into one text
- Count word frequencies
- Return top 10 most common words
- Exclude common stop words

def get_word_frequency(reviews, top_n=10):
    # YOUR CODE HERE
    pass

# Test your function
print("Top 10 words:", get_word_frequency(SAMPLE_REVIEWS))
""")

print("""
PROBLEM 3: Sentiment Classification (Rule-based)
Task: Create a simple rule-based sentiment classifier
Requirements:
- Define positive and negative word lists
- Count positive vs negative words in text
- Return 'positive', 'negative', or 'neutral'

positive_words = ['good', 'great', 'excellent', 'amazing', 'fantastic', 'brilliant', 'outstanding']
negative_words = ['bad', 'terrible', 'awful', 'poor', 'disappointing', 'boring', 'disaster']

def rule_based_sentiment(text):
    # YOUR CODE HERE
    pass

# Test with sample reviews
for review in SAMPLE_REVIEWS[:3]:
    sentiment = rule_based_sentiment(review)
    print(f"Review: {review[:50]}...")
    print(f"Sentiment: {sentiment}\n")
""")

print("""
PROBLEM 4: Text Similarity
Task: Calculate similarity between two texts using Jaccard similarity
Requirements:
- Convert texts to word sets
- Calculate Jaccard similarity (intersection/union)
- Return similarity score between 0 and 1

def jaccard_similarity(text1, text2):
    # YOUR CODE HERE
    pass

# Test with sample reviews
text1 = SAMPLE_REVIEWS[0]
text2 = SAMPLE_REVIEWS[2]
similarity = jaccard_similarity(text1, text2)
print(f"Similarity between reviews: {similarity:.3f}")
""")

print("""
PROBLEM 5: Keyword Extraction
Task: Extract important keywords from text using TF-IDF
Requirements:
- Use TfidfVectorizer from sklearn
- Extract top 5 keywords from each review
- Return keywords with their scores

def extract_keywords(texts, top_k=5):
    # YOUR CODE HERE
    pass

# Test with sample reviews
keywords = extract_keywords(SAMPLE_REVIEWS)
for i, kw in enumerate(keywords[:3]):
    print(f"Review {i+1} keywords: {kw}")
""")

# =============================================================================
# INTERMEDIATE LEVEL PROBLEMS (6-10)
# =============================================================================

print("\n🟡 INTERMEDIATE LEVEL PROBLEMS")
print("-" * 35)

print("""
PROBLEM 6: Spam Email Detection
Task: Build a machine learning classifier to detect spam emails
Requirements:
- Use the sample emails above (label them manually first)
- Extract TF-IDF features
- Train a Naive Bayes classifier
- Evaluate performance with accuracy and classification report

# Labels for sample emails (0=ham, 1=spam)
email_labels = [1, 0, 1, 0, 1, 0, 1, 0]  # You may adjust based on your judgment

def build_spam_classifier(emails, labels):
    # YOUR CODE HERE
    # Steps:
    # 1. Split data into train/test
    # 2. Vectorize emails using TF-IDF
    # 3. Train classifier
    # 4. Make predictions
    # 5. Evaluate performance
    pass

# Test your classifier
accuracy, report = build_spam_classifier(SAMPLE_EMAILS, email_labels)
print(f"Accuracy: {accuracy:.3f}")
print("Classification Report:")
print(report)
""")

print("""
PROBLEM 7: Named Entity Recognition (NER)
Task: Extract named entities from news articles
Requirements:
- Use spaCy for NER
- Extract PERSON, ORG, GPE (countries/cities), MONEY entities
- Count frequency of each entity type
- Display results in a structured format

def extract_named_entities(texts):
    # YOUR CODE HERE
    # Use spacy.load('en_core_web_sm')
    pass

# Test with sample news
entities = extract_named_entities(SAMPLE_NEWS)
print("Named Entities Found:")
for ent_type, ents in entities.items():
    print(f"{ent_type}: {ents}")
""")

print("""
PROBLEM 8: Text Summarization (Extractive)
Task: Create an extractive text summarizer
Requirements:
- Calculate sentence importance using TF-IDF
- Rank sentences by importance
- Return top N sentences as summary
- Maintain original sentence order in summary

def extractive_summarizer(text, num_sentences=2):
    # YOUR CODE HERE
    # Steps:
    # 1. Split text into sentences
    # 2. Calculate TF-IDF for sentences
    # 3. Score sentences
    # 4. Select top sentences
    # 5. Return in original order
    pass

# Test with a longer text
long_text = " ".join(SAMPLE_NEWS[:4])
summary = extractive_summarizer(long_text, 2)
print("Original text:")
print(long_text)
print("\nSummary:")
print(summary)
""")

print("""
PROBLEM 9: Topic Modeling with LDA
Task: Discover topics in a collection of documents
Requirements:
- Use sklearn's LatentDirichletAllocation
- Preprocess text (remove stop words, lemmatize)
- Find 3 topics in the news articles
- Display top words for each topic

def topic_modeling(documents, n_topics=3, n_words=5):
    # YOUR CODE HERE
    # Use CountVectorizer and LatentDirichletAllocation
    pass

# Test with sample news
topics = topic_modeling(SAMPLE_NEWS)
for i, topic in enumerate(topics):
    print(f"Topic {i+1}: {topic}")
""")

print("""
PROBLEM 10: Sentiment Analysis with Machine Learning
Task: Build a sentiment classifier using movie reviews
Requirements:
- Create more training data (extend SAMPLE_REVIEWS)
- Use both unigrams and bigrams as features
- Try different classifiers (Naive Bayes, Logistic Regression)
- Compare performance and choose the best model

def advanced_sentiment_classifier(reviews, labels):
    # YOUR CODE HERE
    # Steps:
    # 1. Create TF-IDF features with unigrams and bigrams
    # 2. Train multiple classifiers
    # 3. Compare performance
    # 4. Return best model and its performance
    pass

# You'll need to create labels for SAMPLE_REVIEWS first
# review_labels = [1, 0, 1, 0, 1, 0, 1, 0]  # 1=positive, 0=negative
""")

# =============================================================================
# ADVANCED LEVEL PROBLEMS (11-15)
# =============================================================================

print("\n🔴 ADVANCED LEVEL PROBLEMS")
print("-" * 30)

print("""
PROBLEM 11: Custom Word Embeddings
Task: Create custom word embeddings using Word2Vec
Requirements:
- Train Word2Vec model on your text corpus
- Find similar words for given words
- Visualize embeddings using t-SNE or PCA
- Calculate word analogies (king - man + woman = queen)

def create_word_embeddings(texts):
    # YOUR CODE HERE
    # Use gensim's Word2Vec
    # Steps:
    # 1. Preprocess and tokenize texts
    # 2. Train Word2Vec model
    # 3. Test similarity functions
    # 4. Visualize embeddings
    pass
""")

print("""
PROBLEM 12: Multi-class Text Classification
Task: Classify news articles into categories (sports, business, health, politics)
Requirements:
- Create labeled dataset from SAMPLE_NEWS
- Use advanced feature engineering (TF-IDF + word embeddings)
- Implement cross-validation
- Handle class imbalance if present
- Achieve >80% accuracy

def multiclass_news_classifier(articles, categories):
    # YOUR CODE HERE
    # Advanced techniques:
    # - Feature combination
    # - Hyperparameter tuning
    # - Cross-validation
    # - Class balancing
    pass
""")

print("""
PROBLEM 13: Dependency Parsing and Syntax Analysis
Task: Analyze grammatical structure of sentences
Requirements:
- Use spaCy for dependency parsing
- Extract subject-verb-object relationships
- Identify sentence complexity (simple, compound, complex)
- Find passive voice constructions

def syntax_analyzer(text):
    # YOUR CODE HERE
    # Extract:
    # - Dependencies
    # - POS tags
    # - Sentence structure
    # - Grammatical relationships
    pass
""")

print("""
PROBLEM 14: Real-time Text Processing Pipeline
Task: Build a complete NLP pipeline for processing streaming text
Requirements:
- Create a pipeline class that chains multiple NLP tasks
- Include: cleaning, tokenization, NER, sentiment, keywords
- Make it efficient for real-time processing
- Add caching for repeated texts
- Include error handling and logging

class NLPPipeline:
    def __init__(self):
        # YOUR CODE HERE
        pass
    
    def process(self, text):
        # YOUR CODE HERE
        # Return comprehensive analysis
        pass
    
    def batch_process(self, texts):
        # YOUR CODE HERE
        pass
""")

print("""
PROBLEM 15: Custom Language Model Evaluation
Task: Create a system to evaluate text generation quality
Requirements:
- Implement BLEU score calculation
- Calculate perplexity for language models
- Measure semantic similarity using embeddings
- Create a comprehensive evaluation framework
- Test with generated vs reference texts

def evaluate_text_generation(generated_texts, reference_texts):
    # YOUR CODE HERE
    # Metrics to implement:
    # - BLEU score
    # - ROUGE score
    # - Semantic similarity
    # - Fluency assessment
    pass
""")

print("\n" + "=" * 60)
print("BONUS CHALLENGES")
print("=" * 60)

print("""
🏆 BONUS CHALLENGE 1: Build a Chatbot
Create a rule-based chatbot that can:
- Understand user intents (greeting, question, complaint, etc.)
- Extract entities from user input
- Generate appropriate responses
- Maintain conversation context

🏆 BONUS CHALLENGE 2: Fake News Detection
Build a system to detect potentially fake news:
- Analyze writing style and patterns
- Check for emotional language and bias
- Verify factual claims (if possible)
- Combine multiple signals for final prediction

🏆 BONUS CHALLENGE 3: Multilingual NLP
Extend any of the above problems to work with multiple languages:
- Use multilingual models
- Handle different scripts and encodings
- Compare performance across languages
""")

print("\n" + "=" * 60)
print("GETTING STARTED TIPS")
print("=" * 60)
print("""
1. Start with Problem 1 and work your way up
2. Install required libraries: pip install nltk spacy scikit-learn gensim
3. Download spaCy model: python -m spacy download en_core_web_sm
4. Test each function thoroughly before moving to the next
5. Use real datasets for better practice (IMDB reviews, news datasets, etc.)
6. Document your solutions and create a portfolio
7. Experiment with different approaches for each problem

Good luck with your NLP journey! 🚀
""")