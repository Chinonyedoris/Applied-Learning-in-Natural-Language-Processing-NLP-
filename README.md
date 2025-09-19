# Applied-Learning-in-Natural-Language-Processing-NLP-
Applied Learning Assignments that deepened my understanding of NLP concepts and hands-on techniques.

🚀 Applied Learning in Natural Language Processing (NLP)

Over the past weeks, I’ve been working on a series of Applied Learning Assignments that deepened my understanding of NLP concepts and hands-on techniques.

## Applied Learning Assignments 1 

🔹 Defined NLP in my own words and explored its real-world applications.
🔹 Identified key challenges that make NLP complex.
🔹 Used regex to extract patterns like emails and words ending with -ing.
🔹 Wrote a Python script to clean text:

import re, string text = "NLP makes AI smarter! But, sometimes, it’s challenging..." cleaned = re.sub(r'[^\w\s]', '', text).lower().split() print(cleaned) # ['nlp', 'makes', 'ai', 'smarter', 'but', 'sometimes', 'its', 'challenging'] 

## Applied Learning Assignments 2 

🔹 Applied text cleaning on noisy text (emojis, URLs, numbers).
🔹 Performed tokenization at word and sentence level using NLTK.
🔹 Practiced stemming and lemmatization:

from nltk.stem import PorterStemmer from spacy.lang.en import English import spacy ps = PorterStemmer() words = ["running", "flies", "studies", "easily", "studying", "better"] print([ps.stem(w) for w in words]) # ['run', 'fli', 'studi', 'easili', 'studi', 'better'] nlp = spacy.load("en_core_web_sm") print([token.lemma_ for token in nlp(" ".join(words))]) # ['run', 'fly', 'study', 'easily', 'study', 'well'] 

## Applied Learning Assignments 3 

🔹 Built a vocabulary and generated one-hot encoded vectors.
🔹 Implemented Bag of Words and TF-IDF using scikit-learn:

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer docs = ["The quick brown fox jumps over the lazy dog.", "The dog sleeps in the kernel"] vectorizer = CountVectorizer() print(vectorizer.fit_transform(docs).toarray()) # Bag of Words matrix 

🔹 Trained a Word2Vec model with gensim and explored embeddings:

from gensim.models import Word2Vec sentences = [["the","cat","meows"],["the","dog","barks"],["the","bird","sings"]] model = Word2Vec(sentences, min_count=1, vector_size=50) print(model.wv['dog']) # Word embedding for 'dog' 

🔹 Explored pretrained GloVe embeddings:

from gensim.downloader import load glove = load("glove-wiki-gigaword-50") print(glove.most_similar("king", topn=5)) # [('prince', ...), ('queen', ...), ('monarch', ...)] 

💡 This journey has been both theoretical and practical, exposing me to foundational NLP techniques widely applied in chatbots, search engines, recommendation systems, and sentiment analysis.

Excited to keep building! 🚀✨

#NLP #MachineLearning #ArtificialIntelligence #DataScience #Python #AppliedLearning

🚀 Applied Learning in Natural Language Processing (NLP)
Over the past weeks, I’ve been working on a series of Applied Learning Assignments that deepened my understanding of NLP concepts and hands-on techniques.
Applied Learning Assignments 1
🔹 Defined NLP in my own words and explored its real-world applications.
🔹 Identified key challenges that make NLP complex.
🔹 Used regex to extract patterns like emails and words ending with -ing.
🔹 Wrote a Python script to clean text by removing punctuation, converting to lowercase, and splitting into words.
Applied Learning Assignments 2
🔹 Carried out text cleaning on noisy data (emojis, URLs, numbers).
🔹 Applied tokenization at both word and sentence level using NLTK.
🔹 Practiced stemming with Porter Stemmer and lemmatization with spaCy.
Applied Learning Assignments 3
🔹 Built a vocabulary and generated one-hot encoded vectors.
🔹 Implemented Bag of Words and TF-IDF representations using CountVectorizer and TfidfVectorizer.
🔹 Trained a Word2Vec model with gensim and retrieved embeddings for words like dog.
🔹 Explored pretrained GloVe embeddings to find semantic relationships (e.g., most similar words to king).
💡 This journey has been both theoretical and practical, giving me exposure to foundational NLP techniques widely applied in search engines, chatbots, sentiment analysis, and recommendation systems.
Grateful for the opportunity to learn, practice, and share! 🙏
#NLP #MachineLearning #ArtificialIntelligence #DataScience #Python #AppliedLearning
