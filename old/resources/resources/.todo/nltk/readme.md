# NLTK Cheatsheet

## 1. Installing NLTK
- pip install nltk  # Install NLTK

## 2. Importing NLTK
- import nltk  # Import NLTK
- nltk.download('all')  # Download all NLTK resources (optional)

## 3. Tokenization
- from nltk.tokenize import word_tokenize, sent_tokenize  # Import tokenizers
- words = word_tokenize('Hello, world!')  # Tokenize into words
- sentences = sent_tokenize('Hello. How are you?')  # Tokenize into sentences

## 4. Stop Words
- from nltk.corpus import stopwords  # Import stopwords
- stop_words = set(stopwords.words('english'))  # Get English stop words
- filtered_words = [word for word in words if word.lower() not in stop_words]  # Remove stop words

## 5. Stemming
- from nltk.stem import PorterStemmer  # Import stemmer
- ps = PorterStemmer()  # Create stemmer object
- stemmed_word = ps.stem('running')  # Stem a word

## 6. Lemmatization
- from nltk.stem import WordNetLemmatizer  # Import lemmatizer
- lemmatizer = WordNetLemmatizer()  # Create lemmatizer object
- lemmatized_word = lemmatizer.lemmatize('running')  # Lemmatize a word

## 7. Part-of-Speech Tagging
- from nltk import pos_tag  # Import POS tagger
- pos_tags = pos_tag(words)  # Get POS tags for words

## 8. Named Entity Recognition
- from nltk import ne_chunk  # Import named entity chunker
- named_entities = ne_chunk(pos_tags)  # Get named entities

## 9. Frequency Distribution
- from nltk.probability import FreqDist  # Import frequency distribution
- freq_dist = FreqDist(words)  # Get frequency distribution
- freq_dist.plot(30, cumulative=False)  # Plot the 30 most common words

## 10. Concordance
- text = nltk.Text(words)  # Create NLTK text object
- text.concordance('Hello')  # Find occurrences of a word

## 11. N-grams
- from nltk import ngrams  # Import n-gram function
- bi_grams = list(ngrams(words, 2))  # Create bigrams

## 12. Collocations
- from nltk.collocations import BigramCollocationFinder  # Import collocation finder
- finder = BigramCollocationFinder.from_words(words)  # Find bigram collocations
- finder.nbest(nltk.metrics.BigramAssocMeasures().pmi, 10)  # Get top 10 collocations

## 13. Text Classification
- from nltk.classify import NaiveBayesClassifier  # Import classifier
- classifier = NaiveBayesClassifier.train(training_data)  # Train classifier

## 14. Saving and Loading Models
- import pickle  # Import pickle for serialization
- with open('model.pkl', 'wb') as f:  # Save model
  - pickle.dump(classifier, f)
- with open('model.pkl', 'rb') as f:  # Load model
  - classifier = pickle.load(f)
