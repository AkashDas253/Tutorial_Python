# spaCy Cheatsheet

## 1. Installing spaCy
- pip install spacy  # Install spaCy
- python -m spacy download en_core_web_sm  # Download English model

## 2. Importing spaCy
- import spacy  # Import spaCy

## 3. Loading a Model
- nlp = spacy.load('en_core_web_sm')  # Load English model

## 4. Processing Text
- doc = nlp('Hello, world!')  # Process text

## 5. Tokenization
- tokens = [token.text for token in doc]  # Get tokens

## 6. Part-of-Speech Tagging
- pos_tags = [(token.text, token.pos_) for token in doc]  # Get POS tags

## 7. Named Entity Recognition
- entities = [(ent.text, ent.label_) for ent in doc.ents]  # Get named entities

## 8. Lemmatization
- lemmas = [token.lemma_ for token in doc]  # Get lemmas

## 9. Dependency Parsing
- for token in doc:
  - print(token.text, token.dep_, token.head.text)  # Get dependency relations

## 10. Similarity
- doc1 = nlp('dog')
- doc2 = nlp('cat')
- similarity = doc1.similarity(doc2)  # Compute similarity

## 11. Text Classification
- from spacy.pipeline.textcat import Config, Config, Config  # Import text classifier
- text_cat = nlp.add_pipe('textcat')  # Add text classification component

## 12. Custom Pipeline Components
- def custom_component(doc):
  - # Process doc
  - return doc
- nlp.add_pipe(custom_component, last=True)  # Add custom component to pipeline

## 13. Visualizing Entities
- from spacy import displacy  # Import visualizer
- displacy.render(doc, style='ent')  # Visualize entities in the text

## 14. Exporting Models
- nlp.to_disk('model_directory')  # Save model to disk
- nlp2 = spacy.load('model_directory')  # Load model from disk

## 15. Using Custom Vocab
- vocab = nlp.vocab  # Access vocab
- word = vocab['word']  # Access specific word in vocab
