# Transformers Cheatsheet

## 1. Installing Transformers
- pip install transformers  # Install Transformers library

## 2. Importing Libraries
- from transformers import pipeline, AutoTokenizer, AutoModel  # Import essential classes

## 3. Loading Pre-trained Models
- tokenizer = AutoTokenizer.from_pretrained('model_name')  # Load tokenizer
- model = AutoModel.from_pretrained('model_name')  # Load model

## 4. Using Pipelines
- classifier = pipeline('sentiment-analysis')  # Load sentiment analysis pipeline
- results = classifier('I love using Transformers!')  # Analyze sentiment

## 5. Tokenization
- tokens = tokenizer('Hello, world!', return_tensors='pt')  # Tokenize input text
- input_ids = tokens['input_ids']  # Get input IDs
- attention_mask = tokens['attention_mask']  # Get attention mask

## 6. Making Predictions
- outputs = model(input_ids, attention_mask=attention_mask)  # Get model outputs
- logits = outputs.logits  # Extract logits

## 7. Fine-tuning Models
- from transformers import Trainer, TrainingArguments  # Import Trainer
- training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
)  # Define training arguments

## 8. Using Datasets
- from datasets import load_dataset  # Import datasets library
- dataset = load_dataset('dataset_name')  # Load dataset

## 9. Data Preprocessing
- def preprocess_function(examples):
  - return tokenizer(examples['text'], truncation=True)  # Tokenize examples
- tokenized_datasets = dataset.map(preprocess_function, batched=True)  # Preprocess dataset

## 10. Evaluating Models
- from transformers import pipeline  # Import pipeline
- evaluator = pipeline('text-classification', model='model_name')  # Load evaluation pipeline
- results = evaluator(['Example text 1', 'Example text 2'])  # Evaluate texts

## 11. Using Transformers for Translation
- translator = pipeline('translation_en_to_fr')  # Load translation pipeline
- translation = translator('Hello, how are you?')  # Translate text

## 12. Saving and Loading Models
- model.save_pretrained('model_directory')  # Save model
- tokenizer.save_pretrained('model_directory')  # Save tokenizer
- model = AutoModel.from_pretrained('model_directory')  # Load model
- tokenizer = AutoTokenizer.from_pretrained('model_directory')  # Load tokenizer

## 13. Multi-Modal Tasks
- from transformers import VisionEncoderDecoderModel  # Import vision encoder-decoder
- model = VisionEncoderDecoderModel.from_pretrained('model_name')  # Load multi-modal model
