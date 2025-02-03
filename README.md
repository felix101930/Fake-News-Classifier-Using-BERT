# Fake News Classification using Fine-Tuned BERT Models

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A natural language processing (NLP) project leveraging fine-tuned BERT models to classify news articles as fake or real. Built with the Hugging Face Transformers library.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Approach](#approach)
- [Usage](#usage)
- [Acknowledgments](#acknowledgments)

## Overview
This project fine-tunes pre-trained BERT models to detect fake news articles using the **WELFake_Dataset** from Kaggle. The implementation focuses on efficiency by:
- Using a 10% balanced subset of the original dataset
- Leveraging **DistilBERT-base-uncased** for faster training
- Implementing layer freezing for model optimization
- Providing an inference pipeline for real-world use
  
## Dataset  

The **WELFake_Dataset** is preprocessed as follows:  

- **Balanced Split**: 50% fake news, 50% real news  
- **Stratified Sampling**: 10% random sample from the original dataset  

You can access the dataset on Kaggle: [WELFake Dataset](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)  



## Approach  

### Model Selection  
- Evaluated **BERT-base**, **RoBERTa**, and **DistilBERT** variants  

### Fine-Tuning  
- Trained for **2 epochs** with a **batch size of 32**  
- Used **Hugging Face Trainer API** for training and evaluation  
- **Accuracy** as the primary evaluation metric

## Usage  

### Inference Pipeline  

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="path/to/fine-tuned-model",
    tokenizer="distilbert-base-uncased"
)

sample_text = "Breaking: NASA announces discovery of alien life on Mars"
results = classifier(sample_text)
```
## Acknowledgments  
- Implementation from **Mehdi Rezvandehy**    
## TODO  

- [ ] Visualize results comparing different models  
- [ ] Use the `evaluate` function to sample the rest of the **WELFake_Dataset** for further evaluation
