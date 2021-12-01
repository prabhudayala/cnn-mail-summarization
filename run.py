from tqdm import tqdm
import os
from transformers import pipeline
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BartTokenizer, BartForConditionalGeneration, T5Tokenizer, T5ForConditionalGeneration
import abstractive
import format_data
import extractive

def format_and_process_data():
    format_data.format_data_from_dataset()
    
def train_abstractive():
    abstractive.main_abstractive()

#dummy method
def hello():
    return "hi"
output_dir = './abstractive/'

global model
model=None
global tokenizer
tokenizer=None
global summarizer
summarizer=None

if os.path.isfile('./abstractive/pytorch_model.bin'):
    model = BartForConditionalGeneration.from_pretrained(output_dir)
    tokenizer = BartTokenizer.from_pretrained(output_dir)
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, framework="pt")

def predict_abstactive_summary(file_name='sample.txt'):
    if os.path.isfile('./abstractive/pytorch_model.bin'):
        with open(file_name) as f:
            text = f.read()
        answer = summarizer(text)[0]['summary_text']
        return answer
    else:
        print('Training model....')
        abstractive.main_abstractive()

def train_extractive():
    extractive.train_extractive_from_dataset()
def predict_extractive_summary(file_name='sample.txt'):
    return extractive.generate_summary(file_name)