import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import random
import pandas as pd
from collections import Counter
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from collections import Counter, defaultdict
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import re
from sklearn.utils import shuffle
import os
from tqdm import tqdm, trange
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from transformers import AutoTokenizer
from torch.optim import Adam
from transformers  import BertTokenizer, BertConfig
from transformers  import AdamW, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler
import wandb

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def corrupt_dataset(args, data, num_labels):
    total_number = len(data)
    new_data = data.clone()
    random_indices = torch.rand(int(args.noise_ratio * total_number)) * total_number
    random_indices = random_indices.int()
    new_labels = torch.rand(int(len(random_indices))) * num_labels
    new_labels = new_labels.int()
    for i in range(len(random_indices)):
        temp_label = data[random_indices[i]]
        while temp_label == new_labels[i]:
            new_labels[i] = (torch.rand(1)[0] * num_labels).int()
        new_data[random_indices[i]] = new_labels[i]
    return new_data 

def corrupt_dataset_asymmetric(args, data, num_labels):
    total_number = len(data)
    new_data = data.clone()
    random_indices = torch.rand(int(args.noise_ratio * total_number)) * total_number
    random_indices = random_indices.int()
    for i in range(len(random_indices)):
        temp_label = data[random_indices[i]]
        new_data[random_indices[i]] = (new_data[random_indices[i]] + 1) % num_labels
    return new_data 

def load_dataset(args, dataset):

    if dataset == 'sst':
        df_train = pd.read_csv("/localscratch/yzhuang43/datasets/SST-2/train.tsv", delimiter='\t', header=0)
        
        df_val = pd.read_csv("/localscratch/yzhuang43/datasets/SST-2/dev.tsv", delimiter='\t', header=0)
        
        df_test = pd.read_csv("/localscratch/yzhuang43/datasets/SST-2/sst-test.tsv", delimiter='\t', header=None, names=['sentence', 'label'])

        train_sentences = df_train.sentence.values
        val_sentences = df_val.sentence.values
        test_sentences = df_test.sentence.values
        train_labels = df_train.label.values
        val_labels = df_val.label.values
        test_labels = df_test.label.values   
    

    if dataset == '20news':
        
        VALIDATION_SPLIT = 0.8
        newsgroups_train  = fetch_20newsgroups(data_home=args.path, subset='train',  shuffle=True, random_state=0)
        print(newsgroups_train.target_names)
        print(len(newsgroups_train.data))

        newsgroups_test  = fetch_20newsgroups(data_home=args.path, subset='test',  shuffle=False)

        print(len(newsgroups_test.data))

        train_len = int(VALIDATION_SPLIT * len(newsgroups_train.data))

        train_sentences = newsgroups_train.data[:train_len]
        val_sentences = newsgroups_train.data[train_len:]
        test_sentences = newsgroups_test.data
        train_labels = newsgroups_train.target[:train_len]
        val_labels = newsgroups_train.target[train_len:]
        test_labels = newsgroups_test.target 
    
    return train_sentences, val_sentences, test_sentences, train_labels, val_labels, test_labels

def read_data(args, num_labels):
    # load dataset
    
    train_sentences, val_sentences, test_sentences, train_labels, val_labels, test_labels = load_dataset(args, args.dataset)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    train_input_ids = []
    val_input_ids = []
    test_input_ids = []
    
    if args.dataset == '20news':
        MAX_LEN = 150
    else:
        MAX_LEN = 256

    for sent in train_sentences:
    # `encode` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
        encoded_sent = tokenizer.encode(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            # This function also supports truncation and conversion
                            # to pytorch tensors, but we need to do padding, so we
                            # can't use these features :( .
                            max_length = MAX_LEN,          # Truncate all sentences.
                            #return_tensors = 'pt',     # Return pytorch tensors.
                    )
    # Add the encoded sentence to the list.
        train_input_ids.append(encoded_sent)
        

    for sent in val_sentences:
        encoded_sent = tokenizer.encode(
                            sent,                    
                            add_special_tokens = True, 
                            max_length = MAX_LEN,         
                    )
        val_input_ids.append(encoded_sent)

    for sent in test_sentences:
        encoded_sent = tokenizer.encode(
                            sent,                     
                            add_special_tokens = True, 
                            max_length = MAX_LEN,        
                    )
        test_input_ids.append(encoded_sent)

    # Pad our input tokens
    train_input_ids = pad_sequences(train_input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    val_input_ids = pad_sequences(val_input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    test_input_ids = pad_sequences(test_input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    # Create attention masks
    train_attention_masks = []
    val_attention_masks = []
    test_attention_masks = []

    # Create a mask of 1s for each token followed by 0s for padding
    for seq in train_input_ids:
        seq_mask = [float(i>0) for i in seq]
        train_attention_masks.append(seq_mask)
    for seq in val_input_ids:
        seq_mask = [float(i>0) for i in seq]
        val_attention_masks.append(seq_mask)
    for seq in test_input_ids:
        seq_mask = [float(i>0) for i in seq]
        test_attention_masks.append(seq_mask)

    # Convert all of our data into torch tensors, the required datatype for our model

    train_inputs = torch.tensor(train_input_ids)
    validation_inputs = torch.tensor(val_input_ids)
    train_labels = torch.tensor(train_labels)
    validation_labels = torch.tensor(val_labels)
    train_masks = torch.tensor(train_attention_masks)
    validation_masks = torch.tensor(val_attention_masks)
    test_inputs = torch.tensor(test_input_ids)
    test_labels = torch.tensor(test_labels)
    test_masks = torch.tensor(test_attention_masks)

    if args.noise_type == 's':
        noisy_train_labels = corrupt_dataset(args, train_labels, num_labels)
        noisy_validation_labels = corrupt_dataset(args, validation_labels, num_labels)
    else:
        noisy_train_labels = corrupt_dataset_asymmetric(args, train_labels, num_labels)
        noisy_validation_labels = corrupt_dataset_asymmetric(args, validation_labels, num_labels)
    
    # Create an iterator of our data with torch DataLoader. 
    return train_inputs, train_masks, train_labels, noisy_train_labels, validation_inputs, validation_masks, validation_labels, noisy_validation_labels, test_inputs, test_masks, test_labels

    

def create_dataset(args, inputs, masks, labels, batch_size, noisy_labels=None):
    if noisy_labels != None:
        data = TensorDataset(inputs, masks, labels, noisy_labels)
    else:
        data = TensorDataset(inputs, masks, labels)
    
    sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    return sampler, dataloader