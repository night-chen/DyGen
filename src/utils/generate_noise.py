import json
from src.utils.utils2 import *
import numpy as np
import torch
import random
import pandas as pd
from scipy import stats
from math import inf
from sklearn.datasets import fetch_20newsgroups
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from transformers  import BertTokenizer

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def corrupt_dataset_SN(args, data, num_labels):
    total_number = len(data)
    new_data = data.detach().clone()
    noise_ratio = args.noise_ratio * num_labels / (num_labels - 1)
    for i in range(len(new_data)):
        if random.random() > noise_ratio:
            continue
        else:
            new_data[i] = torch.randint(low=0, high=num_labels, size=(1, ))
    return new_data 

def corrupt_dataset_ASN(args, data, num_labels):
    total_number = len(data)
    new_data = data.detach().clone()
    for i in range(len(new_data)):
        if random.random() > args.noise_ratio:
            continue
        else:
            new_data[i] = (new_data[i] + 1) % num_labels
    return new_data 

def corrupt_dataset_IDN(args, inputs, labels, num_labels):
    flip_distribution = stats.truncnorm((0-args.noise_ratio)/0.1, (1-args.noise_ratio)/0.1, loc=args.noise_ratio, scale=0.1)
    flip_rate = flip_distribution.rvs(len(labels))
    W = torch.randn(num_labels, inputs.shape[-1], num_labels).float()
    new_label = labels.detach().clone()
    for i in range(len(new_label)):
        p = inputs[i].float().view(1,-1).mm(W[labels[i].long()].squeeze(0)).squeeze(0)
        p[labels[i]] = -inf
        p = flip_rate[i] * torch.softmax(p, dim=0)
        p[labels[i]] += 1 - flip_rate[i]
        new_label[i] = torch.multinomial(p,1)
    return new_label

def load_dataset(args, dataset):

    if dataset == '20news':
        
        VALIDATION_SPLIT = 0.8
        newsgroups_train  = fetch_20newsgroups(data_home='/localscratch/yzhuang43/datasets/20news', subset='train',  shuffle=True, random_state=args.seed)
        newsgroups_train  = fetch_20newsgroups(data_home=args.path, subset='train',  shuffle=True, random_state=args.seed)
        print(newsgroups_train.target_names)
        print(len(newsgroups_train.data))

        newsgroups_test  = fetch_20newsgroups(data_home='/localscratch/yzhuang43/datasets/20news', subset='test',  shuffle=False)

        print(len(newsgroups_test.data))

        train_len = int(VALIDATION_SPLIT * len(newsgroups_train.data))

        train_sentences = newsgroups_train.data[:train_len]
        val_sentences = newsgroups_train.data[train_len:]
        test_sentences = newsgroups_test.data
        train_labels = newsgroups_train.target[:train_len]
        val_labels = newsgroups_train.target[train_len:]
        test_labels = newsgroups_test.target 
        return train_sentences, val_sentences, test_sentences, train_labels, val_labels, test_labels
    elif dataset == 'agnews':
        VALIDATION_SPLIT = 0.8
        labels_in_domain = [1, 2]

        train_df = pd.read_csv('/localscratch/yzhuang43/data-valuation/text_classification/data/agnews/train.csv', header=None)
        train_df.rename(columns={0: 'label',1: 'title', 2:'sentence'}, inplace=True)
        print(train_df.dtypes)
        train_in_df_sentence = []
        train_in_df_label = []
        counts = [0, 0, 0, 0]
        for i in range(len(train_df.sentence.values)):
            sentence_temp = ''.join(str(train_df.sentence.values[i]))
            if counts[train_df.label.values[i]-1] < 10000:
                train_in_df_sentence.append(sentence_temp)
                train_in_df_label.append(train_df.label.values[i]-1)
                counts[train_df.label.values[i]-1] += 1

        test_df = pd.read_csv('/localscratch/yzhuang43/data-valuation/text_classification/data/agnews/test.csv', header=None)
        test_df.rename(columns={0: 'label',1: 'title', 2:'sentence'}, inplace=True)
        test_in_df_sentence = []
        test_in_df_label = []
        for i in range(len(test_df.sentence.values)):
            test_in_df_sentence.append(str(test_df.sentence.values[i]))
            test_in_df_label.append(test_df.label.values[i]-1)

        train_len = int(VALIDATION_SPLIT * len(train_in_df_sentence))

        train_sentences = train_in_df_sentence[:train_len]
        val_sentences = train_in_df_sentence[train_len:]
        test_sentences = test_in_df_sentence
        train_labels = train_in_df_label[:train_len]
        val_labels = train_in_df_label[train_len:]
        test_labels = test_in_df_label
        print(len(train_sentences))
        print(len(val_sentences))
        print(len(test_sentences))
        return train_sentences, val_sentences, test_sentences, train_labels, val_labels, test_labels
    elif dataset == 'wos':
        TESTING_SPLIT = 0.8
        VALIDATION_SPLIT = 0.8
        file_path = '/localscratch/yzhuang43/data-valuation/text_classification/data/wos/X.txt'
        with open(file_path, 'r') as read_file:
            x_temp = read_file.readlines()
            x_all = []
            for x in x_temp:
                x_all.append(str(x))

        print(len(x_all))

        file_path = '/localscratch/yzhuang43/data-valuation/text_classification/data/wos/Y.txt'
        with open(file_path, 'r') as read_file:
            y_temp= read_file.readlines()
            y_all = []
            for y in y_temp:
                y_all.append(int(y))
        print(len(y_all))
        print(max(y_all), min(y_all))

        train_val_len = int(TESTING_SPLIT * len(x_all))
        train_len = int(VALIDATION_SPLIT * train_val_len)

        train_sentences = x_all[:train_len]
        val_sentences = x_all[train_len:train_val_len]
        test_sentences = x_all[train_val_len:]

        train_labels = y_all[:train_len]
        val_labels = y_all[train_len:train_val_len]
        test_labels = y_all[train_val_len:]

        print(len(train_labels))
        print(len(val_labels))
        print(len(test_labels))
        return train_sentences, val_sentences, test_sentences, train_labels, val_labels, test_labels
    elif dataset == 'trec' or dataset == 'chemprot' or dataset == 'semeval':
        noisy_train_labels, train_sentences, train_labels = [], [], []
        noisy_val_labels, val_sentences, val_labels = [], [], []
        noisy_test_labels, test_sentences, test_labels = [], [], []
        with open(args.folder_path+'/train.json', encoding='utf-8') as f:
            for line in f.readlines():
                d = json.loads(line)
                noisy_train_labels.append(d['weak_label'])
                train_sentences.append(d['text'])
                train_labels.append(d['label'])
            f.close()
        with open(args.folder_path+'/valid.json', encoding='utf-8') as f:
            for line in f.readlines():
                d = json.loads(line)
                noisy_val_labels.append(d['weak_label'])
                val_sentences.append(d['text'])
                val_labels.append(d['label'])
            f.close()
        with open(args.folder_path+'/test.json', encoding='utf-8') as f:
            for line in f.readlines():
                d = json.loads(line)
                noisy_test_labels.append(d['weak_label'])
                test_sentences.append(d['text'])
                test_labels.append(d['label'])
            f.close()
        return train_sentences, val_sentences, test_sentences, train_labels, val_labels, test_labels, noisy_train_labels, noisy_val_labels, noisy_test_labels
    
    

def read_data(args, num_labels):
    # load dataset
    train_sentences, val_sentences, test_sentences, train_labels, val_labels, test_labels = load_dataset(args, args.dataset)

    tokenizer = BertTokenizer.from_pretrained(args.bert, do_lower_case=True)

    train_input_ids = []
    val_input_ids = []
    test_input_ids = []
    
    if args.dataset == '20news':
        MAX_LEN = 150
    elif args.dataset == 'chemprot':
        MAX_LEN = 512
    else:
        MAX_LEN = 128

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

    if args.noise_type == 'SN':
        noisy_train_labels = corrupt_dataset_SN(args, train_labels, num_labels)
        noisy_validation_labels = corrupt_dataset_SN(args, validation_labels, num_labels)
    elif args.noise_type == 'ASN':
        noisy_train_labels = corrupt_dataset_ASN(args, train_labels, num_labels)
        noisy_validation_labels = corrupt_dataset_ASN(args, validation_labels, num_labels)
    elif args.noise_type == 'IDN':
        noisy_train_labels = corrupt_dataset_IDN(args, train_inputs, train_labels, num_labels)
        noisy_validation_labels = corrupt_dataset_IDN(args, validation_inputs, validation_labels, num_labels)
    # Create an iterator of our data with torch DataLoader. 
    return train_inputs, train_masks, train_labels, noisy_train_labels, validation_inputs, validation_masks, validation_labels, noisy_validation_labels, test_inputs, test_masks, test_labels

def read_noisy_data(args, num_labels):
    # load dataset
    train_sentences, val_sentences, test_sentences, train_labels, val_labels, test_labels, noisy_train_labels, noisy_val_labels, noisy_test_labels = load_dataset(args, args.dataset)

    tokenizer = BertTokenizer.from_pretrained(args.bert, do_lower_case=True)

    train_input_ids = []
    val_input_ids = []
    test_input_ids = []
    if args.dataset == 'trec':
        MAX_LEN = 64
    elif args.dataset == 'chemprot':
        MAX_LEN = 512
    else:
        MAX_LEN = 128

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
    noisy_train_labels = torch.tensor(noisy_train_labels)
    noisy_validation_labels = torch.tensor(noisy_val_labels)
    noisy_test_labels = torch.tensor(noisy_test_labels)
    return train_inputs, train_masks, train_labels, noisy_train_labels, validation_inputs, validation_masks, validation_labels, noisy_validation_labels, test_inputs, test_masks, test_labels


def create_dataset(args):
    if args.dataset == '20news' or args.dataset == 'agnews' or args.dataset == 'wos':
        if args.dataset == '20news':
            num_labels = 20
            args.num_classes = 20
        elif args.dataset == 'agnews':
            num_labels = 4
            args.num_classes = 4
        elif args.dataset == 'wos':
            num_labels = 134
            args.num_classes = 134
        if args.saved_dataset == 'n':
            train_inputs, train_masks, train_labels, noisy_train_labels, validation_inputs, validation_masks, validation_labels, noisy_validation_labels, test_inputs, test_masks, test_labels = read_data(args, num_labels)
            train_data = TensorDataset(train_inputs, train_masks, train_labels, noisy_train_labels)
            train_sampler = SequentialSampler(train_data)
            train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
            validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels, noisy_validation_labels)
            validation_sampler = SequentialSampler(validation_data)
            validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=args.eval_batch_size)
            prediction_data = TensorDataset(test_inputs, test_masks, test_labels)
            prediction_sampler = SequentialSampler(prediction_data)
            prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=args.eval_batch_size)
            file_path = args.path + '/saved_data/{}-{}-{}-train-{}.pt'.format(args.dataset, args.noise_type, args.noise_ratio, args.seed)
            torch.save(train_data, file_path)
            file_path = args.path + '/saved_data/{}-{}-{}-val-{}.pt'.format(args.dataset, args.noise_type, args.noise_ratio, args.seed)
            torch.save(validation_data, file_path)
            file_path = args.path + '/saved_data/{}-{}-{}-test-{}.pt'.format(args.dataset, args.noise_type, args.noise_ratio, args.seed)
            torch.save(prediction_data, file_path)
            
        else:
            train_file_path = args.path + '/saved_data/{}-{}-{}-train-{}.pt'.format(args.dataset, args.noise_type, args.noise_ratio, args.seed)
            train_data = torch.load(train_file_path)
            train_sampler = SequentialSampler(train_data)
            train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
            val_file_path = args.path + '/saved_data/{}-{}-{}-val-{}.pt'.format(args.dataset, args.noise_type, args.noise_ratio, args.seed)
            validation_data = torch.load(val_file_path)
            validation_sampler = SequentialSampler(validation_data)
            validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=args.eval_batch_size)
            prediction_file_path = args.path + '/saved_data/{}-{}-{}-test-{}.pt'.format(args.dataset, args.noise_type, args.noise_ratio, args.seed)
            prediction_data = torch.load(prediction_file_path)
            prediction_sampler = SequentialSampler(prediction_data)
            prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=args.eval_batch_size)
    elif args.dataset == 'trec' or args.dataset == 'chemprot' or args.dataset == 'semeval':
        if args.dataset == 'trec':
            num_labels = 6
            args.num_classes = 6
        elif args.dataset == 'chemprot':
            num_labels = 10
            args.num_classes = 10
        elif args.dataset == 'semeval':
            num_labels = 9
            args.num_classes = 9
        if args.saved_dataset == 'n':
            train_inputs, train_masks, train_labels, noisy_train_labels, validation_inputs, validation_masks, validation_labels, noisy_validation_labels, test_inputs, test_masks, test_labels = read_noisy_data(args, num_labels)
            train_data = TensorDataset(train_inputs, train_masks, train_labels, noisy_train_labels)
            train_sampler = SequentialSampler(train_data)
            train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
            validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels, noisy_validation_labels)
            validation_sampler = SequentialSampler(validation_data)
            validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=args.eval_batch_size)
            prediction_data = TensorDataset(test_inputs, test_masks, test_labels)
            prediction_sampler = SequentialSampler(prediction_data)
            prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=args.eval_batch_size)
            file_path = args.path + '/saved_data/{}-{}-{}-train-{}.pt'.format(args.dataset, args.noise_type, args.noise_ratio, args.seed)
            torch.save(train_data, file_path)
            file_path = args.path + '/saved_data/{}-{}-{}-val-{}.pt'.format(args.dataset, args.noise_type, args.noise_ratio, args.seed)
            torch.save(validation_data, file_path)
            file_path = args.path + '/saved_data/{}-{}-{}-test-{}.pt'.format(args.dataset, args.noise_type, args.noise_ratio, args.seed)
            torch.save(prediction_data, file_path)
            
        else:
            train_file_path = args.path + '/saved_data/{}-{}-{}-train-{}.pt'.format(args.dataset, args.noise_type, args.noise_ratio, args.seed)
            train_data = torch.load(train_file_path)
            train_sampler = SequentialSampler(train_data)
            train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
            val_file_path = args.path + '/saved_data/{}-{}-{}-val-{}.pt'.format(args.dataset, args.noise_type, args.noise_ratio, args.seed)
            validation_data = torch.load(val_file_path)
            validation_sampler = SequentialSampler(validation_data)
            validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=args.eval_batch_size)
            prediction_file_path = args.path + '/saved_data/{}-{}-{}-test-{}.pt'.format(args.dataset, args.noise_type, args.noise_ratio, args.seed)
            prediction_data = torch.load(prediction_file_path)
            prediction_sampler = SequentialSampler(prediction_data)
            prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=args.eval_batch_size)

    return train_data, train_sampler, train_dataloader, validation_data, validation_sampler, validation_dataloader, prediction_data, prediction_sampler, prediction_dataloader
