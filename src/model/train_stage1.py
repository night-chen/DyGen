import os
import numpy as np
import torch
from torch import nn
from tqdm import tqdm, trange
from transformers  import AdamW, BertModel, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler
from src.utils.utils import *
from src.utils.utils2 import *
from src.dynamics.euclidean_dist import *

def kl_div(p, q):
    return (p * ((p + 1e-5).log() - (q + 1e-5).log())).sum(-1)

class NLLModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        num_labels = args.num_classes
        self.args = args
        self.models = nn.ModuleList()
        self.device = [i % args.n_gpu for i in range(args.n_model)]
        self.loss_fnt = nn.CrossEntropyLoss()
        for i in range(args.n_model):
            model = BertForSequenceClassification.from_pretrained(args.bert, num_labels=num_labels, output_hidden_states=True)
            model.to(self.device[i])
            self.models.append(model)

    def forward(self, input_ids, attention_mask, labels=None):
        num_models = len(self.models)
        outputs = []
        for i in range(num_models):
            output = self.models[i](
                input_ids=input_ids.to(self.device[i]),
                attention_mask=attention_mask.to(self.device[i]),
                labels=labels.to(self.device[i]) if labels is not None else None,
                return_dict=False,
            )
            outputs.append(output)

        model_output = outputs
        if labels is not None:
            loss = sum([output[0] for output in outputs]) / num_models
            logits = [output[1] for output in outputs]
            probs = [F.softmax(logit, dim=-1) for logit in logits]
            avg_prob = torch.stack(probs, dim=0).mean(0)
            reg_loss = sum([kl_div(avg_prob, prob) for prob in probs]) / num_models
            loss = loss + self.args.alpha_t * reg_loss.mean()
            return loss
        return model_output

def accurate_nb(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat)

def train_stage1(args, train_dataloader, validation_dataloader, prediction_dataloader):
    model = NLLModel(args)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.to(args.device)

    t_total = len(train_dataloader) * args.epochs
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-9)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.06*t_total), num_training_steps=t_total)
    scaler = GradScaler()
    # Store our loss and accuracy for plotting

    best_val = -np.inf
    # trange is a tqdm wrapper around the normal python range
    dists_epochs = []
    train_embeds_list = [None] * len(model.models)
    eval_embeds_list = [None] * len(model.models)
    test_embeds_list = [None] * len(model.models)
    dists_list = [None] * len(model.models)
    num_epochs = 0
    for epoch in trange(args.epochs, desc="Epoch"): 
      # Training
        # Set our model to training mode (as opposed to evaluation mode)
        # Tracking variables
        tr_loss =  0
        nb_tr_examples, nb_tr_steps = 0, 0
        model.train()
        # Train the data for one epoch
        for step, batch in enumerate(train_dataloader):
            # Add batch to GPU
            batch = tuple(t.to(args.device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, _, b_labels = batch

            if num_epochs < int(args.epochs/10):
                args.alpha_t = 0
            else:
                args.alpha_t = args.alpha_t_hi
            
            loss_ce = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            if torch.cuda.device_count() > 1:
                loss_ce = loss_ce.mean()
            scaler.scale(loss_ce).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            model.zero_grad()
            tr_loss += loss_ce.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1
        print("Train cross entropy loss: {}".format(tr_loss/nb_tr_steps))
        num_epochs += 1
        
        # Validation
        # Put model in evaluation mode to evaluate loss on the validation set
        model.eval()
        train_embeds = [None] * len(model.models)
        train_labels = [None] * len(model.models)
        train_logits = [None] * len(model.models)
        for batch in train_dataloader:
            batch = tuple(t.to(args.device) for t in batch)
            b_input_ids, b_input_mask, _, b_labels = batch
            with torch.no_grad():
                outputs = model(b_input_ids, attention_mask=b_input_mask)
                for idx in range(len(outputs)):
                    logits = outputs[idx][0]
                    if train_embeds[idx] == None:
                        train_embeds[idx] = outputs[idx][-1][-1][:,0,:].squeeze()
                        train_labels[idx] = b_labels
                        train_logits[idx] = F.softmax(outputs[idx][0], dim=-1)
                    else:
                        train_embeds[idx] = torch.cat((train_embeds[idx], outputs[idx][-1][-1][:,0,:].squeeze()), 0)
                        train_labels[idx] = torch.cat((train_labels[idx], b_labels), 0)
                        train_logits[idx] = torch.cat((train_logits[idx], F.softmax(outputs[idx][0], dim=-1)), 0)
        for idx in range(len(outputs)):
            if train_embeds_list[idx] == None:
                train_embeds_list[idx] = torch.zeros((args.epochs, train_embeds[idx].shape[0], train_embeds[idx].shape[1]))
                train_embeds_list[idx][0] = train_embeds[idx].detach()
            else:
                train_embeds_list[idx][epoch] = train_embeds[idx].detach()
        # Tracking variables 
        eval_accurate_nb = 0
        nb_eval_examples = 0
        logits_list = []
        labels_list = []

        model.eval()
        eval_embeds = [None] * len(model.models)
        eval_labels = [None] * len(model.models)
        eval_logits = [None] * len(model.models)
        # Evaluate data for one epoch
        for batch in validation_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(args.device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels, _ = batch
            # Telling the model not to compute or store gradients, saving memory and speeding up validation
            with torch.no_grad():
            # Forward pass, calculate logit predictions
                outputs = model(b_input_ids, attention_mask=b_input_mask)
                for idx in range(len(outputs)):
                    if eval_embeds[idx] == None:
                        eval_embeds[idx] = outputs[idx][-1][-1][:,0,:].squeeze()
                        eval_labels[idx] = b_labels
                        eval_logits[idx] = F.softmax(outputs[idx][0], dim=-1)
                    else:
                        eval_embeds[idx] = torch.cat((eval_embeds[idx], outputs[idx][-1][-1][:,0,:].squeeze()), 0)
                        eval_labels[idx] = torch.cat((eval_labels[idx], b_labels), 0)
                        eval_logits[idx] = torch.cat((eval_logits[idx], F.softmax(outputs[idx][0], dim=-1)), 0)
                logits = [output[0] for output in outputs]
                logits = logits[-1]
                logits_list.append(logits)
                labels_list.append(b_labels)
            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            tmp_eval_nb = accurate_nb(logits, label_ids)
    
            eval_accurate_nb += tmp_eval_nb
            nb_eval_examples += label_ids.shape[0]
        for idx in range(len(model.models)):
            if eval_embeds_list[idx] == None:
                eval_embeds_list[idx] = torch.zeros((args.epochs, eval_embeds[idx].shape[0], eval_embeds[idx].shape[1]))
                eval_embeds_list[idx][0] = eval_embeds[idx].detach()
            else:
                eval_embeds_list[idx][epoch] = eval_embeds[idx].detach()
        eval_accuracy = eval_accurate_nb/nb_eval_examples
        print("Validation Accuracy: {}".format(eval_accuracy))
        scheduler.step(eval_accuracy)

        if eval_accuracy > best_val:
            best_val = eval_accuracy
            best_model = model
        
        # Put model in evaluation mode
        model.eval()
        # Tracking variables 
        eval_accurate_nb = 0
        nb_test_examples = 0
        logits_list = []
        labels_list = []
        test_embeds = [None] * len(model.models)
        test_labels = [None] * len(model.models)
        # Predict 
        for batch in prediction_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(args.device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Telling the model not to compute or store gradients, saving memory and speeding up prediction
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                # outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
                outputs = model(b_input_ids, attention_mask=b_input_mask)
                logits = [output[0] for output in outputs]
                logits = logits[-1] #torch.stack(logits, dim=0).mean(0)
                for idx in range(len(outputs)):
                    if test_embeds[idx] == None:
                        test_embeds[idx] = outputs[idx][-1][-1][:,0,:].squeeze()
                        test_labels[idx] = b_labels
                    else:
                        test_embeds[idx] = torch.cat((test_embeds[idx], outputs[idx][-1][-1][:,0,:].squeeze()), 0)
                        test_labels[idx] = torch.cat((test_labels[idx], b_labels), 0)
                logits_list.append(logits)
                labels_list.append(b_labels)
            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            tmp_eval_nb = accurate_nb(logits, label_ids)
            eval_accurate_nb += tmp_eval_nb
            nb_test_examples += label_ids.shape[0]
        for idx in range(len(outputs)):
            if test_embeds_list[idx] == None:
                test_embeds_list[idx] = torch.zeros((args.epochs, test_embeds[idx].shape[0], test_embeds[idx].shape[1]))
                test_embeds_list[idx][0] = test_embeds[idx].detach()
            else:
                test_embeds_list[idx][epoch] = test_embeds[idx].detach()

        print("Test Accuracy: {}".format(eval_accurate_nb/nb_test_examples))
        
        if args.dataset != 'wos' and args.dataset != 'tacred':
            full_dists = [None] * len(model.models)
            for idx in range(len(model.models)):
                dists_embeds = torch.cat((train_embeds[idx], eval_embeds[idx], test_embeds[idx]), 0)
                dists_labels = torch.cat((train_labels[idx], eval_labels[idx], test_labels[idx]), 0)
                dists = euclidean_dist(args, dists_embeds, dists_labels)
                full_dists[idx] = dists
                dists = [dists[i][dists_labels[i]] for i in range(len(dists))]
                dists_epochs.append(dists)
            
                if dists_list[idx] == None:
                    dists_list[idx] = torch.zeros((args.epochs, full_dists[idx].shape[0], full_dists[idx].shape[1]))
                    dists_list[idx][0] = full_dists[idx].detach()
                else:
                    dists_list[idx][epoch] = full_dists[idx].detach()
        else:
            full_dists = [None] * len(model.models)
            for idx in range(len(model.models)):
                dists_embeds = torch.cat((train_embeds[idx], eval_embeds[idx], test_embeds[idx]), 0)
                dists_labels = torch.cat((train_labels[idx], eval_labels[idx], test_labels[idx]), 0)
                dists = euclidean_dist_wos(args, dists_embeds, dists_labels)
                full_dists[idx] = dists
                dists = [dists[i][dists_labels[i]] for i in range(len(dists))]
                dists_epochs.append(dists)
            
                if dists_list[idx] == None:
                    dists_list[idx] = torch.zeros((args.epochs, full_dists[idx].shape[0], full_dists[idx].shape[1]))
                    dists_list[idx][0] = full_dists[idx].detach()
                else:
                    dists_list[idx][epoch] = full_dists[idx].detach()
    train_embeds_list = torch.stack(train_embeds_list, dim=0)
    eval_embeds_list = torch.stack(eval_embeds_list, dim=0)
    test_embeds_list = torch.stack(test_embeds_list, dim=0)
    dists_list = torch.stack(dists_list, dim=0)
    return train_embeds_list, eval_embeds_list, test_embeds_list, best_model, dists_list