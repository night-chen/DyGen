import torch
import torch.nn as nn
import torch.optim as optim
import time
import torch.nn.functional as F
from torch.autograd import Variable
from src.model.network import *
from src.utils.utils import *
from src.utils.utils2 import *

def kl_div(p, q):
    return (p * ((p + 1e-5).log() - (q + 1e-5).log())).sum(-1)

class stage2_gen:
    def __init__(self, args, grad_scaler, train_z0_dataloader, val_z0_dataloader, test_z0_dataloader, len_train, len_val, len_test, emb_dim):
        self.args = args
        self.grad_scaler = grad_scaler
        self.time = time.time()

        self.train_loader = train_z0_dataloader
        self.val_loader = val_z0_dataloader
        self.test_loader = test_z0_dataloader
        self.len_train = len_train
        self.len_val = len_val
        self.len_test = len_test

        print('\n===> AE Training Start')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.emb_dim = emb_dim
        self.AEs = nn.ModuleList()
        for i in range(args.n_model):
            AE = text_CVAE(self.args, self.emb_dim)
            AE.to(self.device)
            self.AEs.append(AE)

        self.optimizer = optim.Adam(self.AEs.parameters(), lr=self.args.lr)

    def generate_alpha_from_original_proba(self, y_prior):
        if self.args.knn_mode=='onehot':
            proba = torch.zeros(len(y_prior), self.args.num_classes).to(self.device)
            return proba.scatter(1,y_prior.view(-1, 1), self.args.prior_norm)+1.0+1/self.args.num_classes
        elif self.args.knn_mode=='proba': # proba
            if self.args.selected_class == self.args.num_classes:
                return self.args.prior_norm * y_prior + 1.0 + 1 / self.args.num_classes
            else: # topk
                values, indices = torch.topk(y_prior, k=int(self.args.selected_class), dim=1)
                y_prior_temp = Variable(torch.zeros_like(y_prior)).to(self.device)
                return self.args.prior_norm * y_prior_temp.scatter(1, indices, values) + 1.0 + 1 / self.args.num_classes

    def loss_function(self, y_tilde_data, y_tilde_recon, alpha_prior, alpha_infer):
        
        num_models = len(self.AEs)
        recon_loss_b = 0
        KL_loss_b = 0
        reg_loss_b = 0
        alpha = alpha_infer - 1
        alpha = alpha / torch.sum(alpha, dim=-1).unsqueeze(-1)
        avg_alpha = torch.mean(alpha, 0)
        for i in range(num_models):
            recon_loss = nn.BCEWithLogitsLoss(reduction='sum')(y_tilde_recon[i].float().squeeze(), y_tilde_data.float())

            KL = torch.sum(torch.lgamma(alpha_prior[i]) - torch.lgamma(alpha_infer[i]) + (alpha_infer[i] - alpha_prior[i]) *
                            torch.digamma(alpha_infer[i]), dim=1)

            recon_loss_b += recon_loss
            KL_loss_b += torch.sum(KL)
            temp_alpha = alpha_infer[i] - 1
            temp_alpha = temp_alpha / torch.sum(temp_alpha, dim=-1).unsqueeze(-1)
            reg_loss = kl_div(avg_alpha.squeeze(), temp_alpha.squeeze())
            reg_loss_b += torch.sum(reg_loss)
        return recon_loss_b/num_models, KL_loss_b/num_models, reg_loss_b/num_models

    def update_model(self):
        self.AEs.train()
        num_models = len(self.AEs)
        epoch_loss, epoch_recon, epoch_kl, epoch_reg = 0, 0, 0, 0
        batch = 0

        for step, batch in enumerate(self.train_loader):
            batch = tuple(t.to(self.args.device) for t in batch)
            b_inputs, b_priors, b_labels = batch
            label_one_hot = F.one_hot(b_labels, self.args.num_classes)
            
            alpha_prior = [None] * num_models
            for idx in range(num_models):
                alpha_prior[idx] = self.generate_alpha_from_original_proba(b_priors[:, idx].squeeze())

            with torch.cuda.amp.autocast():
                y_recon_list = []
                alpha_infer_list = []
                for idx in range(num_models):
                    y_recon, alpha_infer = self.AEs[idx](b_inputs[:,idx,:].squeeze(), label_one_hot)
                    y_recon_list.append(y_recon)
                    alpha_infer_list.append(alpha_infer)
                y_recon_list = torch.stack(y_recon_list, dim=0)
                alpha_infer_list = torch.stack(alpha_infer_list, dim=0)
                recon_loss, kl_loss, reg_loss = self.loss_function(label_one_hot, y_recon_list, alpha_prior, alpha_infer_list)
                loss = recon_loss + self.args.beta * kl_loss + self.lambda_t * reg_loss
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()

            self.optimizer.zero_grad()

            epoch_loss += loss.item()
            epoch_recon += torch.mean(recon_loss).item()
            epoch_kl += torch.mean(kl_loss).item()
            epoch_reg += torch.mean(reg_loss).item()

        time_elapse = time.time() - self.time

        return epoch_loss, epoch_recon, epoch_kl, epoch_reg, time_elapse

    def save_result(self, train_loss_total, train_loss_recon, train_loss_kl):
        self.train_loss.append(train_loss_total/self.len_train)
        self.train_recon_loss.append(train_loss_recon/self.len_train)
        self.train_kl_loss.append(train_loss_kl/self.len_train)

        print('Train', train_loss_total/self.len_train, train_loss_recon/self.len_train)

        return

    def run(self, func, test_z_dataloader, best_model):
        # initialize
        self.train_loss, self.train_recon_loss, self.train_kl_loss, self.train_reg_loss = [], [], [], []

        self.accuracy_list = []
        best_acc = 0
        for epoch in range(self.args.total_iter):
            if epoch < self.args.warmup_epochs * self.args.total_iter:
                self.lambda_t = 0
            else:
                self.lambda_t = self.args.lambda_t
            train_loss, train_recon, train_kl, train_reg, time_train = self.update_model()
            # print result loss
            print('=' * 50)
            print('Epoch', epoch, 'Time', time_train)
            self.save_result(train_loss, train_recon, train_kl)
            accuracy_pre, accuracy = func.merge_classifier_and_autoencoder(test_z_dataloader, best_model, self.AEs, self.args.vae_batch_size)
            self.accuracy_list.append(accuracy)
            if accuracy > best_acc:
                best_acc = accuracy
            print("The test performance after NPC:", accuracy)
        self.train_loss = torch.tensor(self.train_loss).numpy()
        self.train_recon_loss = torch.tensor(self.train_recon_loss).numpy()
        self.train_kl_loss = torch.tensor(self.train_kl_loss).numpy()
        self.train_reg_loss = torch.tensor(self.train_reg_loss).numpy()
        self.accuracy_list = torch.tensor(self.accuracy_list).numpy()
        np.save(self.args.save_path, self.accuracy_list)
        return self.AEs, best_acc

    def evaluate(self):
        logits_list = None
        test_acc = 0
        for step, batch in enumerate(self.test_loader):
            batch = tuple(t.to(self.args.device) for t in batch)
            b_inputs, b_masks, b_labels, b_z0 = batch
            label_one_hot = F.one_hot(b_labels, self.args.num_classes)
            y_recon, alpha_infer = self.AE(b_inputs, label_one_hot)
            if logits_list == None:
                logits_list = y_recon
            else:
                logits_list = torch.cat((logits_list, y_recon), 0)
        return logits_list

    def accurate_nb(self, preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat)
