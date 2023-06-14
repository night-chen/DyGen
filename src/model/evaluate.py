"""
Evaluate the performance after the generative model. 
Modified from: https://github.com/BaeHeeSun/NPC/blob/main/evaluate.py
"""
import time
import pickle

from src.model.network import *

class Acc_calculator_mbr:
    def __init__(self, args, len_test):
        self.args = args
        self.n_classes = args.num_classes
        self.time = time.time()
        self.len_test = len_test

    def merge_classifier_and_autoencoder(self, test_dataloader, best_model, vae_model, vae_batch_size):
        best_model.eval()
        vae_model.eval()
        batch_size = vae_batch_size
        accuracy = 0
        accuracy_pre = 0
        for batch in test_dataloader:
            batch = tuple(t.to(self.args.device) for t in batch)
            b_input_ids, b_input_mask, b_labels, b_z0 = batch
            # classifier side ==> P(y^|x)
            with torch.no_grad():
                outputs = best_model(b_input_ids, attention_mask=b_input_mask)
                logits = [output[0] for output in outputs]
                p_y_tilde = [F.softmax(logit, dim=-1).detach().cpu() for logit in logits]
                avg_p_y_tilde = torch.mean(torch.stack(p_y_tilde, dim=0), 0)
                _, pseudo_label_pre = torch.max(avg_p_y_tilde, dim=-1)
                accuracy_pre += torch.sum(b_labels == pseudo_label_pre.to(b_labels.device)).item()

            # autoencoder side ==> P(y|y^,x)
            batch_size = len(b_labels)
            iter_m = len(outputs)
            p_y_y_tilde_list = []
            for i in range(iter_m):
                p_y_bar_x_y_tilde = torch.zeros(batch_size, self.n_classes, self.n_classes)
                for lab in range(self.n_classes):
                    label_one_hot = torch.zeros(batch_size, self.n_classes)
                    label_one_hot[:, lab] = 1
                    label_one_hot = label_one_hot.to(self.args.device)
                    _, alpha_infer = vae_model[i](b_z0[:,i,:].squeeze(), label_one_hot)

                    alpha_infer = alpha_infer.detach().cpu() - 1.0
                    p_y_bar_x_y_tilde[:, :, lab] = alpha_infer / torch.sum(alpha_infer, dim=1).view(-1, 1)

                    del label_one_hot

                # P(y|y^,x)*P(y^|x)=P(y,y^|x)
                p_y_expansion = p_y_tilde[i].squeeze().reshape(batch_size, 1, self.n_classes).repeat([1, self.n_classes, 1])  # batch*class*label
                p_y_y_tilde = p_y_bar_x_y_tilde * p_y_expansion  # batch*class*label
                p_y_y_tilde_list.append(p_y_y_tilde)
            if iter_m == 1:
                p_y_y_tilde_final = p_y_y_tilde_list[-1].squeeze()
            else:
                p_y_y_tilde_final = torch.stack(p_y_y_tilde_list, dim=0).mean(0)
            _, pseudo_label = torch.max(torch.sum(p_y_y_tilde_final, dim=2), dim=1)
            accuracy += torch.sum(b_labels == pseudo_label.to(b_labels.device)).item()

        return accuracy_pre / self.len_test, accuracy / self.len_test