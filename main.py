import argparse
import os
import numpy as np
import torch
from torch import nn
from tqdm import tqdm, trange
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from transformers import AutoTokenizer
from torch.optim import Adam
from transformers  import BertTokenizer, BertConfig
from transformers  import AdamW, BertModel, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler
from src.utils.utils2 import *
from src.model.generate_prior import *
from src.dynamics.euclidean_dist import *
from src.model.train_stage1 import *
from src.model.train_stage2 import *
from src.model.evaluate import *
from src.utils.generate_noise import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--vae_lr", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=128, type=int, help="Batch size for training.")
    parser.add_argument("--vae_batch_size", default=32, type=int, help="Batch size for training.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs for training.")
    parser.add_argument("--vae_epochs", default=20, type=int, help="Number of epochs for training.")
    parser.add_argument("--seed", default=0, type=int, help="Number of epochs for training.")
    parser.add_argument("--dataset", default='20news', type=str, help="dataset")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument('--saved_dataset', type=str, default='n', help='whether save the preprocessed pt file of the dataset')
    parser.add_argument("--noise_ratio", type=float, default=0.0, help='The ratio of noisy data to be poisoned.')
    parser.add_argument("--noise_type", type=str, default="s")
    parser.add_argument("--n_model", type=int, default=2, help='The number of detection-relabeling iterations.')
    parser.add_argument("--knn_mode", type=str, default='onehot', help='Choose the relabeling method: second_close')
    parser.add_argument("--selected_class", type=str, default='1', help='Choose the relabeling method: second_close')
    parser.add_argument("--prior_norm", type=int, default=5, help='Choose the relabeling method: second_close')
    parser.add_argument("--alpha_gp", type=float, default=0.0, help='The hyperparameter before the gradient penalty loss.')
    parser.add_argument("--length_scale", type=float, default=0.5, help='The length scale.')
    parser.add_argument("--gamma", type=float, default=0.999, help='The length scale.')
    parser.add_argument("--conditional", action="store_true")
    parser.add_argument('--softplus_beta', type=float, default=1, help='softplus beta')
    parser.add_argument("--total_iter", type=int, default=10, help='total iter (Default : 10)')
    parser.add_argument('--beta', type=float, default=1.0, help='coefficient on kl loss, beta vae')
    parser.add_argument('--clip_gradient_norm', type=float, default=100000, help='max norm for gradient clipping')
    
    # switches
    parser.add_argument('--lambda_t', type=float, default=5)
    parser.add_argument('--alpha_t_hi', type=float, default=5)
    parser.add_argument('--warmup_epochs', type=float, default=0.1)
    parser.add_argument("--path", type=str, default='./datasets/20news')
    parser.add_argument("--bert", type=str, default="bert-base-uncased")
    
    args = parser.parse_args()
    args.n_gpu = 1
    print(args)
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    set_seed(args)

    if args.dataset == '20news':
        num_labels = 20
        args.num_classes = 20
    elif args.dataset == 'agnews':
        num_labels = 4
        args.num_classes = 4
    elif args.dataset =='wos':
        num_labels = 134
        args.num_classes = 134
    elif args.dataset == 'trec':
        num_labels = 6
        args.num_classes = 6
    elif args.dataset == 'chemprot':
        num_labels = 10
        args.num_classes = 10
    elif args.dataset == 'semeval':
        num_labels = 9
        args.num_classes = 9
    
    train_data, train_sampler, train_dataloader, validation_data, validation_sampler, validation_dataloader, test_data, test_sampler, test_dataloader = create_dataset(args)
    noisy_train_labels = torch.tensor([train_data[idx][-1] for idx in range(len(train_data))])
    train_inputs = torch.stack([train_data[idx][0] for idx in range(len(train_data))], dim=0)
    train_masks = torch.stack([train_data[idx][1] for idx in range(len(train_data))], dim=0)
    train_labels = torch.stack([train_data[idx][2] for idx in range(len(train_data))], dim=0)
    validation_inputs = torch.stack([validation_data[idx][0] for idx in range(len(validation_data))], dim=0)
    validation_masks = torch.stack([validation_data[idx][1] for idx in range(len(validation_data))], dim=0)
    validation_labels = torch.stack([validation_data[idx][2] for idx in range(len(validation_data))], dim=0)
    noisy_validation_labels = torch.tensor([validation_data[idx][-1] for idx in range(len(validation_data))])
    test_inputs = torch.stack([test_data[idx][0] for idx in range(len(test_data))], dim=0)
    test_masks = torch.stack([test_data[idx][1] for idx in range(len(test_data))], dim=0)
    test_labels = torch.stack([test_data[idx][2] for idx in range(len(test_data))], dim=0)
    number_points = int((len(train_data)+len(validation_data)) * args.noise_ratio)

    print("================Start Training Stage I Model: Encode the Trajectory!================")
    z_train, z_val, z_test, best_model, dists_list = train_stage1(args, train_dataloader, validation_dataloader, test_dataloader)

    print("================Start Training Stage II Model: Compute the Training Dynamics!================")
    dists_score_list = []
    markers_list = []
    for idx in range(dists_list.shape[0]):
        dists = dists_list[idx].squeeze()
        dists_labels = torch.cat((noisy_train_labels, noisy_validation_labels), 0)
        dists_mean = torch.mean(dists, 0)
        dists_mean = torch.tensor([dists_mean[i, dists_labels[i]] for i in range(len(dists_labels))])
        dists_var = torch.std(dists, 0)
        dists_var = torch.tensor([dists_var[i, dists_labels[i]] for i in range(len(dists_labels))])
        dists_score = dists_mean + dists_var
        dists_score = dists_score[:len(dists_labels)]
        markers = torch.zeros(len(dists_labels))
        number_points = int(len(dists_score) * args.noise_ratio)
        noisy_points = torch.topk(dists_score, number_points, largest=True).indices
        markers[noisy_points] = 1
        dists_score_list.append(dists_score.unsqueeze(0))
        markers_list.append(markers.unsqueeze(0))
    dists_score_list = torch.stack(dists_score_list, dim=0)
    markers_list = torch.stack(markers_list, dim=0)

    print("================Start Training Stage III Model: Compute the Prior!================")
    print(z_train.shape) # (models, epochs, batch, dim)
    z_train = z_train.permute(2,0,1,3)
    B, M, N, D = z_train.shape
    z_train = z_train.reshape(B, M, N*D)
    z0_train = z_train[:, :, :D]

    z_val = z_val.permute(2,0,1,3)
    B2, M2, N2, D2 = z_val.shape
    z_val = z_val.reshape(B2, M2, N2*D2)
    z0_val = z_val[:, :, :D2]

    z_test = z_test.permute(2,0,1,3)
    B3, M3, N3, D3 = z_test.shape
    z_test = z_test.reshape(B3, M3, N3*D3)
    z0_test = z_test[:, :, :D3]
    train_priors = []
    val_priors = []
    for idx in range(M):
        knn_inputs = torch.cat((train_inputs, validation_inputs), 0)
        knn_masks = torch.cat((train_masks, validation_masks), 0)
        knn_z0 = torch.cat((z0_train[:, idx, :], z0_val[:, idx, :]), 0).squeeze()
        knn_labels = torch.cat((noisy_train_labels, noisy_validation_labels))
        knn_true_labels = torch.cat((train_labels, validation_labels))
        knn_data = TensorDataset(knn_inputs, knn_masks, knn_z0, knn_labels)
        knn_sampler = SequentialSampler(knn_data)
        knn_dataloader = DataLoader(knn_data, sampler=knn_sampler, batch_size=args.vae_batch_size)
        
        knn_prior = KNN_prior_dynamic(args, knn_data, knn_z0, knn_labels, knn_true_labels, markers_list[idx].squeeze())
        if args.knn_mode == 'onehot': 
            _, priors = knn_prior.get_prior(best_model)
        else:
            priors, _ = knn_prior.get_prior(best_model)
    
        priors = torch.tensor(priors)
        train_priors.append(priors[:B])
        val_priors.append(priors[B:])
    train_priors = torch.stack(train_priors, dim=0)
    val_priors = torch.stack(val_priors, dim=0)
    train_priors = train_priors.permute(1,0)
    val_priors = val_priors.permute(1,0)
    if M == 1:
        train_priors = train_priors.squeeze().unsqueeze(-1)
        val_priors = val_priors.squeeze().unsqueeze(-1)

    scaler = torch.cuda.amp.GradScaler()

    # prepare datasets for generative model
    train_z_data = TensorDataset(z_train, train_priors, noisy_train_labels)
    val_z_data = TensorDataset(z_val, val_priors, noisy_validation_labels)
    test_z_data = TensorDataset(test_inputs, test_masks, test_labels, z_test)

    train_z_sampler = SequentialSampler(train_z_data)
    val_z_sampler = SequentialSampler(val_z_data)
    test_z_sampler = SequentialSampler(test_z_data)
    train_z_dataloader = DataLoader(train_z_data, sampler=train_z_sampler, batch_size=args.vae_batch_size)
    val_z_dataloader = DataLoader(val_z_data, sampler=val_z_sampler, batch_size=args.vae_batch_size)
    test_z_dataloader = DataLoader(test_z_data, sampler=test_z_sampler, batch_size=args.vae_batch_size)    

    args.save_path = './cache/dygen_acc_{}_{}.npy'.format(args.noise_ratio, args.total_iter)
    

    print("================Start Training Stage IV Model: Estimate Transition Matrix!================")
    gen_model = stage2_gen(args, scaler, train_z_dataloader, val_z_dataloader, test_z_dataloader, len(train_z_data), len(val_z_data), len(test_z_data), z_train.shape[-1])
    func = Acc_calculator_mbr(args, len(test_z_data))
    vae_model, best_acc = gen_model.run(func, test_z_dataloader, best_model)

    accuracy_pre, accuracy = func.merge_classifier_and_autoencoder(test_z_dataloader, best_model, vae_model, args.vae_batch_size)
    print("The performance after denoising:", accuracy)
    f = open('./cache/{}-results.logs'.format(args.dataset), 'a')
    f.write('DyGen-'+args.noise_type+'-'+str(args.noise_ratio)+'-'+str(args.total_iter)+'-'+str(args.lambda_t)+'-'+str(args.warmup_epochs)+'-'+str(args.n_model)+'-'+str(args.epochs)+'-'+str(args.seed))
    f.write('\n')
    f.write("The orignal model performance is: "+str(accuracy_pre))
    f.write('\n')
    f.write("The performance after NPC is: "+str(accuracy))
    f.write('\n')
    f.write("="*12)
    f.write('\n')
        
if __name__ == "__main__":
    main()