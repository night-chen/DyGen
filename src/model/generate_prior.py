"""
Compute the prior function.
"""
import time
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from src.model.network import *

class KNN_prior_dynamic:
    def __init__(self, args, dataset, z0, noisy_train_labels, true_train_labels, noisy_markers):
        self.args = args
        self.n_classes = self.args.num_classes
        self.time = time.time()

        self.dataset = dataset
        self.y_hat = noisy_train_labels
        self.y = true_train_labels
        self.noisy_markers = noisy_markers
        self.z0 = z0
        self.emb_dim = z0.shape[-1]

        print('\n===> Prior Generation with KNN Start')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # For each datapoint, get initial class name
    def get_prior(self, best_model):
        # Load Classifier
        self.net = best_model
        self.net.to(self.device)

        # k-nn
        self.net.eval()
        # knn mode on
        neigh = KNeighborsClassifier(n_neighbors=10, weights='distance')
        embeddings, class_confi = [], []
        knn_embeds = self.z0[self.noisy_markers==0, :]
        class_confi = self.y_hat[self.noisy_markers==0]

        # class_confi = self.y_hat
        knn_embeds = knn_embeds.cpu().detach().numpy()
        class_confi = class_confi.cpu().detach().numpy()
        neigh.fit(knn_embeds, class_confi)
        print('Time : ', time.time() - self.time)

        # 2. predict class of training dataset
        knn_embeds = self.z0.cpu().detach().numpy()
        class_preds = neigh.predict(knn_embeds)
        class_preds = torch.tensor(np.int64(class_preds))
        print('Prior made {} errors with train/val noisy labels'.format(torch.sum(class_preds!=self.y_hat)))
        print('Prior made {} errors with train/val clean labels'.format(torch.sum(class_preds!=self.y)))
        noisy_preds = torch.tensor([(class_preds[i] != self.y_hat[i]) and (class_preds[i] == self.y[i]) for i in range(len(class_preds))])
        print('Prior detected {} real noisy samples'.format(torch.sum(noisy_preds)))

        # # proba
        dict = {}
        model_output = neigh.predict_proba(knn_embeds)
        if model_output.shape[1] < self.n_classes:
            tmp = np.zeros((model_output.shape[0], self.n_classes))
            tmp[:, neigh.classes_] = neigh.predict_proba(embeddings)
            dict['proba'] = tmp
        else:
            dict['proba'] = model_output  # data*n_class

        print('Time : ', time.time() - self.time, 'proba information saved')


        return dict['proba'], class_preds.numpy()