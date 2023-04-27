import torch
from torch import nn, optim
from torch.nn import functional as F
from typing import Union, Mapping, Iterable, List
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.distributions import kl_divergence as kld
from torch.optim.lr_scheduler import ReduceLROnPlateau

from _loss import LossFunction
from _logger import mt
from _utilities import one_hot, get_k_elements
from _basic_layers import *
from _compat import Literal

import time
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
from sklearn.utils import class_weight
from collections import Counter 
from itertools import chain
from copy import deepcopy


class VAE(ReparameterizeLayerBase, MMDLayerBase):
    """Vanilla single-cell VAE"""
    def __init__(self, *,
                       adata: Optional[sc.AnnData] = None,
                       hidden_stacks: List[int] = [128], 
                       n_latent: int = 10,
                       n_batch: int = 0,
                       n_label: int = 0,
                       batch_key = None,
                       label_key = None,
                       dispersion:  Literal["gene", "gene-batch", "gene-cell"] = "gene-cell", 
                       log_variational: bool = True,
                       bias: bool = True,
                       use_batch_norm: bool = True,
                       use_layer_norm: bool = False,
                       batch_hidden_dim: int = 8,
                       batch_embedding: Literal["embedding", "onehot"] = "onehot",
                       constrain_latent_embedding: bool = False,
                       encode_libsize: bool = False,
                       dropout_rate: float = 0.1,
                       activation_fn: nn.Module = nn.ReLU,
                       inject_batch: bool = True,
                       inject_label: bool = False,
                       use_attention: bool = False,
                       device: Union[str, torch.device] = "cpu") -> None:
        """
        @brief VAE                    Variational Autoencoder for a Raw Gene Expression Matrix
        
        @param adata                  sc.AnnData object as input dataset
        @param hidden_stacks          List of number of nodes in each autencoder layer. The size of last element is 
                                      the dimension of the hidden distribution
        @param n_batch
        @param dispersion

        @param bias, 
               dropout_rate, 
               use_batch_norm, 
               use_layer_norm, 
               activation_fn          parameters for the linear autoencoders
        @param device                 device to be used for the VAE model. Default is CUDA
        """

        super(VAE, self).__init__()
        self.adata = adata
        self.in_dim = adata.shape[1] if adata else -1
        self.n_hidden = hidden_stacks[-1]
        self.n_latent = n_latent
        self._hidden_stacks = hidden_stacks 

        if n_batch > 0 and not batch_key:
            raise ValueError("Please provide a batch key if n_batch is greater than 0")
        if n_label > 0 and not label_key:
            raise ValueError("Please provide a label key if n_batch is greater than 0")

        self.label_key = label_key
        self.batch_key = batch_key
        self.n_batch = n_batch
        self.n_label = n_label
        self.new_adata_key = 'new_adata'
        self.new_adata_code = None
        self.log_variational = log_variational
        self.constrain_latent_embedding = constrain_latent_embedding
        self.device=device
        self.initialize_dataset()

        self.batch_embedding = batch_embedding
        if batch_embedding == "onehot":
            batch_hidden_dim = self.n_batch
        self.batch_hidden_dim = batch_hidden_dim
        self.inject_batch = inject_batch
        self.inject_label = inject_label
        self.encode_libsize = encode_libsize
        self.dispersion = dispersion
        self.use_attention = use_attention

        self.fcargs = dict(
            bias           = bias, 
            dropout_rate   = dropout_rate, 
            use_batch_norm = use_batch_norm, 
            use_layer_norm = use_layer_norm,
            activation_fn  = activation_fn,
            device         = device
        )
       
        

        #############################
        # Model Trainable Variables #
        #############################

        if self.dispersion == "gene":
            self.px_rate = torch.nn.Parameter(torch.randn(self.in_dim))
        elif self.dispersion == "gene-batch":
            self.px_rate = torch.nn.Parameter(torch.randn(self.in_dim, self.n_batch))
        elif self.dispersion == "gene-cell":
            pass 
        else:
            raise ValueError(
                "dispersion must be one of ['gene', 'gene-batch',"
                " 'gene-cell'], but input was "
                "{}.format(self.dispersion)"
            )

        ############
        # ENCODERS #
        ############

        self.encoder = SAE(
            self.in_dim if not self.encode_libsize else self.in_dim + 1,
            stacks = hidden_stacks,
            # n_cat_list = [self.n_batch] if self.n_batch > 0 else None,
            cat_dim = batch_hidden_dim,
            cat_embedding = batch_embedding,
            encode_only = True,
            **self.fcargs
        )

        # The latent cell representation z ~ Logisticnormal(0, I)
        self.z_mean_fc = nn.Linear(self.n_hidden, self.n_latent)
        self.z_var_fc = nn.Linear(self.n_hidden, self.n_latent)
        self.z_transformation = nn.Softmax(dim=-1)

        ############
        # DECODERS #
        ############

        if self.n_batch > 0 and self.n_label > 0 and inject_batch and inject_label:
            decoder_n_cat_list = [self.n_batch, self.n_label]
        elif self.n_batch > 0 and inject_batch:
            decoder_n_cat_list = [self.n_batch]
        elif self.n_label > 0 and inject_label:
            decoder_n_cat_list = [self.n_label]
        else:
            decoder_n_cat_list = None
        
        self.decoder_n_cat_list = decoder_n_cat_list

        self.decoder = FCLayer(
            in_dim = self.n_latent, 
            out_dim = self.n_hidden,
            n_cat_list = decoder_n_cat_list,
            cat_dim = batch_hidden_dim,
            cat_embedding = batch_embedding,
            use_layer_norm=False,
            use_batch_norm=True,
            dropout_rate=0,
            device=device
        )

        self.px_rna_rate_decoder = nn.Linear(self.n_hidden, self.in_dim)
        self.px_rna_scale_decoder = nn.Sequential(
            nn.Linear(self.n_hidden, self.in_dim),
            nn.Softmax(dim=-1)
        )
        self.px_rna_dropout_decoder = nn.Linear(self.n_hidden, self.in_dim)
        
        if self.n_label > 0:
            self.att = FCSelfAttention(dim = 1, dim_head = 5, n_heads = 5)
            self.fc = nn.Sequential(
                nn.Linear(self.n_latent, self.n_label),
            )

        self.to(device)

    def initialize_dataset(self):
        print("Initializing dataset into memory")
        if self.batch_key is not None:
            n_batch_ = len(np.unique(self.adata.obs[self.batch_key]))
            if self.n_batch != n_batch_:
                print(f"warning: the provided n_batch={self.n_batch} does not match the number of batch in the adata.")

                print(f"         setting n_batch to {n_batch_}")
                self.n_batch = n_batch_
            self.batch_category = pd.Categorical(self.adata.obs[self.batch_key])
            self.batch_category_summary = dict(Counter(self.batch_category))
            
        if self.label_key is not None:
            n_label_ = len(np.unique(list(filter(lambda x: x != self.new_adata_key, self.adata.obs[self.label_key]))))

            if self.n_label != n_label_:
                print(f"warning: the provided n_label={self.n_label} does not match the number of batch in the adata.")

                print(f"         setting n_label to {n_label_}")
                self.n_label = n_label_
            self.label_category = pd.Categorical(self.adata.obs[self.label_key])
            self.label_category_summary = dict(Counter(list(filter(lambda x: x != self.new_adata_key, self.label_category))))

            self.label_category_weight = len(self.label_category) / len(self.label_category.categories) * torch.tensor([self.label_category_summary[x] for x in list(filter(lambda x: x != self.new_adata_key, self.label_category.categories))], dtype=torch.float64).to(self.device)

            if self.new_adata_key in self.label_category.categories:
                self.new_adata_code = list(self.label_category.categories).index(self.new_adata_key)

        # X = self.adata.X.toarray() if issparse(self.adata.X) else self.adata.X
        X = self.adata.X

        self._n_record = X.shape[0]
        self._indices = np.array(list(range(self._n_record)))
        batch_categories, label_categories = None, None
        if self.batch_key is not None:
            if self.batch_key not in self.adata.obs.columns:
                raise ValueError(f"batch_key {self.batch_key} is not found in AnnData obs")
            batch_categories = np.array(self.batch_category.codes)
        if self.label_key is not None:
            if self.label_key not in self.adata.obs.columns:
                raise ValueError(f"label_key {self.label_key} is not found in AnnData obs")
            label_categories = np.array(self.label_category.codes)
        
        if self.constrain_latent_embedding and "X_pca" in self.adata.obsm.keys():
            P = self.adata.obsm["X_pca"]
            if batch_categories is not None and label_categories is not None:
                _dataset = list(zip(X, P, batch_categories, label_categories))
            elif batch_categories is not None:
                _dataset = list(zip(X, P, batch_categories))
            elif label_categories is not None:
                _dataset = list(zip(X, P, label_categories))
            else:
                _dataset = list(zip(X, P))
        else:
            if batch_categories is not None and label_categories is not None:
                _dataset = list(zip(X, batch_categories, label_categories))
            elif batch_categories is not None:
                _dataset = list(zip(X, batch_categories))
            elif label_categories is not None:
                _dataset = list(zip(X, label_categories))
            else:
                _dataset = X
        _shuffle_indices = list(range(len(_dataset)))
        np.random.shuffle(_shuffle_indices)
        self._dataset = np.array([_dataset[i] for i in _shuffle_indices])
        self._shuffle_indices = [x for x, _ in sorted(zip(range(len(_dataset)), _shuffle_indices), key=lambda x: x[1])]


    def as_dataloader(
        self, 
        subset_indices: Union[torch.tensor, np.ndarray] = None,
        n_per_batch: int = 128, 
        train_test_split: bool = False,
        random_seed: bool = 42,
        validation_split: bool = .2,
        shuffle: bool = True
    ):
        indices = subset_indices if subset_indices is not None else self._indices
        if shuffle:
            np.random.shuffle(indices)
        if train_test_split:
            np.random.seed(random_seed)
            split = int(np.floor(validation_split * self._n_record))
            if split % n_per_batch == 1:
                n_per_batch -= 1
            elif (self._n_record - split) % n_per_batch == 1:
                n_per_batch += 1
            train_indices, val_indices = indices[split:], indices[:split]
            train_sampler = SubsetRandomSampler(train_indices)
            valid_sampler = SubsetRandomSampler(val_indices)
            return DataLoader(indices, n_per_batch,  sampler = train_sampler), DataLoader(indices, n_per_batch, sampler = valid_sampler)
        if len(indices) % n_per_batch == 1:
            n_per_batch -= 1
        return DataLoader(indices, n_per_batch, shuffle = shuffle)


    def encode(self, X: torch.tensor, batch_index: torch.tensor = None, eps: float = 1e-4):
        # Encode for hidden space
        # if batch_index is not None and self.inject_batch:
        #    X = torch.hstack([X, batch_index])
        libsize = torch.log(X.sum(1))
        if self.log_variational:
            X = torch.log(1+X)
        q = self.encoder.encode(torch.hstack([X,libsize.unsqueeze(1)])) if self.encode_libsize else self.encoder.encode(X)
        q_mu = self.z_mean_fc(q)
        q_var = torch.exp(self.z_var_fc(q)) + eps
        z = Normal(q_mu, q_var.sqrt()).rsample()
        H = dict(
            q = q,
            q_mu = q_mu, 
            q_var = q_var,
            z = z
        )

        return H 

    def decode(self, 
        H: Mapping[str, torch.tensor], 
        lib_size:torch.tensor, 
        batch_index: torch.tensor = None, 
        label_index: torch.tensor = None,
        eps: float = 1e-4
    ):
        z = H["z"] # cell latent representation
        if batch_index is not None and label_index is not None and self.inject_batch and self.inject_label:
            z = torch.hstack([z, batch_index, label_index])
        elif batch_index is not None and self.inject_batch:
            z = torch.hstack([z, batch_index])
        elif label_index is not None and self.inject_label:
            z = torch.hstack([z, label_index])
        # eps to prevent numerical overflow and NaN gradient
        px = self.decoder(z)
        h = None

        if self.use_attention:
            h = rearrange(z, 'n (d z)-> n d z', z = 1)
            h = rearrange(self.att(h), 'n d z -> n (d z)')

        px_rna_scale = self.px_rna_scale_decoder(px) 
        px_rna_scale = px_rna_scale * lib_size.unsqueeze(1) 

        if self.dispersion == "gene-cell":
            px_rna_rate = self.px_rna_rate_decoder(px) ## In logits 
        elif self.dispersion == "gene-batch":
            px_rna_rate = F.linear(one_hot(batch_index, self.n_batch), self.px_rate)
        elif self.dispersion == "gene":
            px_rna_rate = self.px_rate

        px_rna_dropout = self.px_rna_dropout_decoder(px)  ## In logits
    
        R = dict(
            h = h,
            px = px,
            px_rna_scale = px_rna_scale,
            px_rna_rate = px_rna_rate,
            px_rna_dropout = px_rna_dropout
        )
        return R

    def forward(
        self,  
        X: torch.tensor, 
        lib_size: torch.tensor, 
        batch_index: torch.tensor = None,
        label_index: torch.tensor = None,
        P: torch.tensor = None
    ):
        H = self.encode(X, batch_index)
        q_mu = H["q_mu"]
        q_var = H["q_var"]
        mean = torch.zeros_like(q_mu)
        scale = torch.ones_like(q_var)
        kldiv_loss = kld(Normal(q_mu, q_var.sqrt()), Normal(mean, scale)).sum(dim = 1)
        prediction_loss = torch.tensor(0.)
        R = self.decode(H, lib_size, batch_index, label_index)
   
        
        reconstruction_loss = LossFunction.zinb_reconstruction_loss(
            X,
            mu = R['px_rna_scale'],
            theta = R['px_rna_rate'].exp(), 
            gate_logits = R['px_rna_dropout']
        )

        '''
        reconstruction_loss = -zinb(mu=R['px_rna_scale'], theta=R['px_rna_rate'].exp(), zi_logits=R['px_rna_dropout']).log_prob(X).sum(dim=-1)
        '''

        acc = torch.tensor(0.)

        if self.n_label > 0:
            criterion = nn.CrossEntropyLoss(weight=self.label_category_weight)
            if self.use_attention:
                prediction = self.fc(R['h'])
            else:
                prediction = self.fc(H['z'])

            if self.new_adata_code and self.new_adata_code in label_index:
                prediction_index = (label_index != self.new_adata_code).squeeze() 
                prediction_loss = criterion(prediction[prediction_index], one_hot(label_index[prediction_index], self.n_label))
            else:
                prediction_loss = criterion(prediction, one_hot(label_index, self.n_label))

        latent_constrain = torch.tensor(0.)
        if self.constrain_latent_embedding and P is not None:
            latent_constrain = nn.MSELoss()(P, q_mu)

        loss_record = {
            "reconstruction_loss": reconstruction_loss,
            "prediction_loss": prediction_loss,
            "kldiv_loss": kldiv_loss,
            "latent_constrain_loss": latent_constrain
        }
        return H, R, loss_record

    def calculate_metric(self, X, kl_weight, pred_weight):
        epoch_total_loss = 0
        epoch_reconstruction_loss = 0
        epoch_kldiv_loss = 0
        epoch_prediction_loss = 0
        b = 0
        with torch.no_grad():
            for b, X in enumerate(X):
                X = self._dataset[X.cpu().numpy()]
                batch_index, label_index = None, None
                if self.n_batch > 0 and self.n_label > 0:
                    if self.constrain_latent_embedding:
                        X, P, batch_index, label_index = get_k_elements(X,0), get_k_elements(X,1), get_k_elements(X,2), get_k_elements(X,3)
                    else:
                        X, batch_index, label_index = get_k_elements(X,0), get_k_elements(X,1), get_k_elements(X,2)
                elif self.n_batch > 0:
                    if self.constrain_latent_embedding:
                        X, P, batch_index = get_k_elements(X,0), get_k_elements(X,1),  get_k_elements(X,2)
                    else:
                        X, batch_index = get_k_elements(X,0), get_k_elements(X,1)
                elif self.n_label > 0:
                    if self.constrain_latent_embedding:
                        X, P, label_index = get_k_elements(X,0), get_k_elements(X,1), get_k_elements(X,2)
                    else: 
                        X, label_index = get_k_elements(X,0), get_k_elements(X,1)
                
                X = torch.tensor(np.vstack(list(map(lambda x: x.toarray() if issparse(x) else x, X))))


                if self.n_label > 0:
                    label_index = torch.tensor(label_index)
                    if not isinstance(label_index, torch.FloatTensor):
                        label_index = label_index.type(torch.FloatTensor)
                    label_index = label_index.to(self.device).unsqueeze(1)
                if self.n_batch > 0:
                    batch_index = torch.tensor(batch_index)
                    if not isinstance(batch_index, torch.FloatTensor):
                        batch_index = batch_index.type(torch.FloatTensor)
                    batch_index = batch_index.to(self.device).unsqueeze(1)  

                if not isinstance(X, torch.FloatTensor):
                    X = X.type(torch.FloatTensor)
                X = X.to(self.device)   

                
                lib_size = X.sum(1).to(self.device) 

                H, R, L = self.forward(X, lib_size, batch_index, label_index)

                reconstruction_loss = L['reconstruction_loss']
                prediction_loss = pred_weight * L['prediction_loss'] 
                kldiv_loss = kl_weight * L['kldiv_loss']    
                
                avg_reconstruction_loss = reconstruction_loss.sum()
                avg_kldiv_loss = kldiv_loss.sum()


                epoch_reconstruction_loss += avg_reconstruction_loss.item()
                epoch_kldiv_loss += avg_kldiv_loss.item()
                if self.n_label > 0:
                    epoch_prediction_loss += prediction_loss.sum().item() 
                
                loss = avg_reconstruction_loss + avg_kldiv_loss + prediction_loss.sum()

        epoch_total_loss =  epoch_reconstruction_loss  + epoch_kldiv_loss 
        return {
            "loss": loss,
            "epoch_reconstruction_loss":  epoch_reconstruction_loss / (b+1),
            "epoch_kldiv_loss": epoch_kldiv_loss / (b+1),
            "epoch_total_loss": epoch_total_loss / (b+1),
        }

    
    def fit(self,
            max_epoch:int = 35, 
            n_per_batch:int = 128,
            kl_weight: float = 1.,
            pred_weight: float = 1.,
            optimizer_parameters: Iterable = None,
            validation_split: float = .2,
            lr: bool = 1e-3,
            lr_schedule: bool = False,
            lr_factor: float = 0.6,
            lr_patience: int = 30,
            lr_threshold: float = 0.0,
            lr_min: float = 1e-6,
            n_epochs_kl_warmup: Union[int, None] = 400,
            weight_decay: float = 1e-6,
            random_seed: int = 12,
            subset_indices: Union[torch.tensor, np.ndarray] = None,
            pred_last_n_epoch: int = 10,
            pred_last_n_epoch_fconly: bool = False
        ):
        self.train()
        if n_epochs_kl_warmup:
            n_epochs_kl_warmup = min(max_epoch, n_epochs_kl_warmup)
            kl_warmup_gradient = kl_weight / n_epochs_kl_warmup
            kl_weight_max = kl_weight
            kl_weight = 0.

        if optimizer_parameters is None:
            optimizer = optim.AdamW(self.parameters(), lr, weight_decay=weight_decay)
        else:
            optimizer = optim.AdamW(optimizer_parameters, lr, weight_decay=weight_decay)

        scheduler = ReduceLROnPlateau(
            optimizer,
            patience=lr_patience,
            factor=lr_factor,
            threshold=lr_threshold,
            min_lr=lr_min,
            threshold_mode="abs",
            verbose=True,
        ) if lr_schedule else None 

        labels=None

        from tqdm import tqdm
        best_state_dict = None
        best_score = 0
        current_score = 0
        pbar = tqdm(range(max_epoch), desc="Epoch")
        for epoch in range(1, max_epoch+1):
            pbar.desc = "Epoch {}".format(epoch)
            epoch_total_loss = 0
            epoch_reconstruction_loss = 0
            epoch_kldiv_loss = 0
            epoch_prediction_loss = 0
            X_train, X_test = self.as_dataloader(
                n_per_batch=n_per_batch, 
                train_test_split = True, 
                validation_split = validation_split, 
                random_seed=random_seed,
                subset_indices=subset_indices
            )
            if self.n_label > 0 and epoch+1 == max_epoch - pred_last_n_epoch:
                print("saving transcriptome only state dict")
                self.gene_only_state_dict = deepcopy(self.state_dict())
                if  pred_last_n_epoch_fconly:
                    optimizer = optim.AdamW(chain(self.att.parameters(), self.fc.parameters()), lr, weight_decay=weight_decay)

            for b, X in enumerate(X_train):
                P = None
                X = self._dataset[X.cpu().numpy()]
                batch_index, label_index = None, None
                if self.n_batch > 0 or self.n_label > 0:
                    if not isinstance(X, Iterable) and len(X) > 1:
                        raise ValueError()
                    if self.n_batch > 0 and self.n_label > 0:
                        if self.constrain_latent_embedding:
                            X, P, batch_index, label_index = get_k_elements(X,0), get_k_elements(X,1), get_k_elements(X,2), get_k_elements(X,3)
                        else:
                            X, batch_index, label_index = get_k_elements(X,0), get_k_elements(X,1), get_k_elements(X,2)
                    elif self.n_batch > 0:
                        if self.constrain_latent_embedding:
                            X, P, batch_index = get_k_elements(X,0), get_k_elements(X,1),  get_k_elements(X,2)
                        else:
                            X, batch_index = get_k_elements(X,0), get_k_elements(X,1)
                    elif self.n_label > 0:
                        if self.constrain_latent_embedding:
                            X, P, label_index = get_k_elements(X,0), get_k_elements(X,1), get_k_elements(X,2)
                        else: 
                            X, label_index = get_k_elements(X,0), get_k_elements(X,1)


                X = torch.tensor(np.vstack(list(map(lambda x: x.toarray() if issparse(x) else x, X))))
                if self.constrain_latent_embedding:
                    P = torch.tensor(P).type(torch.FloatTensor).to(self.device)

                if self.n_label > 0:
                    label_index = torch.tensor(label_index)
                    if not isinstance(label_index, torch.FloatTensor):
                        label_index = label_index.type(torch.FloatTensor)
                    label_index = label_index.to(self.device).unsqueeze(1)
                if self.n_batch > 0:
                    batch_index = torch.tensor(batch_index)
                    if not isinstance(batch_index, torch.FloatTensor):
                        batch_index = batch_index.type(torch.FloatTensor)
                    batch_index = batch_index.to(self.device).unsqueeze(1)

                if not isinstance(X, torch.FloatTensor):
                    X = X.type(torch.FloatTensor)
                X = X.to(self.device)

                
                lib_size = X.sum(1).to(self.device)

                H, R, L = self.forward(X, lib_size, batch_index, label_index, P)
                

                reconstruction_loss = L['reconstruction_loss']
                prediction_loss = pred_weight * L['prediction_loss'] 
                kldiv_loss = kl_weight * L['kldiv_loss']    

                avg_reconstruction_loss = reconstruction_loss.sum()  / n_per_batch
                avg_kldiv_loss = kldiv_loss.sum()  / n_per_batch


                epoch_reconstruction_loss += avg_reconstruction_loss.item()
                epoch_kldiv_loss += avg_kldiv_loss.item()
                if self.n_label > 0:
                    epoch_prediction_loss += prediction_loss.sum().item()

                if epoch > max_epoch - pred_last_n_epoch:
                    loss = avg_reconstruction_loss + avg_kldiv_loss + prediction_loss.sum()
                else: 
                    loss = avg_reconstruction_loss + avg_kldiv_loss
    
                if self.constrain_latent_embedding:
                    loss += L['latent_constrain_loss']

                epoch_total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            loss_record = self.calculate_metric(X_test, kl_weight, pred_weight)
            if lr_schedule:
                scheduler.step(loss_record["epoch_total_loss"])
            pbar.set_postfix({
                    "loss": '{:.2e}'.format(loss_record["epoch_total_loss"]), 
                    'reconst': '{:.2e}'.format(loss_record["epoch_reconstruction_loss"]),
                    'kldiv': '{:.2e}'.format(loss_record["epoch_kldiv_loss"]),
                })
            pbar.update(1)
            if n_epochs_kl_warmup:
                kl_weight = min( kl_weight + kl_warmup_gradient, kl_weight_max)
            random_seed += 1
        if current_score < best_score:
            print("restoring state dict with best performance")
            self.load_state_dict(best_state_dict)
        pbar.close()
        self.trained_state_dict = deepcopy(self.state_dict())

    
    @torch.no_grad()
    def get_latent_embedding(self, latent_key: Literal["z", "q_mu"] = "q_mu") -> np.ndarray:
        X = self.as_dataloader(subset_indices = list(range(len(self._dataset))), shuffle=False)
        Zs = []
        for x in X:
            x = self._dataset[x.cpu().numpy()]
            batch_index = None
            label_index = None
            if self.n_batch > 0 or self.n_label > 0:
                if not isinstance(x, Iterable) and len(x) > 1:
                    raise ValueError()
                if self.n_batch > 0 and self.n_label > 0:
                    X, batch_index, label_index = get_k_elements(x,0), get_k_elements(x,1), get_k_elements(X,2)
                elif self.n_batch > 0:
                    X, batch_index = get_k_elements(x,0), get_k_elements(x,1)
                elif self.n_label > 0:
                    X, label_index = get_k_elements(x,0), get_k_elements(x,1)
                        
            if self.n_label > 0:
                label_index = torch.tensor(label_index)
                if not isinstance(label_index, torch.FloatTensor):
                    label_index = label_index.type(torch.FloatTensor)
                label_index = label_index.to(self.device).unsqueeze(1)
            if self.n_batch > 0:
                batch_index = torch.tensor(batch_index)
                if not isinstance(batch_index, torch.FloatTensor):
                    batch_index = batch_index.type(torch.FloatTensor)
                batch_index = batch_index.to(self.device).unsqueeze(1)
                
            X = torch.tensor(np.vstack(list(map(lambda x: x.toarray() if issparse(x) else x, X))))
            if not isinstance(X, torch.FloatTensor):
                X = X.type(torch.FloatTensor)
            X = X.to(self.device)
                    
            H = self.encode(X, batch_index if batch_index != None else None)
            Zs.append(H[latent_key].detach().cpu().numpy())
        return np.vstack(Zs)[self._shuffle_indices]

    @torch.no_grad()
    def get_normalized_expression(self, k = 'px_rna_scale') -> np.ndarray:
        X = self.as_dataloader(subset_indices = list(range(len(self._dataset))), shuffle=False)
        Zs = []
        for x in X:
            batch_index = None
            if self.n_batch > 0 or self.n_label > 0:
                if not isinstance(x, list) and len(x) > 1:
                    raise ValueError()
                if self.n_batch > 0 and self.n_label > 0:
                    x, batch_index, label_index = x
                    batch_index = batch_index.unsqueeze(1)
                    label_index = label_index.unsqueeze(1)
                elif self.n_batch > 0:
                    x, batch_index = x
                    batch_index = batch_index.unsqueeze(1)
                elif self.n_label > 0:
                    x, label_index = x
                    label_index = label_index.unsqueeze(1)

            _,R,_ = self.forward(
                x.to(self.device), 
                x.sum(1).to(self.device), 
                batch_index.to(self.device),
                label_index.to(self.device)
            )
            Zs.append(R[k].detach().cpu().numpy())
        return np.vstack(Zs)[self._shuffle_indices]


    def to(self, device:str):
        super(VAE, self).to(device)
        self.device=device 
        return self

    def transfer(self, new_adata: sc.AnnData, batch_key: str):
        new_batch_category = new_adata.obs[batch_key]
        original_batch_dim = self.batch_hidden_dim
        new_n_batch = len(np.unique(new_batch_category))

        if self.batch_embedding == "embedding":
            original_embedding_weight = self.decoder.cat_embedding[0].weight


        
        new_adata.obs[self.batch_key] = new_adata.obs[batch_key]
        
        original_batch_categories = self.batch_category.categories

        self.adata = sc.concat([self.adata, new_adata])
        self.initialize_dataset()
        if self.batch_embedding == "onehot":
            self.batch_hidden_dim = self.n_batch
            
        if self.n_label > 0 and self.inject_batch and self.inject_label:
            decoder_n_cat_list = [self.n_batch + new_n_batch, self.n_label]
        elif self.inject_batch:
            decoder_n_cat_list = [self.n_batch]
        elif self.n_label > 0 and self.inject_label:
            decoder_n_cat_list = [self.n_label]
        else:
            decoder_n_cat_list = None
        
        self.decoder_n_cat_list = decoder_n_cat_list
        
        original_weight = torch.tensor(self.decoder._fclayer[0].weight)

        self.decoder = FCLayer(
            in_dim = self.n_latent, 
            out_dim = self.n_hidden,
            n_cat_list = self.decoder_n_cat_list,
            cat_dim = self.batch_hidden_dim,
            cat_embedding = self.batch_embedding,
            use_layer_norm=False,
            use_batch_norm=True,
            dropout_rate=0,
            device=self.device
        )


        if self.batch_embedding == 'embedding':
            new_embedding = nn.Embedding(self.n_batch + new_n_batch, self.batch_hidden_dim).to(self.device)
            original_category_index = [list(self.batch_category.categories).index(x) for x in original_batch_categories]
            new_embedding_weight = new_embedding.weight.detach()
            new_embedding_weight[original_category_index] = original_embedding_weight.detach()
            new_embedding.weight = nn.Parameter(new_embedding_weight)
            new_embedding = new_embedding.to(self.device)
            self.decoder.cat_embedding[0] = new_embedding

        new_weight = torch.tensor(self.decoder._fclayer[0].weight)
        new_weight[:,:(self.n_latent + original_batch_dim)] = original_weight[:,:(self.n_latent + original_batch_dim)]
        self.decoder._fclayer[0].weight = nn.Parameter(new_weight)
        self.to(self.device)
