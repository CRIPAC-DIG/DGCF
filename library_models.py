# -*- coding: utf-8 -*
'''
This is a supporting library with the code of the model.

Paper: Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks. S. Kumar, X. Zhang, J. Leskovec. ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 2019. 
'''

from __future__ import division
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim
import numpy as np
import math, random
import sys
from collections import defaultdict
import os
import cPickle
import gpustat
from itertools import chain
from tqdm import tqdm, trange, tqdm_notebook, tnrange
import csv

PATH = "./"

try:
    get_ipython
    trange = tnrange
    tqdm = tqdm_notebook
except NameError:
    pass

total_reinitialization_count = 0

# A NORMALIZATION LAYER
class NormalLinear(nn.Linear):
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.normal_(0, stdv)
        if self.bias is not None:
            self.bias.data.normal_(0, stdv)


# THE DGCF MODULE
class DGCF(nn.Module):
    def __init__(self, args, num_features, num_users, num_items):
        super(DGCF,self).__init__()

        print "*** Initializing the DGCF model ***"
        self.modelname = args.model
        self.embedding_dim = args.embedding_dim
        self.num_users = num_users
        self.num_items = num_items
        self.user_static_embedding_size = num_users
        self.item_static_embedding_size = num_items
        self.adj = args.adj
        self.no_zero = args.no_zero
        self.no_first = args.no_first
        self.method = args.method
        self.length = args.length

        print "Initializing user and item embeddings"
        self.initial_user_embedding = nn.Parameter(torch.Tensor(args.embedding_dim))
        self.initial_item_embedding = nn.Parameter(torch.Tensor(args.embedding_dim))

        if self.adj:
            rnn_input_size_items = rnn_input_size_users = 2 * self.embedding_dim + 1 + num_features
        elif self.no_zero:
            rnn_input_size_items = rnn_input_size_users = self.embedding_dim + num_features
        elif self.no_first:
            rnn_input_size_items = rnn_input_size_users = self.embedding_dim + 1
        else:
            rnn_input_size_items = rnn_input_size_users = self.embedding_dim + 1 + num_features


        print "Initializing user and item RNNs"
        self.item_rnn = nn.RNNCell(rnn_input_size_users, self.embedding_dim)
        self.user_rnn = nn.RNNCell(rnn_input_size_items, self.embedding_dim)

        print "Initializing linear layers"
        self.linear_layer1 = nn.Linear(self.embedding_dim, 50)
        self.linear_layer2 = nn.Linear(50, 2)
        self.prediction_layer = nn.Linear(self.user_static_embedding_size + self.item_static_embedding_size + self.embedding_dim * 2, self.item_static_embedding_size + self.embedding_dim)
        self.embedding_layer = NormalLinear(1, self.embedding_dim)
        print "*** DGCF initialization complete ***\n\n"

        if self.adj or self.no_zero or self.no_first:
            print("Initializing aggregate layers")
            if self.method == 'mean' or self.method == 'attention':
                self.weigh_item = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
                self.weigh_user = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
                self.linear_three = nn.Linear(self.embedding_dim, 1, bias=False)
            elif self.method == 'gat':
                self.weigh_item = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
                self.weigh_user = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
                self.linear_three1 = nn.Linear(self.embedding_dim, 1, bias=False)
                self.linear_three2 = nn.Linear(self.embedding_dim, 1, bias=False)
            elif self.method == 'lstm':
                self.item_lstm = nn.LSTM(self.embedding_dim, self.embedding_dim, 1, batch_first=True)
                self.user_lstm = nn.LSTM(self.embedding_dim, self.embedding_dim, 1, batch_first=True)
                self.weigh_cen = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
                self.weigh_adj = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        
    # def forward(self, user_embeddings, item_embeddings, timediffs=None, features=None, select=None):
    #     if select == 'item_update':
    #         input1 = torch.cat([user_embeddings, timediffs, features], dim=1)
    #         item_embedding_output = self.item_rnn(input1, item_embeddings)
    #         return F.normalize(item_embedding_output)
    #
    #     elif select == 'user_update':
    #         input2 = torch.cat([item_embeddings, timediffs, features], dim=1)
    #         user_embedding_output = self.user_rnn(input2, user_embeddings)
    #         return F.normalize(user_embedding_output)
    #
    #     elif select == 'project':
    #         user_projected_embedding = self.context_convert(user_embeddings, timediffs, features)
    #         #user_projected_embedding = torch.cat([input3, item_embeddings], dim=1)
    #         return user_projected_embedding
    def forward(self, user_embeddings, item_embeddings, timediffs=None,  features=None, adj_embeddings=None, select=None):
        # if select == 'item_update':
        #     if self.adj:
        #         input1 = torch.cat([user_embeddings, timediffs, features, adj_embeddings], dim=1)
        #     else:
        #         input1 = torch.cat([user_embeddings, timediffs, features], dim=1)
        #     item_embedding_output = self.item_rnn(input1, item_embeddings)
        #     return F.normalize(item_embedding_output)
        #
        # elif select == 'user_update':
        #     if self.adj:
        #         input2 = torch.cat([item_embeddings, timediffs, features, adj_embeddings], dim=1)
        #     else:
        #         input2 = torch.cat([item_embeddings, timediffs, features], dim=1)
        #     user_embedding_output = self.user_rnn(input2, user_embeddings)
        #     return F.normalize(user_embedding_output)
        if select == 'item_update':
            if self.adj:
                # item_all.append(adj_embeddings)
                input1 = torch.cat([user_embeddings, timediffs, features, adj_embeddings], dim=1)
                item_embedding_output = self.item_rnn(input1, item_embeddings)
            elif self.no_zero:
                input1 = torch.cat([features, adj_embeddings], dim=1)
                item_embedding_output = self.item_rnn(input1, user_embeddings)
            elif self.no_first:
                input1 = torch.cat([timediffs, adj_embeddings], dim=1)
                item_embedding_output = self.item_rnn(input1, item_embeddings)
            else:
                input1 = torch.cat([user_embeddings, timediffs, features], dim=1)
                item_embedding_output = self.item_rnn(input1, item_embeddings)
            return F.normalize(item_embedding_output)

        elif select == 'user_update':
            if self.adj:
                input2 = torch.cat([item_embeddings, timediffs, features, adj_embeddings], dim=1)
                user_embedding_output = self.user_rnn(input2, user_embeddings)
            elif self.no_zero:
                input2 = torch.cat([features, adj_embeddings], dim=1)
                user_embedding_output = self.user_rnn(input2, item_embeddings)
            elif self.no_first:
                input2 = torch.cat([timediffs, adj_embeddings], dim=1)
                user_embedding_output = self.user_rnn(input2, user_embeddings)
            else:
                input2 = torch.cat([item_embeddings, timediffs, features], dim=1)
                user_embedding_output = self.user_rnn(input2, user_embeddings)
            return F.normalize(user_embedding_output)


        elif select == 'project':
            user_projected_embedding = self.context_convert(user_embeddings, timediffs, features)
            #user_projected_embedding = torch.cat([input3, item_embeddings], dim=1)
            return user_projected_embedding


    def aggregate_attention(self, embeddings, length_mask, max_length, center_embedding, select=None, train=True):
        mask = torch.arange(max_length)[None, :] < length_mask[:, None]
        if select == 'user_upate':
            user_em = self.weigh_user(center_embedding)
            item_em = self.weigh_item(embeddings)
        else:
            user_em = self.weigh_item(center_embedding)
            item_em = self.weigh_user(embeddings)
        alpha = self.linear_three(torch.sigmoid(item_em + user_em.unsqueeze(1)))
        fin_em = torch.sum(alpha*embeddings*mask.view(mask.shape[0], -1, 1).float().cuda(), 1)
        return fin_em

    def aggregate_gat(self, embeddings, length_mask, max_length, center_embedding, select=None, train=True):
        mask = torch.arange(max_length)[None, :] < length_mask[:, None]
        if select == 'user_upate':
            user_em = self.weigh_user(center_embedding)
            item_em = self.weigh_item(embeddings)
        else:
            user_em = self.weigh_item(center_embedding)
            item_em = self.weigh_user(embeddings)
        #alpha = torch.nn.LeakyReLU()(self.linear_three(item_em + user_em.unsqueeze(1))).squeeze(-1)
        alpha = torch.nn.LeakyReLU()(self.linear_three1(item_em).squeeze(-1) + self.linear_three2(user_em))
        zero_vec = -9e15 * torch.ones_like(mask).float().cuda()
        attention = torch.softmax(torch.where(mask.cuda()>0, alpha, zero_vec), dim=1).unsqueeze(-1)
        #print(torch.where(mask.cuda() > 0, alpha, zero_vec).shape, alpha.shape, attention.shape, mask.shape)
        fin_em = torch.sum(attention * embeddings * mask.view(mask.shape[0], -1, 1).float().cuda(), 1)
        return fin_em

    def aggregate_lstm(self, embeddings, length_mask, max_length, center_embedding, select=None, train=True):
        #pack = nn_utils.rnn.pack_padded_sequence(embeddings, length_mask, batch_first=True)
        #h0 = Variable(torch.randn(1, embeddings.shape[0],self.embedding_dim))
        if select == 'user_upate':
            out, _ = self.user_lstm(embeddings)
        else:
            out, _ = self.item_lstm(embeddings)
        lstm_em = out[torch.arange(embeddings.shape[0]), length_mask-1, :]
        fin_em = self.weigh_cen(center_embedding) + self.weigh_adj(lstm_em)
        return fin_em

    def aggregate_mean(self, embeddings, length_mask, max_length, center_embedding, select=None):
        mask = torch.arange(max_length)[None, :] < length_mask[:, None]
        if select == 'user_upate':
            em = self.weigh_item(embeddings)
        else:
            em = self.weigh_user(embeddings)
        em_mean = torch.div(torch.sum(em.mul(mask.unsqueeze(2).float().cuda()), 1) + center_embedding, length_mask.unsqueeze(1).float().cuda()+1)
        return em_mean


    def context_convert(self, embeddings, timediffs, features):
        new_embeddings = embeddings * (1 + self.embedding_layer(timediffs))
        return new_embeddings

    def predict_label(self, user_embeddings):
        X_out = nn.ReLU()(self.linear_layer1(user_embeddings))
        X_out = self.linear_layer2(X_out)
        return X_out

    def predict_item_embedding(self, user_embeddings):
        X_out = self.prediction_layer(user_embeddings)
        return X_out

def adj_pad(adj_seq):
    adjs = []
    length = [len(seq) for seq in adj_seq]
    max_length = max(length)
    for seq in adj_seq:
        adjs.append(list(seq) + (max_length - len(seq))*[0])
    return adjs, length, max_length


def adj_sample(adj_seq, sam_l):
    adjs = []
    length = [len(seq[:sam_l]) for seq in adj_seq]
    max_length = max(length)
    for seq in adj_seq:
        adjs.append(seq[::-1][:sam_l] + (max_length - len(seq[:sam_l]))*[0])
    return adjs, length, max_length



# INITIALIZE T-BATCH VARIABLES
def reinitialize_tbatches():
    global current_tbatches_interactionids, current_tbatches_user, current_tbatches_item, current_tbatches_timestamp, current_tbatches_feature, current_tbatches_label, current_tbatches_previous_item
    global tbatchid_user, tbatchid_item, current_tbatches_user_timediffs, current_tbatches_item_timediffs, current_tbatches_user_timediffs_next
    global current_tbatches_user_adj, current_tbatches_item_adj  # item和user的邻居：item的邻居是购买过item的user，user的邻居是买过的item

    # list of users of each tbatch up to now
    current_tbatches_interactionids = defaultdict(list)
    current_tbatches_user = defaultdict(list)
    current_tbatches_item = defaultdict(list)
    current_tbatches_timestamp = defaultdict(list)
    current_tbatches_feature = defaultdict(list)
    current_tbatches_label = defaultdict(list)
    current_tbatches_previous_item = defaultdict(list)
    current_tbatches_user_timediffs = defaultdict(list)
    current_tbatches_item_timediffs = defaultdict(list)
    current_tbatches_user_timediffs_next = defaultdict(list)

    current_tbatches_user_adj = defaultdict(list)
    current_tbatches_item_adj = defaultdict(list)

    # the latest tbatch a user is in
    tbatchid_user = defaultdict(lambda: -1)

    # the latest tbatch a item is in
    tbatchid_item = defaultdict(lambda: -1)

    global total_reinitialization_count
    total_reinitialization_count +=1


# CALCULATE LOSS FOR THE PREDICTED USER STATE 
def calculate_state_prediction_loss(model, tbatch_interactionids, user_embeddings_time_series, y_true, loss_function):
    # PREDCIT THE LABEL FROM THE USER DYNAMIC EMBEDDINGS
    prob = model.predict_label(user_embeddings_time_series[tbatch_interactionids,:])
    y = Variable(torch.LongTensor(y_true).cuda()[tbatch_interactionids])
    
    loss = loss_function(prob, y)

    return loss


# SAVE TRAINED MODEL TO DISK
def save_model(model, optimizer, args, epoch, user_embeddings, item_embeddings, train_end_idx, user_adj, item_adj,
               user_embeddings_time_series=None, item_embeddings_time_series=None, path=PATH):
    print "*** Saving embeddings and model ***"
    state = {
            'user_embeddings': user_embeddings.data.cpu().numpy(),
            'item_embeddings': item_embeddings.data.cpu().numpy(),
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'train_end_idx': train_end_idx,
            'user_adj': user_adj,
            'item_adj': item_adj
            }

    if user_embeddings_time_series is not None:
        state['user_embeddings_time_series'] = user_embeddings_time_series.data.cpu().numpy()
        state['item_embeddings_time_series'] = item_embeddings_time_series.data.cpu().numpy()
    if not args.length:
        directory = os.path.join(path, 'saved_models/%s/' % args.network)
    else:
        directory = os.path.join(path, 'saved_models_%d/%s/' % (args.length, args.network))
    if not os.path.exists(directory):
        os.makedirs(directory)
    if model.adj:
        if args.l2u == 1.0 and args.l2i == 1.0:
            filename = os.path.join(directory,
                                    "adj_checkpoint.%s.%s.ep%d.tp%.1f.pth.tar" % (
                                    args.model, args.method, epoch, args.train_proportion))
        else:
            filename = os.path.join(directory,
                                    "adj_checkpoint.%s.%s.user%.1f.item%.1f.ep%d.tp%.1f.pth.tar" % (args.model, args.method, args.l2u, args.l2i,  epoch, args.train_proportion))
    elif model.no_zero:
        if args.l2u == 1.0 and args.l2i == 1.0:
            filename = os.path.join(directory,
                                    "adj_checkpoint.%s.%s.%s.ep%d.tp%.1f.pth.tar" % (
                                        args.model, args.method, 'no_zero', epoch, args.train_proportion))
        else:
            filename = os.path.join(directory,
                                    "adj_checkpoint.%s.%s.%s.user%.1f.item%.1f.ep%d.tp%.1f.pth.tar" % (
                                    args.model, args.method, 'no_zero', args.l2u, args.l2i,  epoch, args.train_proportion))
    elif model.no_first:
        if args.l2u == 1.0 and args.l2i == 1.0:
            filename = os.path.join(directory,
                                    "adj_checkpoint.%s.%s.%s.ep%d.tp%.1f.pth.tar" % (
                                        args.model, args.method, 'no_first', epoch, args.train_proportion))
        else:
            filename = os.path.join(directory,
                                    "adj_checkpoint.%s.%s.%s.user%.1f.item%.1f.ep%d.tp%.1f.pth.tar" % (
                                        args.model, args.method, 'no_first', args.l2u, args.l2i, epoch, args.train_proportion))
    else:
        if args.l2u == 1.0 and args.l2i == 1.0:
            filename = os.path.join(directory,
                                    "checkpoint.%s.ep%d.tp%.1f.pth.tar" % (args.model, epoch, args.train_proportion))
        else:
            filename = os.path.join(directory, "checkpoint.%s.user%.1f.item%.1f.ep%d.tp%.1f.pth.tar" % (args.model, epoch, args.l2u, args.l2i, args.train_proportion))
    torch.save(state, filename)
    print "*** Saved embeddings and model to file: %s ***\n\n" % filename


# LOAD PREVIOUSLY TRAINED AND SAVED MODEL
def load_model(model, optimizer, args, epoch):
    modelname = args.model
    if not args.length:
        dic = 'saved_models/'
    else:
        dic = 'saved_models_%s/' % args.length
    if model.adj:
        if args.l2u == 1.0 and args.l2i == 1.0:
            filename = PATH + dic +"%s/adj_checkpoint.%s.%s.ep%d.tp%.1f.pth.tar" % (
                args.network, modelname, model.method, epoch, args.train_proportion)
        else:
            filename = PATH + dic + "%s/adj_checkpoint.%s.%s.user%.1f.item%.1f.ep%d.tp%.1f.pth.tar" % (
            args.network, modelname, model.method, args.l2u, args.l2i, epoch, args.train_proportion)
    elif model.no_zero:
        if args.l2u == 1.0 and args.l2i == 1.0:
            filename = PATH + dic + "%s/adj_checkpoint.%s.%s.%s.ep%d.tp%.1f.pth.tar" % (
                args.network, modelname, model.method, 'no_zero', epoch, args.train_proportion)
        else:
            filename = PATH + dic + "%s/adj_checkpoint.%s.%s.%s.user%.1f.item%.1f.ep%d.tp%.1f.pth.tar" % (
                args.network, modelname, model.method, 'no_zero', args.l2u, args.l2i, epoch, args.train_proportion)
    elif model.no_first:
        if args.l2u == 1.0 and args.l2i == 1.0:
            filename = PATH + dic + "%s/adj_checkpoint.%s.%s.%s.ep%d.tp%.1f.pth.tar" % (
                args.network, modelname, model.method, 'no_first', epoch, args.train_proportion)
        else:
            filename = PATH + dic + "%s/adj_checkpoint.%s.%s.%s.user%.1f.item%.1f.ep%d.tp%.1f.pth.tar" % (
                args.network, modelname, model.method, 'no_first', args.l2u, args.l2i, epoch, args.train_proportion)
    else:
        if args.l2u == 1.0 and args.l2i == 1.0:
            filename = PATH + dic + "%s/checkpoint.%s.ep%d.tp%.1f.pth.tar" % (
            args.network, modelname, epoch, args.train_proportion)
        else:
            filename = PATH + dic + "%s/checkpoint.%s.user%.1f.item%.1f.ep%d.tp%.1f.pth.tar" % (args.network, modelname, args.l2u, args.l2i, epoch, args.train_proportion)
    checkpoint = torch.load(filename)
    print "Loading saved embeddings and model: %s" % filename
    args.start_epoch = checkpoint['epoch']
    user_embeddings = Variable(torch.from_numpy(checkpoint['user_embeddings']).cuda())
    item_embeddings = Variable(torch.from_numpy(checkpoint['item_embeddings']).cuda())
    if model.adj or model.no_zero or model.no_first:
        user_adj = checkpoint['user_adj']
        item_adj = checkpoint['item_adj']
    else:
        user_adj = None
        item_adj = None
    try:
        train_end_idx = checkpoint['train_end_idx'] 
    except KeyError:
        train_end_idx = None

    try:
        user_embeddings_time_series = Variable(torch.from_numpy(checkpoint['user_embeddings_time_series']).cuda())
        item_embeddings_time_series = Variable(torch.from_numpy(checkpoint['item_embeddings_time_series']).cuda())
    except:
        user_embeddings_time_series = None
        item_embeddings_time_series = None

    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    return [model, optimizer, user_embeddings, item_embeddings, user_adj, item_adj,
            user_embeddings_time_series, item_embeddings_time_series, train_end_idx]


# SET USER AND ITEM EMBEDDINGS TO THE END OF THE TRAINING PERIOD 
def set_embeddings_training_end(user_embeddings, item_embeddings, user_embeddings_time_series, item_embeddings_time_series, user_data_id, item_data_id, train_end_idx):
    userid2lastidx = {}
    for cnt, userid in enumerate(user_data_id[:train_end_idx]):
        userid2lastidx[userid] = cnt
    itemid2lastidx = {}
    for cnt, itemid in enumerate(item_data_id[:train_end_idx]):
        itemid2lastidx[itemid] = cnt

    try:
        embedding_dim = user_embeddings_time_series.size(1)
    except:
        embedding_dim = user_embeddings_time_series.shape[1]
    for userid in userid2lastidx:
        user_embeddings[userid, :embedding_dim] = user_embeddings_time_series[userid2lastidx[userid]]
    for itemid in itemid2lastidx:
        item_embeddings[itemid, :embedding_dim] = item_embeddings_time_series[itemid2lastidx[itemid]]

    user_embeddings.detach_()
    item_embeddings.detach_()


# SELECT THE GPU WITH MOST FREE MEMORY TO SCHEDULE JOB 
def select_free_gpu():
    mem = []
    gpus = list(set([0,1,2,3,4,5,6,7]))
    for i in gpus:
        gpu_stats = gpustat.GPUStatCollection.new_query()
        mem.append(gpu_stats.jsonify()["gpus"][i]["memory.used"])
    return str(gpus[np.argmin(mem)])

