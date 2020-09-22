# -*- coding: utf-8 -*
'''
This code trains the DGCF model for the given dataset.
The task is: interaction prediction.

How to run: 
$ python DGCF.py --network reddit --model DGCF --epochs 50

Reference Paper: Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks. S. Kumar, X. Zhang, J. Leskovec. ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 2019.
'''
from library_data import *
import library_models as lib
from library_models import *
from IPython import embed

# INITIALIZE PARAMETERS
parser = argparse.ArgumentParser()
parser.add_argument('--network', required=True, help='Name of the network/dataset')
parser.add_argument('--model', default="DGCF", help='Model name to save output in file')
parser.add_argument('--adj', action='store_true', help='The second order relationship')
parser.add_argument('--no_zero', action='store_true', help='The zero order relationship')
parser.add_argument('--no_first', action='store_true', help='The first order relationship')
parser.add_argument('--method', default="mean", help='The way of aggregate adj')
parser.add_argument('--length', type=int, default=None, help='sample length')
parser.add_argument('--l2u', type=float, default=1.0, help='regular coefficient of user')
parser.add_argument('--l2i', type=float, default=1.0, help='regular coefficient of item')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--gpu', default=-1, type=int, help='ID of the gpu to run on. If set to -1 (default), the GPU with most free memory will be chosen.')
parser.add_argument('--epochs', default=50, type=int, help='Number of epochs to train the model')
parser.add_argument('--embedding_dim', default=128, type=int, help='Number of dimensions of the dynamic embedding')
parser.add_argument('--span_num', default=500, type=int, help='time span number')
parser.add_argument('--train_proportion', default=0.8, type=float, help='Fraction of interactions (from the beginning) that are used for training.The next 10% are used for validation and the next 10% for testing')
parser.add_argument('--state_change', default=False, type=bool, help='True if training with state change of users along with interaction prediction. False otherwise. By default, set to True.')
args = parser.parse_args()
print(args)

args.datapath = "data/%s.csv" % args.network 
if args.train_proportion > 0.8:
    sys.exit('Training sequence proportion cannot be greater than 0.8.')

# SET GPU
if args.gpu == -1:
    args.gpu = select_free_gpu()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# LOAD DATA
[user2id, user_sequence_id, user_timediffs_sequence, user_previous_itemid_sequence,
 item2id, item_sequence_id, item_timediffs_sequence, 
 timestamp_sequence, feature_sequence, y_true] = load_network(args)
num_interactions = len(user_sequence_id)
num_users = len(user2id) 
num_items = len(item2id) + 1 # one extra item for "none-of-these"
num_features = len(feature_sequence[0])
true_labels_ratio = len(y_true)/(1.0+sum(y_true)) # +1 in denominator in case there are no state change labels, which will throw an error.

embed()
print "*** Network statistics:\n  %d users\n  %d items\n  %d interactions\n  %d/%d true labels ***\n\n" % (num_users, num_items, num_interactions, sum(y_true), len(y_true))

# SET TRAINING, VALIDATION, TESTING, and TBATCH BOUNDARIES
train_end_idx = validation_start_idx = int(num_interactions * args.train_proportion)
test_start_idx = int(num_interactions * (args.train_proportion+0.1))
test_end_idx = int(num_interactions * (args.train_proportion+0.2))

# SET BATCHING TIMESPAN
'''
Timespan is the frequency at which the batches are created and the DGCF model is trained.
As the data arrives in a temporal order, the interactions within a timespan are added into batches (using the T-batch algorithm).
The batches are then used to train DGCF.
Longer timespans mean more interactions are processed and the training time is reduced, however it requires more GPU memory.
Longer timespan leads to less frequent model updates.
'''
timespan = timestamp_sequence[-1] - timestamp_sequence[0]   #总的时间间隔
tbatch_timespan = timespan / args.span_num                           #

# INITIALIZE MODEL AND PARAMETERS
model = DGCF(args, num_features, num_users, num_items).cuda()
weight = torch.Tensor([1, true_labels_ratio]).cuda()
crossEntropyLoss = nn.CrossEntropyLoss(weight=weight)
MSELoss = nn.MSELoss()

# INITIALIZE EMBEDDING
initial_user_embedding = nn.Parameter(F.normalize(torch.rand(args.embedding_dim).cuda(), dim=0)) # the initial user and item embeddings are learned during training as well
initial_item_embedding = nn.Parameter(F.normalize(torch.rand(args.embedding_dim).cuda(), dim=0))
model.initial_user_embedding = initial_user_embedding
model.initial_item_embedding = initial_item_embedding

user_embeddings = initial_user_embedding.repeat(num_users, 1) # initialize all users to the same embedding
item_embeddings = initial_item_embedding.repeat(num_items, 1) # initialize all items to the same embedding
item_embedding_static = Variable(torch.eye(num_items).cuda()) # one-hot vectors for static embeddings
user_embedding_static = Variable(torch.eye(num_users).cuda()) # one-hot vectors for static embeddings

# INITIALIZE MODEL
learning_rate = 1e-3
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=args.l2)

# RUN THE DGCF MODEL
'''
THE MODEL IS TRAINED FOR SEVERAL EPOCHS. IN EACH EPOCH, DGCF USES THE TRAINING SET OF INTERACTIONS TO UPDATE ITS PARAMETERS.
'''
print "*** Training the DGCF model for %d epochs ***" % args.epochs
#with trange(args.epochs) as progress_bar1:
user_adj = None
item_adj = None
for ep in range(args.epochs):
    #progress_bar1.set_description('Epoch %d of %d' % (ep, args.epochs))

    # INITIALIZE EMBEDDING TRAJECTORY STORAGE
    user_embeddings_timeseries = Variable(torch.Tensor(num_interactions, args.embedding_dim).cuda())
    item_embeddings_timeseries = Variable(torch.Tensor(num_interactions, args.embedding_dim).cuda())

    optimizer.zero_grad()
    reinitialize_tbatches()
    total_loss, loss, total_interaction_count = 0, 0, 0

    tbatch_start_time = None
    tbatch_to_insert = -1
    tbatch_full = False
    if args.adj or args.no_zero or args.no_first:
        if not args.length:
            user_adj = defaultdict(set)  # 每个user的邻居
            item_adj = defaultdict(set)  # 每个item的邻居
        else:
            user_adj = defaultdict(list)  # 每个user的邻居
            item_adj = defaultdict(list)  # 每个item的邻居

    # TRAIN TILL THE END OF TRAINING INTERACTION IDX
    #with trange(train_end_idx) as progress_bar2:
    for j in range(train_end_idx):
        #progress_bar2.set_description('Processed %dth interactions' % j)

        # READ INTERACTION J
        userid = user_sequence_id[j]
        itemid = item_sequence_id[j]
        feature = feature_sequence[j]
        user_timediff = user_timediffs_sequence[j]
        item_timediff = item_timediffs_sequence[j]
        if args.adj or args.no_zero or args.no_first:
            # 计算user和item的邻居：
            if not args.length:
                user_adj[userid].add(itemid)  # 实时更新user和item的邻居 user_adj is dic, key is user_id ,value is item_id
                item_adj[itemid].add(userid)
            else:
                user_adj[userid].append(itemid)  # 实时更新user和item的邻居 user_adj is dic, key is user_id ,value is item_id
                item_adj[itemid].append(userid)


        # CREATE T-BATCHES: ADD INTERACTION J TO THE CORRECT T-BATCH
        tbatch_to_insert = max(lib.tbatchid_user[userid], lib.tbatchid_item[itemid]) + 1   #tbatch_user:user_id所在的batch  tbatch_item:item_id所在的batch
        lib.tbatchid_user[userid] = tbatch_to_insert                                       #为了保证同一个batch没有相同的item和user
        lib.tbatchid_item[itemid] = tbatch_to_insert

        lib.current_tbatches_user[tbatch_to_insert].append(userid)     #每个batch中的user/item/feature/interactions/timediffs....都用list存
        lib.current_tbatches_item[tbatch_to_insert].append(itemid)
        lib.current_tbatches_feature[tbatch_to_insert].append(feature)
        lib.current_tbatches_interactionids[tbatch_to_insert].append(j)
        lib.current_tbatches_user_timediffs[tbatch_to_insert].append(user_timediff)
        lib.current_tbatches_item_timediffs[tbatch_to_insert].append(item_timediff)
        lib.current_tbatches_previous_item[tbatch_to_insert].append(user_previous_itemid_sequence[j])

        if args.adj or args.no_zero or args.no_first:
            # batch中每个user和item的邻居
            lib.current_tbatches_user_adj[tbatch_to_insert].append(user_adj[userid])  # item
            lib.current_tbatches_item_adj[tbatch_to_insert].append(item_adj[itemid])  # user

        timestamp = timestamp_sequence[j]
        if tbatch_start_time is None:
            tbatch_start_time = timestamp

        # AFTER ALL INTERACTIONS IN THE TIMESPAN ARE CONVERTED TO T-BATCHES,
        # FORWARD PASS TO CREATE EMBEDDING TRAJECTORIES AND CALCULATE PREDICTION LOSS
        # after all interactions in the timespan are converted to t-batchs,
        # forward pass to crate embedding trajectories and calculate prediction loss
        if timestamp - tbatch_start_time > tbatch_timespan:
            tbatch_start_time = timestamp # RESET START TIME FOR THE NEXT TBATCHES

            # ITERATE OVER ALL T-BATCHES
            #with trange(len(lib.current_tbatches_user)) as progress_bar3:   #len(lib.current_tbatches_user):当前batch的数量
            for i in range(len(lib.current_tbatches_user)):
                #progress_bar3.set_description('Processed %d of %d T-batches ' % (i, len(lib.current_tbatches_user)))

                total_interaction_count += len(lib.current_tbatches_interactionids[i])

                # LOAD THE CURRENT TBATCH
                tbatch_userids = torch.LongTensor(lib.current_tbatches_user[i]).cuda() # Recall "lib.current_tbatches_user[i]" has unique elements
                tbatch_itemids = torch.LongTensor(lib.current_tbatches_item[i]).cuda() # Recall "lib.current_tbatches_item[i]" has unique elements
                tbatch_interactionids = torch.LongTensor(lib.current_tbatches_interactionids[i]).cuda()
                feature_tensor = Variable(torch.Tensor(lib.current_tbatches_feature[i]).cuda()) # Recall "lib.current_tbatches_feature[i]" is list of list, so "feature_tensor" is a 2-d tensor
                user_timediffs_tensor = Variable(torch.Tensor(lib.current_tbatches_user_timediffs[i]).cuda()).unsqueeze(1)
                item_timediffs_tensor = Variable(torch.Tensor(lib.current_tbatches_item_timediffs[i]).cuda()).unsqueeze(1)
                tbatch_itemids_previous = torch.LongTensor(lib.current_tbatches_previous_item[i]).cuda()
                item_embedding_previous = item_embeddings[tbatch_itemids_previous,:]

                # PROJECT USER EMBEDDING TO CURRENT TIME
                user_embedding_input = user_embeddings[tbatch_userids,:]
                user_projected_embedding = model.forward(user_embedding_input, item_embedding_previous, timediffs=user_timediffs_tensor, features=feature_tensor, select='project')
                user_item_embedding = torch.cat([user_projected_embedding, item_embedding_previous, item_embedding_static[tbatch_itemids_previous,:], user_embedding_static[tbatch_userids,:]], dim=1)

                # PREDICT NEXT ITEM EMBEDDING
                predicted_item_embedding = model.predict_item_embedding(user_item_embedding)

                # CALCULATE PREDICTION LOSS
                item_embedding_input = item_embeddings[tbatch_itemids,:]
                loss += MSELoss(predicted_item_embedding, torch.cat([item_embedding_input, item_embedding_static[tbatch_itemids,:]], dim=1).detach())

                # UPDATE DYNAMIC EMBEDDINGS AFTER INTERACTION
                if args.adj or args.no_zero or args.no_first:
                    if not args.length:
                        user_adj_, user_length_mask, user_max_length = adj_pad(lib.current_tbatches_user_adj[i])
                        item_adj_, item_length_mask, item_max_length = adj_pad(lib.current_tbatches_item_adj[i])
                    else:
                        user_adj_, user_length_mask, user_max_length = adj_sample(lib.current_tbatches_user_adj[i],
                                                                               args.length)
                        item_adj_, item_length_mask, item_max_length = adj_sample(lib.current_tbatches_item_adj[i],
                                                                               args.length)
                    user_adj_em = item_embeddings[torch.LongTensor(user_adj_).cuda(), :]
                    item_adj_em = user_embeddings[torch.LongTensor(item_adj_).cuda(), :]
                    if model.method == 'mean':
                        # user_adj_embedding = model.aggregate(item_embeddings, lib.current_tbatches_user_adj[i], select='user_update')
                        # item_adj_embedding = model.aggregate(user_embeddings, lib.current_tbatches_item_adj[i], select='item_update')
                        user_adj_embedding = model.aggregate_mean(user_adj_em, torch.LongTensor(user_length_mask),
                                                                  user_max_length, user_embedding_input,
                                                                  select='user_update')
                        item_adj_embedding = model.aggregate_mean(item_adj_em, torch.LongTensor(item_length_mask),
                                                                  item_max_length, item_embedding_input,
                                                                  select='item_update')
                    elif model.method == 'attention':
                        user_adj_embedding = model.aggregate_attention(user_adj_em, torch.LongTensor(user_length_mask),
                                                                       user_max_length, user_embedding_input,
                                                                       select='user_update')
                        item_adj_embedding = model.aggregate_attention(item_adj_em, torch.LongTensor(item_length_mask),
                                                                       item_max_length, item_embedding_input,
                                                                       select='item_update')
                    elif model.method == 'gat':
                        user_adj_embedding = model.aggregate_gat(user_adj_em, torch.LongTensor(user_length_mask),
                                                                       user_max_length, user_embedding_input,
                                                                       select='user_update')
                        item_adj_embedding = model.aggregate_gat(item_adj_em, torch.LongTensor(item_length_mask),
                                                                       item_max_length, item_embedding_input,
                                                                       select='item_update')
                    elif model.method == 'lstm':
                        user_adj_embedding = model.aggregate_lstm(user_adj_em, torch.LongTensor(user_length_mask),
                                                                 user_max_length, user_embedding_input,
                                                                 select='user_update')
                        item_adj_embedding = model.aggregate_lstm(item_adj_em, torch.LongTensor(item_length_mask),
                                                                 item_max_length, item_embedding_input,
                                                                 select='item_update')
                else:
                    user_adj_embedding = None
                    item_adj_embedding = None

                user_embedding_output = model.forward(user_embedding_input, item_embedding_input,
                                                      timediffs=user_timediffs_tensor, features=feature_tensor,
                                                      adj_embeddings=user_adj_embedding, select='user_update')
                item_embedding_output = model.forward(user_embedding_input, item_embedding_input,
                                                      timediffs=item_timediffs_tensor, features=feature_tensor,
                                                      adj_embeddings=item_adj_embedding, select='item_update')
                # user_embedding_output = model.forward(user_embedding_input, item_embedding_input, timediffs=user_timediffs_tensor, features=feature_tensor, select='user_update')
                # item_embedding_output = model.forward(user_embedding_input, item_embedding_input, timediffs=item_timediffs_tensor, features=feature_tensor, select='item_update')

                item_embeddings[tbatch_itemids, :] = item_embedding_output
                user_embeddings[tbatch_userids, :] = user_embedding_output

                #user_embeddings_timeseries[tbatch_interactionids,:] = user_embedding_output
                #item_embeddings_timeseries[tbatch_interactionids,:] = item_embedding_output

                # CALCULATE LOSS TO MAINTAIN TEMPORAL SMOOTHNESS
                loss += args.l2i*MSELoss(item_embedding_output, item_embedding_input.detach())
                loss += args.l2u*MSELoss(user_embedding_output, user_embedding_input.detach())

                # CALCULATE STATE CHANGE LOSS
                # if args.state_change:
                #     loss += calculate_state_prediction_loss(model, tbatch_interactionids, user_embeddings_timeseries, y_true, crossEntropyLoss)

            # BACKPROPAGATE ERROR AFTER END OF T-BATCH
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # RESET LOSS FOR NEXT T-BATCH
            loss = 0
            item_embeddings.detach_() # Detachment is needed to prevent double propagation of gradient
            user_embeddings.detach_()
            item_embeddings_timeseries.detach_()
            user_embeddings_timeseries.detach_()

            # REINITIALIZE
            reinitialize_tbatches()
            tbatch_to_insert = -1

    # END OF ONE EPOCH
    print "\n\nTotal loss in this epoch = %f" % (total_loss)
    item_embeddings_dystat = torch.cat([item_embeddings, item_embedding_static], dim=1)
    user_embeddings_dystat = torch.cat([user_embeddings, user_embedding_static], dim=1)
    # SAVE CURRENT MODEL TO DISK TO BE USED IN EVALUATION.
    save_model(model, optimizer, args, ep, user_embeddings_dystat, item_embeddings_dystat, train_end_idx,
               user_adj, item_adj, user_embeddings_timeseries, item_embeddings_timeseries)

    user_embeddings = initial_user_embedding.repeat(num_users, 1)
    item_embeddings = initial_item_embedding.repeat(num_items, 1)

# END OF ALL EPOCHS. SAVE FINAL MODEL DISK TO BE USED IN EVALUATION.
print "\n\n*** Training complete. Saving final model. ***\n\n"
save_model(model, optimizer, args, ep, user_embeddings_dystat, item_embeddings_dystat, train_end_idx, user_adj, item_adj, user_embeddings_timeseries, item_embeddings_timeseries)
