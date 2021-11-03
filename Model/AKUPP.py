'''
Created on Dec 18, 2018
Tensorflow Implementation of Knowledge Graph Attention Network (KGAT) model in:
Wang Xiang et al. KGAT: Knowledge Graph Attention Network for Recommendation. In KDD 2019.
@author: Xiang Wang (xiangwang@u.nus.edu)
AKUPP added by Xintao Ma on Feb. 2021
'''
import tensorflow as tf
import os
import numpy as np
import scipy.sparse as sp
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class AKUPP(object):
    def __init__(self, data_config, pretrain_data, args):
        self._parse_args(data_config, pretrain_data, args)
        '''
        *********************************************************
        Create Placeholder for Input Data & Dropout.
        '''
        self._build_inputs()

        """
        *********************************************************
        Create Model Parameters for CF & KGE parts.
        """
        self.weights = self._build_weights()

        """
        *********************************************************
        Compute Graph-based Representations of All Users & Items & KG Entities via Message-Passing Mechanism of Graph Neural Networks.
        Different Convolutional Layers:
            1. bi: defined in 'KGAT: Knowledge Graph Attention Network for Recommendation', KDD2019;
            2. gcn:  defined in 'Semi-Supervised Classification with Graph Convolutional Networks', ICLR2018;
            3. graphsage: defined in 'Inductive Representation Learning on Large Graphs', NeurIPS2017.
        """
        self._build_model_phase_I()
        """
        Optimize Recommendation (CF) Part via BPR Loss.
        """
        self._build_loss_phase_I()

        """
        *********************************************************
        Compute Knowledge Graph Embeddings via TransR.
        """
        self._build_model_phase_II()
        """
        Optimize KGE Part via BPR Loss.
        """
        self._build_loss_phase_II()

        self._statistics_params()

    def _parse_args(self, data_config, pretrain_data, args):
        # argument settings
        self.model_type = 'akupp'

        self.pretrain_data = pretrain_data

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_entities = data_config['n_entities']
        self.n_relations = data_config['n_relations']

        self.n_fold = 100

        # initialize the attentive matrix A for phase I.
        self.A_in = data_config['A_in']

        self.all_h_list = data_config['all_h_list']
        self.all_r_list = data_config['all_r_list']
        self.all_t_list = data_config['all_t_list']
        self.all_v_list = data_config['all_v_list']

        self.adj_uni_type = args.adj_uni_type

        self.lr = args.lr

        # settings for CF part.
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size

        # settings for KG part.
        self.kge_dim = args.kge_size
        self.batch_size_kg = args.batch_size_kg
        #settings for ripple
        self.ripple_dim=args.ripple_size
        self.n_hop=args.n_hop
        self.n_memory=args.n_memory
        self.item_update_mode=args.item_update_mode
        self.kge_weight=args.kge_weight
        self.l2_weight=args.l2_weight
        self.using_all_hops=args.using_all_hops

        self.weight_size = eval(args.layer_size)
        self.n_layers = len(self.weight_size)

        self.alg_type = args.alg_type
        self.model_type += '_%s_%s_%s_l%d' % (args.adj_type, args.adj_uni_type, args.alg_type, self.n_layers)

        self.regs = eval(args.regs)
        self.verbose = args.verbose

    def _build_inputs(self):
        # placeholder definition
        self.users = tf.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.placeholder(tf.int32, shape=(None,))
        self.neg_items = tf.placeholder(tf.int32, shape=(None,))

        # for knowledge graph modeling (TransD) not use in ripple TODO
        # self.A_values = tf.placeholder(tf.float32, shape=[len(self.all_v_list)], name='A_values')
        #
        # self.h = tf.placeholder(tf.int32, shape=[None], name='h')
        # self.r = tf.placeholder(tf.int32, shape=[None], name='r')
        # self.pos_t = tf.placeholder(tf.int32, shape=[None], name='pos_t')
        # self.neg_t = tf.placeholder(tf.int32, shape=[None], name='neg_t')
        #TODO rippleset by xintao
        #use attention or not
        #self.A_values = tf.placeholder(tf.float32, shape=[len(self.all_v_list)], name='A_values')
        self.memories_h=[]
        self.memories_r = []
        self.memories_t = []
        self.items = tf.placeholder(dtype=tf.int32, shape=[None], name="items")
        self.labels = tf.placeholder(dtype=tf.float64, shape=[None], name="labels")
        for hop in range(self.n_hop):
            self.memories_h.append(
                tf.placeholder(dtype=tf.int32, shape=[None, self.n_memory], name="memories_h_" + str(hop)))
            self.memories_r.append(
                tf.placeholder(dtype=tf.int32, shape=[None, self.n_memory], name="memories_r_" + str(hop)))
            self.memories_t.append(
                tf.placeholder(dtype=tf.int32, shape=[None, self.n_memory], name="memories_t_" + str(hop)))

        # dropout: node dropout (adopted on the ego-networks);
        # message dropout (adopted on the convolution operations).
        self.node_dropout = tf.placeholder(tf.float32, shape=[None])
        self.mess_dropout = tf.placeholder(tf.float32, shape=[None])

    def _build_weights(self):
        all_weights = dict()

        initializer = tf.contrib.layers.xavier_initializer()

        if self.pretrain_data is None:
            all_weights['user_embed'] = tf.Variable(initializer([self.n_users, self.emb_dim]), name='user_embed')
            all_weights['entity_embed'] = tf.Variable(initializer([self.n_entities, self.emb_dim]), name='entity_embed')
            print('using xavier initialization')
        else:
            all_weights['user_embed'] = tf.Variable(initial_value=self.pretrain_data['user_embed'], trainable=True,
                                                    name='user_embed', dtype=tf.float32)

            item_embed = self.pretrain_data['item_embed']
            other_embed = initializer([self.n_entities - self.n_items, self.emb_dim])

            all_weights['entity_embed'] = tf.Variable(initial_value=tf.concat([item_embed, other_embed], 0),
                                                      trainable=True, name='entity_embed', dtype=tf.float32)
            print('using pretrained initialization')

        all_weights['relation_embed'] = tf.Variable(initializer([self.n_relations, self.kge_dim]),
                                                    name='relation_embed')
        all_weights['trans_W'] = tf.Variable(initializer([self.n_relations, self.emb_dim, self.kge_dim]))

        self.weight_size_list = [self.emb_dim] + self.weight_size

        #TODO rippleset
        self.entity_emb_matrix = tf.get_variable(name="entity_emb_matrix", dtype=tf.float64,
                                                 shape=[self.n_entities, self.ripple_dim],
                                                 initializer=tf.contrib.layers.xavier_initializer())
        self.relation_emb_matrix = tf.get_variable(name="relation_emb_matrix", dtype=tf.float64,
                                                   shape=[self.n_relations, self.ripple_dim, self.ripple_dim],
                                                   initializer=tf.contrib.layers.xavier_initializer())

        for k in range(self.n_layers):
            all_weights['W_gc_%d' %k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k+1]]), name='W_gc_%d' % k)
            all_weights['b_gc_%d' %k] = tf.Variable(
                initializer([1, self.weight_size_list[k+1]]), name='b_gc_%d' % k)

            all_weights['W_bi_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_bi_%d' % k)
            all_weights['b_bi_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_bi_%d' % k)

            all_weights['W_mlp_%d' % k] = tf.Variable(
                initializer([2 * self.weight_size_list[k], self.weight_size_list[k+1]]), name='W_mlp_%d' % k)
            all_weights['b_mlp_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k+1]]), name='b_mlp_%d' % k)

        return all_weights

    def _build_model_phase_I(self):
        if self.alg_type in ['bi', 'akupp']:
            self.ua_embeddings, self.ea_embeddings = self._create_bi_interaction_embed()

        elif self.alg_type in ['gcn']:
            self.ua_embeddings, self.ea_embeddings = self._create_gcn_embed()

        elif self.alg_type in ['graphsage']:
            self.ua_embeddings, self.ea_embeddings = self._create_graphsage_embed()
        else:
            print('please check the the alg_type argument, which should be bi, akupp, gcn, or graphsage.')
            raise NotImplementedError

        self.u_e = tf.nn.embedding_lookup(self.ua_embeddings, self.users)
        self.pos_i_e = tf.nn.embedding_lookup(self.ea_embeddings, self.pos_items)
        self.neg_i_e = tf.nn.embedding_lookup(self.ea_embeddings, self.neg_items)

        self.batch_predictions = tf.matmul(self.u_e, self.pos_i_e, transpose_a=False, transpose_b=True)

    def _build_model_phase_II(self):
        # self.h_e, self.r_e, self.pos_t_e, self.neg_t_e = self._get_kg_inference(self.h, self.r, self.pos_t, self.neg_t)
        # self.A_kg_score = self._generate_transE_score(h=self.h, t=self.pos_t, r=self.r)
        # self.A_out = self._create_attentive_A_out()
        #TODO ripple
        #transformation matrix for updating item embeddings at the end of each hop
        self.transform_matrix = tf.get_variable(name="transform_matrix", shape=[self.ripple_dim, self.ripple_dim], dtype=tf.float64,
                                                initializer=tf.contrib.layers.xavier_initializer())

        # [batch size, dim]
        self.item_embeddings = tf.nn.embedding_lookup(self.entity_emb_matrix, self.items)
        self.h_emb_list = []
        self.r_emb_list = []
        self.t_emb_list = []
        for i in range(self.n_hop):
            # [batch size, n_memory, dim]
            self.h_emb_list.append(tf.nn.embedding_lookup(self.entity_emb_matrix, self.memories_h[i]))

            # [batch size, n_memory, dim, dim]
            self.r_emb_list.append(tf.nn.embedding_lookup(self.relation_emb_matrix, self.memories_r[i]))

            # [batch size, n_memory, dim]
            self.t_emb_list.append(tf.nn.embedding_lookup(self.entity_emb_matrix, self.memories_t[i]))

        o_list = self._key_addressing()

        self.scores = tf.squeeze(self.predict(self.item_embeddings, o_list))
        self.scores_normalized = tf.sigmoid(self.scores)

    #TODO ripple
    def _key_addressing(self):
        o_list = []
        for hop in range(self.n_hop):
            # [batch_size, n_memory, dim, 1]
            h_expanded = tf.expand_dims(self.h_emb_list[hop], axis=3)

            # [batch_size, n_memory, dim]
            Rh = tf.squeeze(tf.matmul(self.r_emb_list[hop], h_expanded), axis=3)

            # [batch_size, dim, 1]
            v = tf.expand_dims(self.item_embeddings, axis=2)


            # [batch_size, n_memory]
            probs = tf.squeeze(tf.matmul(Rh, v), axis=2)

            # [batch_size, n_memory]
            probs_normalized = tf.nn.softmax(probs)

            # [batch_size, n_memory, 1]
            probs_expanded = tf.expand_dims(probs_normalized, axis=2)

            # [batch_size, dim]
            o = tf.reduce_sum(self.t_emb_list[hop] * probs_expanded, axis=1)

            self.item_embeddings = self.update_item_embedding(self.item_embeddings, o)
            o_list.append(o)
        return o_list

    def update_item_embedding(self, item_embeddings, o):
        if self.item_update_mode == "replace":
            item_embeddings = o
        elif self.item_update_mode == "plus":
            item_embeddings = item_embeddings + o
        elif self.item_update_mode == "replace_transform":
            item_embeddings = tf.matmul(o, self.transform_matrix)
        elif self.item_update_mode == "plus_transform":
            item_embeddings = tf.matmul(item_embeddings + o, self.transform_matrix)
        else:
            raise Exception("Unknown item updating mode: " + self.item_update_mode)
        return item_embeddings

    def predict(self, item_embeddings, o_list):
        y = o_list[-1]
        if self.using_all_hops:
            for i in range(self.n_hop - 1):
                y += o_list[i]

        # [batch_size]
        scores = tf.reduce_sum(item_embeddings * y, axis=1)
        return scores

    #TODO not use kg embbeding

    # def _get_kg_inference(self, h, r, pos_t, neg_t):
    #     embeddings = tf.concat([self.weights['user_embed'], self.weights['entity_embed']], axis=0)
    #     embeddings = tf.expand_dims(embeddings, 1)
    #
    #     # head & tail entity embeddings: batch_size *1 * emb_dim
    #     h_e = tf.nn.embedding_lookup(embeddings, h)
    #     pos_t_e = tf.nn.embedding_lookup(embeddings, pos_t)
    #     neg_t_e = tf.nn.embedding_lookup(embeddings, neg_t)
    #
    #     # relation embeddings: batch_size * kge_dim
    #     r_e = tf.nn.embedding_lookup(self.weights['relation_embed'], r)
    #
    #     # relation transform weights: batch_size * kge_dim * emb_dim
    #     trans_M = tf.nn.embedding_lookup(self.weights['trans_W'], r)
    #
    #     # batch_size * 1 * kge_dim -> batch_size * kge_dim
    #     h_e = tf.reshape(tf.matmul(h_e, trans_M), [-1, self.kge_dim])
    #     pos_t_e = tf.reshape(tf.matmul(pos_t_e, trans_M), [-1, self.kge_dim])
    #     neg_t_e = tf.reshape(tf.matmul(neg_t_e, trans_M), [-1, self.kge_dim])
    #
    #     # Remove the l2 normalization terms
    #     # h_e = tf.math.l2_normalize(h_e, axis=1)
    #     # r_e = tf.math.l2_normalize(r_e, axis=1)
    #     # pos_t_e = tf.math.l2_normalize(pos_t_e, axis=1)
    #     # neg_t_e = tf.math.l2_normalize(neg_t_e, axis=1)
    #
    #     return h_e, r_e, pos_t_e, neg_t_e

    def _build_loss_phase_I(self):
        pos_scores = tf.reduce_sum(tf.multiply(self.u_e, self.pos_i_e), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(self.u_e, self.neg_i_e), axis=1)

        regularizer = tf.nn.l2_loss(self.u_e) + tf.nn.l2_loss(self.pos_i_e) + tf.nn.l2_loss(self.neg_i_e)
        regularizer = regularizer / self.batch_size

        # Using the softplus as BPR loss to avoid the nan error.
        base_loss = tf.reduce_mean(tf.nn.softplus(-(pos_scores - neg_scores)))
        # maxi = tf.log(tf.nn.sigmoid(pos_scores - neg_scores))
        # base_loss = tf.negative(tf.reduce_mean(maxi))

        self.base_loss = base_loss
        self.kge_loss = tf.constant(0.0, tf.float32, [1])
        self.reg_loss = self.regs[0] * regularizer
        self.loss = self.base_loss + self.kge_loss + self.reg_loss

        # Optimization process.RMSPropOptimizer
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def _build_loss_phase_II(self):
        # def _get_kg_score(h_e, r_e, t_e):
        #     kg_score = tf.reduce_sum(tf.square((h_e + r_e - t_e)), 1, keepdims=True)
        #     return kg_score
        #
        # pos_kg_score = _get_kg_score(self.h_e, self.r_e, self.pos_t_e)
        # neg_kg_score = _get_kg_score(self.h_e, self.r_e, self.neg_t_e)
        #
        # # Using the softplus as BPR loss to avoid the nan error.
        # kg_loss = tf.reduce_mean(tf.nn.softplus(-(neg_kg_score - pos_kg_score)))
        # # maxi = tf.log(tf.nn.sigmoid(neg_kg_score - pos_kg_score))
        # # kg_loss = tf.negative(tf.reduce_mean(maxi))
        #
        #
        # kg_reg_loss = tf.nn.l2_loss(self.h_e) + tf.nn.l2_loss(self.r_e) + \
        #               tf.nn.l2_loss(self.pos_t_e) + tf.nn.l2_loss(self.neg_t_e)
        # kg_reg_loss = kg_reg_loss / self.batch_size_kg
        #
        # self.kge_loss2 = kg_loss
        # self.reg_loss2 = self.regs[1] * kg_reg_loss
        # self.loss2 = self.kge_loss2 + self.reg_loss2-=0000000009
        #
        # # Optimization process.
        # self.opt2 = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss2)

        #TODO ripplenet loss
        self.base_loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.scores))

        self.kge_loss2 = 0
        for hop in range(self.n_hop):
            h_expanded = tf.expand_dims(self.h_emb_list[hop], axis=2)
            t_expanded = tf.expand_dims(self.t_emb_list[hop], axis=3)
            hRt = tf.squeeze(tf.matmul(tf.matmul(h_expanded, self.r_emb_list[hop]), t_expanded))
            self.kge_loss2 += tf.reduce_mean(tf.sigmoid(hRt))
        self.kge_loss2 = -self.kge_weight * self.kge_loss2

        self.l2_loss = 0
        for hop in range(self.n_hop):
            self.l2_loss += tf.reduce_mean(tf.reduce_sum(self.h_emb_list[hop] * self.h_emb_list[hop]))
            self.l2_loss += tf.reduce_mean(tf.reduce_sum(self.t_emb_list[hop] * self.t_emb_list[hop]))
            self.l2_loss += tf.reduce_mean(tf.reduce_sum(self.r_emb_list[hop] * self.r_emb_list[hop]))
            if self.item_update_mode == "replace nonlinear" or self.item_update_mode == "plus nonlinear":
                self.l2_loss += tf.nn.l2_loss(self.transform_matrix)
        self.l2_loss = self.l2_weight * self.l2_loss

        self.loss2 = self.base_loss2 + self.kge_loss2 + self.l2_loss
        self.opt2 = tf.train.AdamOptimizer(self.lr).minimize(self.loss2)

    def _create_bi_interaction_embed(self):
        A = self.A_in
        # Generate a set of adjacency sub-matrix.
        A_fold_hat = self._split_A_hat(A)

        ego_embeddings = tf.concat([self.weights['user_embed'], self.weights['entity_embed']], axis=0)
        all_embeddings = [ego_embeddings]

        for k in range(0, self.n_layers):
            # A_hat_drop = tf.nn.dropout(A_hat, 1 - self.node_dropout[k], [self.n_users + self.n_items, 1])
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))

            # sum messages of neighbors.
            side_embeddings = tf.concat(temp_embed, 0)

            add_embeddings = ego_embeddings + side_embeddings

            # transformed sum messages of neighbors.
            sum_embeddings = tf.nn.leaky_relu(
                tf.matmul(add_embeddings, self.weights['W_gc_%d' % k]) + self.weights['b_gc_%d' % k])


            # bi messages of neighbors.
            bi_embeddings = tf.multiply(ego_embeddings, side_embeddings)
            # transformed bi messages of neighbors.
            bi_embeddings = tf.nn.leaky_relu(
                tf.matmul(bi_embeddings, self.weights['W_bi_%d' % k]) + self.weights['b_bi_%d' % k])

            ego_embeddings = bi_embeddings + sum_embeddings
            # message dropout.
            ego_embeddings = tf.nn.dropout(ego_embeddings, 1 - self.mess_dropout[k])

            # normalize the distribution of embeddings.
            norm_embeddings = tf.math.l2_normalize(ego_embeddings, axis=1)

            all_embeddings += [norm_embeddings]

        all_embeddings = tf.concat(all_embeddings, 1)

        ua_embeddings, ea_embeddings = tf.split(all_embeddings, [self.n_users, self.n_entities], 0)
        return ua_embeddings, ea_embeddings

    def _create_gcn_embed(self):
        A = self.A_in
        # Generate a set of adjacency sub-matrix.
        A_fold_hat = self._split_A_hat(A)

        embeddings = tf.concat([self.weights['user_embed'], self.weights['entity_embed']], axis=0)
        all_embeddings = [embeddings]

        for k in range(0, self.n_layers):
            # A_hat_drop = tf.nn.dropout(A_hat, 1 - self.node_dropout[k], [self.n_users + self.n_items, 1])
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], embeddings))

            embeddings = tf.concat(temp_embed, 0)
            embeddings = tf.nn.leaky_relu(
                tf.matmul(embeddings, self.weights['W_gc_%d' % k]) + self.weights['b_gc_%d' % k])
            embeddings = tf.nn.dropout(embeddings, 1 - self.mess_dropout[k])

            # normalize the distribution of embeddings.
            norm_embeddings = tf.math.l2_normalize(embeddings, axis=1)

            all_embeddings += [norm_embeddings]

        all_embeddings = tf.concat(all_embeddings, 1)

        ua_embeddings, ea_embeddings = tf.split(all_embeddings, [self.n_users, self.n_entities], 0)
        return ua_embeddings, ea_embeddings

    def _create_graphsage_embed(self):
        A = self.A_in
        # Generate a set of adjacency sub-matrix.
        A_fold_hat = self._split_A_hat(A)

        pre_embeddings = tf.concat([self.weights['user_embed'], self.weights['entity_embed']], axis=0)

        all_embeddings = [pre_embeddings]
        for k in range(self.n_layers):
            # line 1 in algorithm 1 [RM-GCN, KDD'2018], aggregator layer: weighted sum
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], pre_embeddings))
            embeddings = tf.concat(temp_embed, 0)

            # line 2 in algorithm 1 [RM-GCN, KDD'2018], aggregating the previsou embeddings
            embeddings = tf.concat([pre_embeddings, embeddings], 1)
            pre_embeddings = tf.nn.relu(
                tf.matmul(embeddings, self.weights['W_mlp_%d' % k]) + self.weights['b_mlp_%d' % k])

            pre_embeddings = tf.nn.dropout(pre_embeddings, 1 - self.mess_dropout[k])

            # normalize the distribution of embeddings.
            norm_embeddings = tf.math.l2_normalize(embeddings, axis=1)

            all_embeddings += [norm_embeddings]

        all_embeddings = tf.concat(all_embeddings, 1)

        ua_embeddings, ea_embeddings = tf.split(all_embeddings, [self.n_users, self.n_entities], 0)
        return ua_embeddings, ea_embeddings

    def _split_A_hat(self, X):
        A_fold_hat = []

        fold_len = (self.n_users + self.n_entities) // self.n_fold

        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.n_users + self.n_entities
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def _create_attentive_A_out(self):
        indices = np.mat([self.all_h_list, self.all_t_list]).transpose()
        A = tf.sparse.softmax(tf.SparseTensor(indices, self.A_values, self.A_in.shape))
        return A


    def _statistics_params(self):
        # number of params
        total_parameters = 0
        for variable in self.weights.values():
            shape = variable.get_shape()  # shape is an array of tf.Dimension
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        if self.verbose > 0:
            print("#params: %d" % total_parameters)

    def train(self, sess, feed_dict):
        return sess.run([self.opt, self.loss, self.base_loss, self.kge_loss, self.reg_loss], feed_dict)

    def train_A(self, sess, feed_dict):
        #return sess.run([self.opt2, self.loss2, self.kge_loss2, self.reg_loss2], feed_dict)

        return sess.run([self.opt2, self.loss2, self.kge_loss2], feed_dict)

    def eval(self, sess, feed_dict):
        batch_predictions = sess.run(self.batch_predictions, feed_dict)
        return batch_predictions

    """
    Update the attentive laplacian matrix.
    """
    def update_attentive_A(self, sess):
        fold_len = len(self.all_h_list) // self.n_fold
        kg_score = []

        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = len(self.all_h_list)
            else:
                end = (i_fold + 1) * fold_len

            feed_dict = {
                self.h: self.all_h_list[start:end],
                self.r: self.all_r_list[start:end],
                self.pos_t: self.all_t_list[start:end]
            }
            A_kg_score = sess.run(self.A_kg_score, feed_dict=feed_dict)
            kg_score += list(A_kg_score)

        kg_score = np.array(kg_score)

        new_A = sess.run(self.A_out, feed_dict={self.A_values: kg_score})
        new_A_values = new_A.values
        new_A_indices = new_A.indices

        rows = new_A_indices[:, 0]
        cols = new_A_indices[:, 1]
        self.A_in = sp.coo_matrix((new_A_values, (rows, cols)), shape=(self.n_users + self.n_entities,
                                                                       self.n_users + self.n_entities))
        if self.alg_type in ['org', 'gcn']:
            self.A_in.setdiag(1.)
