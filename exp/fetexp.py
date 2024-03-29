import os
import logging
import json
import collections
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
import tensorflow_probability as tfp
import scann
# from bert import tokenization
from locbert import tokenization, optimization
from orqa.utils import scann_utils
import config
from utils import datautils, bert_utils


output_dir = os.path.join(config.OUTPUT_DIR, 'realm_output')
log_dir = os.path.join(config.OUTPUT_DIR, 'realm_output/log')
data_dir = config.DATA_DIR

# orqa_ops = tf.load_op_library('exp/rlmfetops.so')

# num_block_records = 13353718
# num_block_records = 2000000
# n_block_rec_parts = [2670743, 5341486, 8012229, 10682972, 13353718]
# retriever_module_path = os.path.join(data_dir, 'realm_data/cc_news_pretrained/embedder')
# var_name = "block_emb"
# checkpoint_path = os.path.join(retriever_module_path, "encoded", "encoded.ckpt")
# # np_db = tf.train.load_checkpoint(checkpoint_path).get_tensor(var_name)[:4000000]
# block_emb_file = os.path.join(config.DATA_DIR, 'realm_data/realm_blocks/block_emb_2m.pkl')
# block_records_path = os.path.join(config.DATA_DIR, 'realm_data/realm_blocks/blocks_2m.tfr')
# # block_records_path = os.path.join(data_dir, 'realm_data/blocks.tfr')
# np_db = datautils.load_pickle_data(block_emb_file)
pre_load_data = dict()


def load_rt_dataset_single_elem(dataset_path):
    dataset = tf.data.experimental.load(
        dataset_path,
        element_spec=tf.RaggedTensorSpec(ragged_rank=1, dtype=tf.int32))
    print('dataset loaded from {}'.format(dataset_path))
    # for i, v in enumerate(dataset):
    #     print(v)
    #     if i > 3:
    #         break
    # dataset = dataset.batch(n_records)
    # print('batched')
    # t = time.time()
    all_docs = tf.data.experimental.get_single_element(dataset)

    # print(type(all_docs))
    # print(all_docs[:3])
    # print(time.time() - t)
    return all_docs


def load_dataset_parts(dataset_path_prefix, n_parts):
    vals_list = list()
    for i in range(n_parts):
        dataset_path = '{}{}.tfd'.format(dataset_path_prefix, i)
        vals = load_rt_dataset_single_elem(dataset_path)
        vals_list.append(vals)
    if n_parts == 1:
        return vals_list[0]
    return tf.concat(vals_list, axis=0)


def load_blocks_from_pkl():
    blocks_list = datautils.load_pickle_data(os.path.join(config.DATA_DIR, 'realm_data/blocks_tok_id_seqs.pkl'))
    blocks_list = blocks_list[:10000000]
    print('blocks list loaded', len(blocks_list))
    blocks = tf.ragged.constant(blocks_list, dtype=tf.int32)
    print('blocks list to ragged')
    return blocks


def retrieve(
        query_token_id_seqs, input_mask, embedder_path, mode, block_records_path, retriever_beam_size,
        num_block_records):
    """Do retrieval."""
    # print('RRRRRRRRRRRRRRRRRetrieve', mode)
    # retriever_module_path = os.path.join(data_dir, 'realm_data/cc_news_pretrained/embedder')
    # var_name = "block_emb"
    # checkpoint_path = os.path.join(retriever_module_path, "encoded", "encoded.ckpt")

    retriever_module = hub.Module(
        embedder_path,
        tags={"train"} if mode == tf.estimator.ModeKeys.TRAIN else {},
        trainable=True)

    # [1, projection_size]
    question_emb = retriever_module(
        inputs=dict(
            input_ids=query_token_id_seqs,
            # input_mask=tf.ones_like(query_token_id_seqs),
            input_mask=input_mask,
            segment_ids=tf.zeros_like(query_token_id_seqs)),
        signature="projected")
    # return question_emb
    # question_emb_np = np.random.uniform(-1, 1, (4, 128))
    # question_emb = tf.constant(question_emb_np, tf.float32)

    # block_emb, searcher = scann_utils.load_scann_searcher(
    #     var_name="block_emb",
    #     checkpoint_path=os.path.join(embedder_path, "encoded", "encoded.ckpt"),
    #     num_neighbors=retriever_beam_size)
    # np_db = tf.train.load_checkpoint(checkpoint_path).get_tensor(var_name)
    # n_docs = np_db.shape[0]

    # block_emb = tf.compat.v1.get_variable('block_emb_tf', shape=np_db.shape)
    #
    # def init_fn(scaffold, sess):
    #     print('FFFFFFFFFFFFFFFFFFFFFF INIT BLOCK EMB')
    #     sess.run(block_emb.initializer, {block_emb.initial_value: np_db})
    #
    # scaffold = tf.compat.v1.train.Scaffold(init_fn=init_fn)
    scaffold = None

    np_db = pre_load_data['np_db']
    block_emb = tf.constant(np_db)
    searcher = scann.scann_ops.builder(block_emb, retriever_beam_size, "dot_product").tree(
        num_leaves=1000, num_leaves_to_search=200, training_sample_size=100000).score_ah(
        2, anisotropic_quantization_threshold=0.2).reorder(100).build()

    # [1, retriever_beam_size]
    retrieved_block_ids, _ = searcher.search_batched(question_emb)

    # [1, retriever_beam_size, projection_size]
    retrieved_block_emb = tf.gather(block_emb, retrieved_block_ids)

    # retrieved_block_emb, retrieved_block_ids, retrieved_block_emb1 = retrieve_block_ids_dist(
    #     retriever_beam_size, question_emb)

    # [retriever_beam_size]
    # retrieved_block_ids = tf.squeeze(retrieved_block_ids)
    # retrieved_block_ids = tf.constant(np.random.randint(0, 10000, (4, 5)))
    retrieved_block_ids = tf.reshape(retrieved_block_ids, shape=(-1, retriever_beam_size))
    # retrieved_block_ids = tf.reshape(retrieved_block_ids, shape=(-1))

    # # [retriever_beam_size, projection_size]
    # retrieved_block_emb = tf.squeeze(retrieved_block_emb)
    #
    # [1, retriever_beam_size]
    question_emb_ex = tf.expand_dims(question_emb, axis=1)
    retrieved_logits = tf.matmul(question_emb_ex, retrieved_block_emb, transpose_b=True)

    # scaffold = None
    # retrieved_block_ids = tf.constant(np.random.randint(0, 10000, (2, retriever_beam_size)), tf.int32)
    # retrieved_logits = tf.constant(np.random.uniform(-1, 1, (2, retriever_beam_size)), tf.float32)

    blocks_dataset = tf.data.TFRecordDataset(
        block_records_path, buffer_size=512 * 1024 * 1024)
    blocks_dataset = blocks_dataset.batch(
        num_block_records, drop_remainder=True)
    blocks = tf.compat.v1.get_local_variable(
        "blocks",
        initializer=tf.data.experimental.get_single_element(blocks_dataset))
    retrieved_blocks = tf.gather(blocks, retrieved_block_ids)

    print('blocks obtained')

    return scaffold, retrieved_block_ids, retrieved_blocks, retrieved_logits, question_emb
    # return scaffold, retrieved_block_ids, retrieved_blocks, retrieved_logits, retrieved_block_emb1


def pad_sep_to_tensor(tok_id_seqs, sep_tok_id):
    reach_max_len = tf.equal(tok_id_seqs[:, -1], tf.constant(0, tf.int32))
    reach_max_len = 1 - tf.cast(reach_max_len, tf.int32)
    reach_max_len = tf.reshape(reach_max_len, (-1, 1))
    seps_tensor = reach_max_len * sep_tok_id
    # print(seps_tensor)
    # print(tf.concat((q_doc_tok_id_seqs, seps_tensor), axis=1))
    tok_id_seqs = tf.concat((tok_id_seqs, seps_tensor), axis=1)

    is_zero = tf.cast(tf.equal(tok_id_seqs, tf.constant(0)), tf.int32)
    # print(is_zero)
    is_zero_cumsum = tf.cumsum(is_zero, axis=1)
    sep_tensor = tf.cast(tf.equal(is_zero_cumsum, tf.constant(1)), tf.int32) * sep_tok_id
    tok_id_seqs += sep_tensor
    return tok_id_seqs


def get_one_hot_label_vecs(labels, n_types):
    ones_add = tf.ones_like(labels, tf.float32)

    n_samples = tf.shape(labels)[0]
    # print(tf.expand_dims(vals, 1))
    # print(tf.range(n_vals))
    idxs = tf.concat((tf.expand_dims(tf.range(n_samples), 1), tf.expand_dims(labels, 1)), axis=1)
    # print(idxs)
    label_vecs = tf.zeros((n_samples, n_types), tf.float32)
    label_vecs = tf.tensor_scatter_nd_update(label_vecs, idxs, ones_add)
    # print(oh_vecs)
    return label_vecs


def model_fn_zlabels(features, labels, mode, params):
    # print('MMMMMMMMMMMMMMMMMModel_fn', mode)
    embedder_module_path = params['embedder_module_path']
    reader_module_path = params['reader_module_path']
    # embedder_module_path = os.path.join(config.DATA_DIR, 'realm_data/cc_news_pretrained/embedder')
    # reader_module_path = os.path.join(config.DATA_DIR, 'realm_data/cc_news_pretrained/bert')
    lr = params['lr']
    num_train_steps = params['num_train_steps']
    max_seq_len = params['max_seq_len']
    bert_dim = params['bert_dim']
    n_types = params['n_types']
    sep_tok_id = params['sep_tok_id']
    retriever_beam_size = params['retriever_beam_size']
    block_records_path = params['block_records_path']
    num_block_records = params['num_block_records']
    train_log_steps = params['train_log_steps']
    eval_log_steps = params['eval_log_steps']

    # token_ids = tf.constant([[101, 2002, 2003, 1037, 3836, 1012, 102]], dtype=tf.int32)
    tok_id_seq_batch_tensor = features['tok_id_seq_batch'].to_tensor()
    input_mask = 1 - tf.cast(tf.equal(tok_id_seq_batch_tensor, tf.constant(0)), tf.int32)

    block_labels_np = pre_load_data['labels']
    block_labels = tf.constant(block_labels_np, tf.int32)

    # input_mask = features['input_mask']
    with tf.device("/cpu:0"):
        # retriever_outputs = retrieve(tok_id_seq_batch, input_mask, embedder_module_path, mode, retriever_beam_size)
        scaffold, retrieved_block_ids, retrieved_blocks, zx_logits, question_emb = retrieve(
            tok_id_seq_batch_tensor, input_mask, embedder_module_path, mode, block_records_path,
            retriever_beam_size, num_block_records)
        # scaffold, question_emb, retrieved_block_ids = retrieve(
        #     tok_id_seq_batch, input_mask, embedder_module_path, mode, retriever_beam_size)

    retrieved_labels = tf.gather(block_labels, retrieved_block_ids[0])
    retrieved_label_vecs = get_one_hot_label_vecs(retrieved_labels, n_types)

    tokenizer, vocab_lookup_table = bert_utils.get_tf_tokenizer(reader_module_path)
    block_tok_id_seqs = tokenizer.tokenize(retrieved_blocks)
    block_tok_id_seqs = tf.cast(
        block_tok_id_seqs.merge_dims(2, 3).to_tensor(), tf.int32)
    # batch_size = tf.shape(tok_id_seq_batch_tensor)[0]
    blocks_max_seq_len = tf.shape(block_tok_id_seqs)[-1]
    block_tok_id_seqs_flat = tf.reshape(block_tok_id_seqs, (-1, blocks_max_seq_len))

    zx_logits = tf.reshape(zx_logits, (-1, retriever_beam_size))
    log_softmax_zx_logits = tf.nn.log_softmax(zx_logits, axis=1)

    # labels0 = tf.reduce_sum(retrieved_label_vecs[0] * tf.range(n_types, dtype=tf.float32))
    # labels0_sum = tf.reduce_sum(retrieved_label_vecs[0])

    # yzx_logits = tf.matmul(qd_reps, dense_weights)
    # # weight_sum = tf.reduce_sum(dense_layer.weights)
    # yzx_logits = tf.reshape(yzx_logits, (-1, retriever_beam_size, n_types))
    # log_sig_yzx_logits = tf.math.log_sigmoid(yzx_logits)
    # z_log_probs = log_sig_yzx_logits + tf.expand_dims(log_softmax_zx_logits, 2)
    # log_probs = tf.reduce_logsumexp(z_log_probs, axis=1)
    # log_neg_probs = tfp.math.log1mexp(log_probs)

    # prob_sum = tf.math.exp(log_probs) + tf.math.exp(log_neg_probs)

    # kernel_initializer = tf.truncated_normal_initializer(stddev=0.02)
    # projection = tf.layers.dense(
    #     qd_reps,
    #     bert_dim,
    #     kernel_initializer=kernel_initializer)
    # yzx_logits =

    # probs = tf.exp(log_probs)
    # loss = tf.reduce_mean(predictions)

    loss = None
    eval_metric_ops = None
    train_op = None
    label_hits = None
    if mode != tf.estimator.ModeKeys.PREDICT:
        retrieved_label_vecs = tf.reshape(retrieved_label_vecs, (-1, retriever_beam_size, n_types))
        label_hits = tf.reduce_sum(retrieved_label_vecs * tf.expand_dims(labels, 1), axis=2)
        label_hits = tf.cast(tf.less(tf.constant(0.5), label_hits), tf.float32)
        loss_samples = -tf.reduce_sum(
            label_hits * log_softmax_zx_logits + (1 - label_hits) * log_softmax_zx_logits, axis=1)
        # loss_samples = -tf.reduce_sum(labels * log_probs + (1 - labels) * log_neg_probs, axis=1)
        loss = tf.reduce_mean(loss_samples)
        # loss = tf.reduce_mean(loss_samples) + 0.00001 * tf.reduce_mean(question_emb)
        # loss = -tf.reduce_mean(log_softmax_zx_logits)

        train_op = optimization.create_optimizer(
            loss=loss,
            init_lr=lr,
            num_train_steps=num_train_steps,
            num_warmup_steps=min(10000,
                                 max(100, int(num_train_steps / 10))),
            use_tpu=False)

        small_constant = tf.constant(0.00001)
        all_pred_label_vecs = tf.reduce_sum(retrieved_label_vecs, axis=1)
        pos_preds = tf.cast(tf.less(tf.constant(0.5), all_pred_label_vecs), tf.float32)
        # pos_preds = tf.cast(tf.less(tf.constant(0.5), probs), tf.float32)
        n_pred_pos = tf.reduce_sum(pos_preds, axis=1) + small_constant
        n_true_pos = tf.reduce_sum(labels, axis=1) + small_constant
        n_corrects = tf.reduce_sum(pos_preds * labels, axis=1)
        precision = tf.reduce_mean(n_corrects / n_pred_pos)
        recall = tf.reduce_mean(n_corrects / n_true_pos)

        p_mean, p_op = tf.compat.v1.metrics.mean(precision)
        r_mean, r_op = tf.compat.v1.metrics.mean(recall)
        f1 = 2 * p_mean * r_mean / (p_mean + r_mean + small_constant)

        eval_metric_ops = {
            # 'precision': tf.compat.v1.metrics.mean(precision),
            # 'recall': tf.compat.v1.metrics.mean(recall)
            'precision': (p_mean, p_op),
            'recall': (r_mean, r_op),
            'f1': (f1, tf.group(p_op, r_op))
        }
        # eval_metric_ops = None

    # tmp_blocks = tf.constant(['i you date', 'sh ij ko', 'day in day'])
    # blocks_has_answer = orqa_ops.has_answer(blocks=tmp_blocks, answers=features['labels'][0])
    # tmp_blocks = tf.constant([[1, 2], [3, 4]], tf.int32)
    # blocks_has_answer = orqa_ops.zero_out(features['tmp'])
    label_hits_sum = tf.reduce_sum(label_hits) if label_hits is not None else None

    train_logging_hook = tf.estimator.LoggingTensorHook({
        'batch_id': features['batch_id'],
        'loss': loss,
        # 'ws': tf.reduce_sum(dense_weights),
        # 'yzx_logits': tf.reduce_sum(yzx_logits),
        'label_hits': label_hits_sum,
        # 'tmp1': retrieved_labels,
        # 'labels0': labels0,
        # 'labels0_sum': labels0_sum,
    }, every_n_iter=train_log_steps)
    logging_hook = tf.estimator.LoggingTensorHook({
        'batch_id': features['batch_id'],
        'loss': loss,
        # 'pred': tf.reduce_mean(predictions),
        # 'pred': log_probs,
        # 'tmp': tf.reduce_sum(dense_weights),
    }, every_n_iter=eval_log_steps)
    pred_logging_hook = tf.estimator.LoggingTensorHook({
        'labels': features['labels'][0],
        # 'ha': blocks_has_answer,
        # 'hatmp': features['tmp'],
        # 'bl': retrieved_blocks
    }, every_n_iter=eval_log_steps)

    # predictions = {'probs': probs, 'text_ids': features['text_ids'], 'block_ids': retrieved_block_ids}
    predictions = {'text_ids': features['text_ids'], 'block_ids': retrieved_block_ids, 'qemb': question_emb}

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        predictions=predictions,
        training_hooks=[train_logging_hook],
        evaluation_hooks=[logging_hook],
        # prediction_hooks=[pred_logging_hook],
        eval_metric_ops=eval_metric_ops,
        scaffold=scaffold)


def model_fn(features, labels, mode, params):
    # print('MMMMMMMMMMMMMMMMMModel_fn', mode)
    embedder_module_path = params['embedder_module_path']
    reader_module_path = params['reader_module_path']
    # embedder_module_path = os.path.join(config.DATA_DIR, 'realm_data/cc_news_pretrained/embedder')
    # reader_module_path = os.path.join(config.DATA_DIR, 'realm_data/cc_news_pretrained/bert')
    lr = params['lr']
    num_train_steps = params['num_train_steps']
    max_seq_len = params['max_seq_len']
    bert_dim = params['bert_dim']
    n_types = params['n_types']
    sep_tok_id = params['sep_tok_id']
    retriever_beam_size = params['retriever_beam_size']
    block_records_path = params['block_records_path']
    num_block_records = params['num_block_records']
    train_log_steps = params['train_log_steps']
    eval_log_steps = params['eval_log_steps']

    # token_ids = tf.constant([[101, 2002, 2003, 1037, 3836, 1012, 102]], dtype=tf.int32)
    tok_id_seq_batch_tensor = features['tok_id_seq_batch'].to_tensor()
    input_mask = 1 - tf.cast(tf.equal(tok_id_seq_batch_tensor, tf.constant(0)), tf.int32)

    block_labels_np = pre_load_data.get('labels', None)
    block_labels = tf.constant(block_labels_np, tf.int32) if block_labels_np is not None else None

    # input_mask = features['input_mask']
    with tf.device("/cpu:0"):
        # retriever_outputs = retrieve(tok_id_seq_batch, input_mask, embedder_module_path, mode, retriever_beam_size)
        scaffold, retrieved_block_ids, retrieved_blocks, zx_logits = retrieve(
            tok_id_seq_batch_tensor, input_mask, embedder_module_path, mode, block_records_path,
            retriever_beam_size, num_block_records)
        # scaffold, question_emb, retrieved_block_ids = retrieve(
        #     tok_id_seq_batch, input_mask, embedder_module_path, mode, retriever_beam_size)

    retrieved_labels = tf.gather(block_labels, retrieved_block_ids[0]) if block_labels is not None else None
    retrieved_label_vecs = get_one_hot_label_vecs(retrieved_labels, n_types)

    tokenizer, vocab_lookup_table = bert_utils.get_tf_tokenizer(reader_module_path)
    block_tok_id_seqs = tokenizer.tokenize(retrieved_blocks)
    block_tok_id_seqs = tf.cast(
        block_tok_id_seqs.merge_dims(2, 3).to_tensor(), tf.int32)
    # batch_size = tf.shape(tok_id_seq_batch_tensor)[0]
    blocks_max_seq_len = tf.shape(block_tok_id_seqs)[-1]
    block_tok_id_seqs_flat = tf.reshape(block_tok_id_seqs, (-1, blocks_max_seq_len))

    tok_id_seqs_repeat = features['tok_id_seqs_repeat']
    # tok_id_seqs_repeat = features['tok_id_seqs_repeat'].to_tensor()
    q_doc_tok_id_seqs = tf.concat((tok_id_seqs_repeat, block_tok_id_seqs_flat), axis=1).to_tensor()
    q_doc_tok_id_seqs = q_doc_tok_id_seqs[:, :max_seq_len - 1]
    q_doc_tok_id_seqs = pad_sep_to_tensor(q_doc_tok_id_seqs, sep_tok_id)

    q_doc_input_mask = 1 - tf.cast(tf.equal(q_doc_tok_id_seqs, tf.constant(0, dtype=tf.int32)), tf.int32)

    reader_module = hub.Module(
        reader_module_path,
        tags={"train"} if mode == tf.estimator.ModeKeys.TRAIN else {},
        trainable=True)

    # input_mask = tf.sequence_mask(lengths, seqs_shape[1])
    # input_mask = tf.cast(input_mask, tf.int32)
    concat_outputs = reader_module(
        dict(
            # input_ids=tok_id_seq_batch,
            # input_mask=tf.ones_like(tok_id_seq_batch),
            # segment_ids=tf.zeros_like(tok_id_seq_batch)
            # segment_ids=concat_inputs.segment_ids
            input_ids=q_doc_tok_id_seqs,
            # input_mask=tf.ones_like(r_tok_id_seqs),
            input_mask=q_doc_input_mask,
            segment_ids=tf.zeros_like(q_doc_tok_id_seqs)
        ),
        signature="tokens",
        as_dict=True)
    # # predictions = retriever_outputs.logits
    #
    # concat_token_emb = concat_outputs["sequence_output"]
    qd_reps = concat_outputs['sequence_output'][:, 0, :]

    # dense_layer = tf.keras.layers.Dense(n_types)
    # yzx_logits = dense_layer(qd_reps)
    dense_weights = tf.Variable(
        initial_value=np.random.uniform(-0.1, 0.1, (bert_dim, n_types)), trainable=True, dtype=tf.float32)

    zx_logits = tf.reshape(zx_logits, (-1, retriever_beam_size))
    log_softmax_zx_logits = tf.nn.log_softmax(zx_logits, axis=1)

    yzx_logits = tf.matmul(qd_reps, dense_weights)
    # weight_sum = tf.reduce_sum(dense_layer.weights)
    yzx_logits = tf.reshape(yzx_logits, (-1, retriever_beam_size, n_types))
    log_sig_yzx_logits = tf.math.log_sigmoid(yzx_logits)
    z_log_probs = log_sig_yzx_logits + tf.expand_dims(log_softmax_zx_logits, 2)
    log_probs = tf.reduce_logsumexp(z_log_probs, axis=1)
    log_neg_probs = tfp.math.log1mexp(log_probs)
    # prob_sum = tf.math.exp(log_probs) + tf.math.exp(log_neg_probs)

    # kernel_initializer = tf.truncated_normal_initializer(stddev=0.02)
    # projection = tf.layers.dense(
    #     qd_reps,
    #     bert_dim,
    #     kernel_initializer=kernel_initializer)
    # yzx_logits =

    probs = tf.exp(log_probs)
    # loss = tf.reduce_mean(predictions)

    loss = None
    eval_metric_ops = None
    train_op = None
    if mode != tf.estimator.ModeKeys.PREDICT:
        loss_samples = -tf.reduce_sum(labels * log_probs + (1 - labels) * log_neg_probs, axis=1)
        loss = tf.reduce_mean(loss_samples)
        # loss = tf.reduce_mean(loss_samples) + 0.00001 * tf.reduce_mean(question_emb)

        train_op = optimization.create_optimizer(
            loss=loss,
            init_lr=lr,
            num_train_steps=num_train_steps,
            num_warmup_steps=min(10000,
                                 max(100, int(num_train_steps / 10))),
            use_tpu=False)

        small_constant = tf.constant(0.00001)
        pos_preds = tf.cast(tf.less(tf.constant(0.5), probs), tf.float32)
        n_pred_pos = tf.reduce_sum(pos_preds, axis=1) + small_constant
        n_true_pos = tf.reduce_sum(labels, axis=1) + small_constant
        n_corrects = tf.reduce_sum(pos_preds * labels, axis=1)
        precision = tf.reduce_mean(n_corrects / n_pred_pos)
        recall = tf.reduce_mean(n_corrects / n_true_pos)

        p_mean, p_op = tf.compat.v1.metrics.mean(precision)
        r_mean, r_op = tf.compat.v1.metrics.mean(recall)
        f1 = 2 * p_mean * r_mean / (p_mean + r_mean + small_constant)

        eval_metric_ops = {
            # 'precision': tf.compat.v1.metrics.mean(precision),
            # 'recall': tf.compat.v1.metrics.mean(recall)
            'precision': (p_mean, p_op),
            'recall': (r_mean, r_op),
            'f1': (f1, tf.group(p_op, r_op))
        }

    # tmp_blocks = tf.constant(['i you date', 'sh ij ko', 'day in day'])
    # blocks_has_answer = orqa_ops.has_answer(blocks=tmp_blocks, answers=features['labels'][0])
    # tmp_blocks = tf.constant([[1, 2], [3, 4]], tf.int32)
    # blocks_has_answer = orqa_ops.zero_out(features['tmp'])

    train_logging_hook = tf.estimator.LoggingTensorHook({
        'batch_id': features['batch_id'],
        'loss': loss,
        'ws': tf.reduce_sum(dense_weights),
        'yzx_logits': tf.reduce_sum(yzx_logits),
        'tmp': retrieved_label_vecs,
        'tmp1': retrieved_labels,
    }, every_n_iter=train_log_steps)
    logging_hook = tf.estimator.LoggingTensorHook({
        'batch_id': features['batch_id'],
        'loss': loss,
        # 'pred': tf.reduce_mean(predictions),
        # 'pred': log_probs,
        'tmp': tf.reduce_sum(dense_weights),
    }, every_n_iter=eval_log_steps)
    pred_logging_hook = tf.estimator.LoggingTensorHook({
        'labels': features['labels'][0],
        # 'ha': blocks_has_answer,
        # 'hatmp': features['tmp'],
        # 'bl': retrieved_blocks
    }, every_n_iter=eval_log_steps)

    predictions = {'probs': probs, 'text_ids': features['text_ids'], 'block_ids': retrieved_block_ids}

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        predictions=predictions,
        training_hooks=[train_logging_hook],
        evaluation_hooks=[logging_hook],
        # prediction_hooks=[pred_logging_hook],
        eval_metric_ops=eval_metric_ops,
        scaffold=scaffold)


def get_padded_bert_input(tok_id_seqs):
    max_seq_len = max(len(seq) for seq in tok_id_seqs)
    tok_id_seq_batch = np.zeros((len(tok_id_seqs), max_seq_len), np.int32)
    input_mask = np.zeros((len(tok_id_seqs), max_seq_len), np.int32)
    for i, seq in enumerate(tok_id_seqs):
        tok_id_seq_batch[i][:len(seq)] = seq
        for j in range(len(seq)):
            input_mask[i][j] = 1
    return tok_id_seq_batch, input_mask


def to_one_hot(inds, vec_len):
    v = np.zeros(vec_len, dtype=np.float32)
    for ind in inds:
        v[ind] = 1
    return v


class InputData:
    def __init__(self, batch_size, tokenizer, types, type_id_dict, retriever_beam_size, n_train_repeat):
        self.batch_size = batch_size
        self.train_data_file = os.path.join(config.DATA_DIR, 'ultrafine/uf_data/crowd/train.json')
        self.dev_data_file = os.path.join(config.DATA_DIR, 'ultrafine/uf_data/crowd/dev.json')
        self.test_data_file = os.path.join(config.DATA_DIR, 'ultrafine/uf_data/crowd/test.json')
        self.tokenizer = tokenizer
        self.types = types
        self.type_id_dict = type_id_dict
        self.n_types = len(self.types)
        self.retriever_beam_size = retriever_beam_size
        self.n_train_repeat = n_train_repeat

    def input_fn_train(self):
        return self.input_fn(self.train_data_file, n_repeat=self.n_train_repeat)

    def input_fn_dev(self):
        return self.input_fn(self.dev_data_file)

    def input_fn_test(self):
        return self.input_fn(self.test_data_file)

    def input_fn(self, data_file, n_repeat=1):
        batch_size = self.batch_size
        texts = list()
        mstr_sep_texts = list()
        all_label_strs, all_labels = list(), list()
        with open(data_file, encoding='utf-8') as f:
            for i, line in enumerate(f):
                x = json.loads(line)
                mstr = x['mention_span']
                text = '{} [MASK] such as {} {}'.format(
                    ' '.join(x['left_context_token']), mstr, ' '.join(x['right_context_token']))
                # text = '{} {} {}'.format(
                #     ' '.join(x['left_context_token']), mstr, ' '.join(x['right_context_token']))
                mstr_sep_text = '{} [SEP] {} {} {}'.format(
                    mstr, ' '.join(x['left_context_token']), mstr, ' '.join(x['right_context_token']))
                # print(text)
                texts.append(text)
                mstr_sep_texts.append(mstr_sep_text)
                labels = x['y_str']
                tids = [self.type_id_dict.get(t, -1) for t in labels]
                tids = [tid for tid in tids if tid > -1]
                # if i > 5:
                all_label_strs.append(labels)
                all_labels.append(tids)
                # if len(texts) >= 12:
                #     break
        print(len(texts), 'texts')

        def tok_id_seq_gen():
            text_ids, tok_id_seqs, tok_id_seqs_repeat = list(), list(), list()
            label_strs, y_vecs = list(), list()
            for _ in range(n_repeat):
                batch_id = 0
                for text_idx, text in enumerate(texts):
                    tokens = self.tokenizer.tokenize(text)
                    tokens_full = ['[CLS]'] + tokens + ['[SEP]']
                    tok_id_seq = self.tokenizer.convert_tokens_to_ids(tokens_full)
                    tok_id_seqs.append(tok_id_seq)

                    fet_tokens = ['[CLS]'] + self.tokenizer.tokenize(mstr_sep_texts[text_idx]) + ['[SEP]']
                    fet_tok_id_seq = self.tokenizer.convert_tokens_to_ids(fet_tokens)
                    # tok_id_seq = np.array([len(text)], np.float32)
                    for _ in range(self.retriever_beam_size):
                        tok_id_seqs_repeat.append(fet_tok_id_seq)
                    text_ids.append(text_idx)
                    y_vecs.append(to_one_hot(all_labels[text_idx], self.n_types))
                    label_strs.append(all_label_strs[text_idx])

                    if len(tok_id_seqs) >= batch_size:
                        # tok_id_seq_batch, input_mask = get_padded_bert_input(tok_id_seqs)
                        tok_id_seq_batch = tf.ragged.constant(tok_id_seqs)
                        tok_id_seqs_repeat_ragged = tf.ragged.constant(tok_id_seqs_repeat)
                        # y_vecs_tensor = tf.concat(y_vecs)
                        # yield {'tok_id_seq_batch': tok_id_seq_batch, 'input_mask': input_mask}, y_vecs
                        yield {'batch_id': batch_id, 'tok_id_seq_batch': tok_id_seq_batch,
                               'tok_id_seqs_repeat': tok_id_seqs_repeat_ragged,
                               'text_ids': text_ids,
                               'labels': tf.constant(label_strs),
                               'tmp': tf.constant([[1, 2], [3, 4]], tf.int32),
                               }, y_vecs
                        text_ids, tok_id_seqs, tok_id_seqs_repeat, y_vecs = list(), list(), list(), list()
                        label_strs = list()
                        batch_id += 1
                        # y_vecs = list()
                if len(tok_id_seqs) > 0:
                    # tok_id_seq_batch, input_mask = get_padded_bert_input(tok_id_seqs)
                    tok_id_seq_batch = tf.ragged.constant(tok_id_seqs)
                    tok_id_seqs_repeat_ragged = tf.ragged.constant(tok_id_seqs_repeat)
                    # y_vecs_tensor = tf.concat(y_vecs)
                    # yield {'tok_id_seq_batch': tok_id_seq_batch, 'input_mask': input_mask}, y_vecs
                    yield {'batch_id': batch_id, 'tok_id_seq_batch': tok_id_seq_batch,
                           'tok_id_seqs_repeat': tok_id_seqs_repeat_ragged,
                           'text_ids': text_ids,
                           'labels': tf.constant(label_strs),
                           'tmp': tf.constant([[1, 2], [3, 4]], tf.int32),
                           }, y_vecs

        # for v in iter(tok_id_seq_gen()):
        #     print(v)
        dataset = tf.data.Dataset.from_generator(
            tok_id_seq_gen,
            output_signature=(
                {
                    'batch_id': tf.TensorSpec(shape=None, dtype=tf.int32),
                    'tok_id_seq_batch': tf.RaggedTensorSpec(dtype=tf.int32, ragged_rank=1),
                    'tok_id_seqs_repeat': tf.RaggedTensorSpec(dtype=tf.int32, ragged_rank=1),
                    'text_ids': tf.TensorSpec(shape=None, dtype=tf.int32),
                    'labels': tf.TensorSpec(shape=None, dtype=tf.string),
                    'tmp': tf.TensorSpec(shape=None, dtype=tf.int32),
                    # 'block_emb': tf.TensorSpec(shape=block_emb_shape, dtype=tf.float32)},
                },
                tf.TensorSpec(shape=None, dtype=tf.float32)))

        return dataset


def init_pre_load_data(block_emb_file, block_labels_file, type_id_dict):
    # num_block_records = 13353718
    num_block_records = 2000000
    n_block_rec_parts = [2670743, 5341486, 8012229, 10682972, 13353718]
    var_name = "block_emb"
    # checkpoint_path = os.path.join(retriever_module_path, "encoded", "encoded.ckpt")
    # np_db = tf.train.load_checkpoint(checkpoint_path).get_tensor(var_name)[:4000000]
    # block_emb_file = os.path.join(config.DATA_DIR, 'realm_data/realm_blocks/block_emb_2m.pkl')
    # block_emb_file = os.path.join(config.DATA_DIR, 'ultrafine/rlm_fet/enwiki-20151002-type-sents-2m-emb.pkl')
    # block_records_path = os.path.join(data_dir, 'realm_data/blocks.tfr')
    pre_load_data['np_db'] = datautils.load_pickle_data(block_emb_file)
    pre_load_data['labels'] = None
    if block_labels_file is not None:
        z_labels = list()
        with open(block_labels_file, encoding='utf-8') as f:
            for i, line in enumerate(f):
                label = line.strip()
                tid = type_id_dict.get(label, 0)
                z_labels.append(tid)
        pre_load_data['labels'] = np.array(z_labels, np.int32)


def __setup_logging(name, to_file):
    logger = tf.get_logger()
    # logger.setLevel('ERROR')
    logger.setLevel('INFO')
    logger.propagate = False
    if to_file:
        import datetime
        str_today = datetime.date.today().strftime('%y-%m-%d')
        log_file = os.path.join(log_dir, '{}-{}-{}.log'.format(
            name, str_today, config.MACHINE_NAME)) if to_file else None
        logger.addHandler(logging.FileHandler(log_file, mode='a'))
        logger.info('logging to {}'.format(log_file))


def predict_results(estimator, input_fn):
    results_file = os.path.join(config.DATA_DIR, 'tmp/uf_wia_results_200.txt')
    qemb_file = os.path.join(config.DATA_DIR, 'realm_output/uf_test_qembs.pkl')
    fout = open(results_file, 'w', encoding='utf-8')
    qembs = list()
    for i, pred in enumerate(estimator.predict(input_fn)):
        x = {'text_id': int(pred['text_ids']), 'block_ids': [int(v) for v in pred['block_ids']]}
        qembs.append(pred['qemb'])
        fout.write('{}\n'.format(json.dumps(x)))
        # print(pred)
        if i > 2:
            break
        if i % 100 == 0:
            print(i)
    fout.close()

    qembs = np.array(qembs)
    datautils.save_pickle_data(qembs, qemb_file)


def train_fet(block_records_path, block_emb_file, block_labels_file, model_dir, mode, log_file_name):
    __setup_logging(log_file_name, mode == 'train')
    logging.info(block_records_path)
    logging.info(block_emb_file)
    logging.info(model_dir)
    logging.info(mode)
    # logfile = os.path.join(output_dir, 'log/realm_et.log')
    # logger = tf.get_logger()
    # # logger.setLevel('ERROR')
    # logger.setLevel('INFO')
    # logger.addHandler(logging.FileHandler(logfile, mode='a'))
    # logger.propagate = False

    # run_train()

    batch_size = 1
    retriever_beam_size = 5
    num_train_steps = 100000
    n_train_repeat = 100
    save_checkpoints_steps = 1000
    log_step_count_steps = 100
    tf_random_seed = 1355
    embedder_module_path = os.path.join(data_dir, 'realm_data/cc_news_pretrained/embedder')
    reader_module_path = os.path.join(data_dir, 'realm_data/cc_news_pretrained/bert')
    vocab_file = os.path.join(reader_module_path, 'assets/vocab.txt')
    # model_dir = os.path.join(config.OUTPUT_DIR, 'tmp/tmpmodels')
    # model_dir = os.path.join(output_dir, 'etdmodels')
    type_vocab_file = os.path.join(config.DATA_DIR, 'ultrafine/uf_data/ontology/types.txt')

    tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case=True)
    sep_tok_id = tokenizer.convert_tokens_to_ids(['[SEP]'])[0]
    print('sep token id', sep_tok_id)
    types, type_id_dict = datautils.load_vocab_file(type_vocab_file)
    n_types = len(types)

    params = {
        'lr': 1e-5, 'batch_size': batch_size, 'max_seq_len': 256, 'bert_dim': 768,
        'retriever_beam_size': retriever_beam_size, 'n_types': n_types,
        'sep_tok_id': sep_tok_id, 'embedder_module_path': embedder_module_path,
        'reader_module_path': reader_module_path, 'num_train_steps': num_train_steps,
        'train_log_steps': 100,
        'eval_log_steps': 500,
        'num_block_records': 2000000,
        'block_records_path': block_records_path,
    }

    assert batch_size == 1
    init_pre_load_data(block_emb_file, block_labels_file, type_id_dict)
    # print(pre_load_data['np_db'].shape)
    params['num_block_records'] = pre_load_data['np_db'].shape[0]
    # exit()
    input_data = InputData(batch_size, tokenizer, types, type_id_dict, retriever_beam_size, n_train_repeat)

    model_fn_use = model_fn if block_labels_file is None else model_fn_zlabels

    run_config = tf.estimator.RunConfig(
        model_dir=model_dir,
        log_step_count_steps=log_step_count_steps,
        save_checkpoints_steps=save_checkpoints_steps,
        save_checkpoints_secs=None,
        tf_random_seed=tf_random_seed)
    estimator = tf.estimator.Estimator(
        config=run_config,
        model_fn=model_fn_use,
        params=params,
        model_dir=model_dir)
    # estimator.train(input_fn)
    # estimator.evaluate(input_fn)

    train_spec = tf.estimator.TrainSpec(
        input_fn=input_data.input_fn_train,
        max_steps=num_train_steps)
    eval_spec = tf.estimator.EvalSpec(
        name="default",
        input_fn=input_data.input_fn_test,
        # exporters=exporters,
        # start_delay_secs=FLAGS.eval_start_delay_secs,
        # throttle_secs=FLAGS.eval_throttle_secs
        )

    # estimator.evaluate(input_data.input_fn_test)
    if mode == 'train':
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    elif mode == 'predict':
        predict_results(estimator, input_data.input_fn_test)
