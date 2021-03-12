import os
# from orqa.utils import bert_utils


def tf_test():
    import tensorflow as tf
    import tensorflow_hub as hub
    from locbert import tokenization

    reader_module_path = '/data/hldai/data/realm_data/cc_news_pretrained/locbert'
    vocab_file = os.path.join(reader_module_path, 'assets/vocab.txt')

    # tokenizer, vocab_lookup_table = bert_utils.get_tf_tokenizer(reader_module_path)
    tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case=True)
    tokens = tokenizer.tokenize('He is a teacher.')
    print(tokens)
    tokens_full = ['[CLS]'] + tokens + ['[SEP]']
    print(tokenizer.convert_tokens_to_ids(tokens_full))

    mode = tf.estimator.ModeKeys.TRAIN
    reader_module = hub.Module(
        reader_module_path,
        tags={"train"} if mode == tf.estimator.ModeKeys.TRAIN else {},
        trainable=True)

    token_ids = tf.constant([[101, 2002, 2003, 1037, 3836, 1012, 102]], dtype=tf.int32)
    mask = tf.constant([[1, 1, 1, 1, 1, 1, 1]], dtype=tf.int32)
    segment_ids = tf.constant([[0, 0, 0, 0, 0, 0, 0]], dtype=tf.int32)

    concat_outputs = reader_module(
        dict(
            input_ids=token_ids,
            input_mask=mask,
            segment_ids=segment_ids),
        signature="tokens",
        as_dict=True)

    concat_token_emb = concat_outputs["sequence_output"]

    a = tf.constant(3)
    b = tf.constant(4)
    c = a + b
    sess = tf.compat.v1.Session()
    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)
    # Evaluate the tensor `c`.
    # print(sess.run(concat_token_emb))  # prints 30.0
    print(sess.run(c))

    # tf.reset_default_graph()
    # sess = tf.compat.v1.Session()
    # init = tf.compat.v1.global_variables_initializer()
    # sess.run(init)
    # # print(sess.run(concat_token_emb))  # prints 30.0
    # print(sess.run(c))


def scann_test():
    import tensorflow as tf
    import numpy as np
    import scann
    from scann import ScannBuilder
    import time

    num_leaves = 1000
    num_leaves_to_search = 100
    training_sample_size = 100000
    dimensions_per_block = 2
    num_neighbors = 5
    question_emb_np = np.random.uniform(-1, 1, (2, 100))
    np_db = np.random.uniform(-1, 1, (10000000, 100)).astype(np.float32)

    with tf.device("/cpu:0"):
        question_emb = tf.constant(question_emb_np, tf.float32)
        tf_db = tf.constant(np_db)
        # searcher = scann.scann_ops_pybind.builder(np_db, 10, "dot_product").tree(
        #     num_leaves=1000, num_leaves_to_search=100, training_sample_size=250000).score_ah(
        #     2, anisotropic_quantization_threshold=0.2).reorder(100).build()
        searcher = scann.scann_ops.builder(tf_db, 10, "dot_product").tree(
            num_leaves=1000, num_leaves_to_search=100, training_sample_size=250000).score_ah(
            2, anisotropic_quantization_threshold=0.2).reorder(100).build()

        t = time.time()
        retrieved_block_ids, _ = searcher.search_batched(question_emb)
        print(retrieved_block_ids)
        print(time.time() - t)

    # queries = np.random.uniform(-1, 1, (5, 3)).astype(np.float32)
    # neighbors, distances = searcher.search_batched(queries)
    # print(neighbors)

    # init_db = tf.compat.v1.py_func(lambda: np_db, [], tf.float32)
    # init_db.set_shape(np_db.shape)
    # tf_db = tf.compat.v1.get_local_variable('v', initializer=init_db)
    # # print(tf_db)
    #
    # # def model_fn():
    # builder = ScannBuilder(
    #     db=tf_db,
    #     num_neighbors=num_neighbors,
    #     distance_measure="dot_product")
    # builder = builder.tree(
    #     num_leaves=num_leaves,
    #     num_leaves_to_search=num_leaves_to_search,
    #     training_sample_size=training_sample_size)
    # builder = builder.score_ah(dimensions_per_block=dimensions_per_block)
    #
    # searcher = builder.create_tf()

    t = time.time()
    vals_mat = np.matmul(question_emb_np, np.transpose(np_db))
    print(vals_mat.shape)
    for vals in vals_mat:
        idxs = np.argpartition(-vals, 10)[:10]
        idxs = idxs[np.argsort(-vals[idxs])]
        print(idxs)
    print(time.time() - t)
    # with tf.compat.v1.Session() as sess:
    #     retrieved_block_ids, _ = searcher.search_batched(question_emb)
    #     print(sess.run(retrieved_block_ids))

    # sess = tf.compat.v1.Session()
    # print(sess.run(z))


# scann_test()
# import logging
# import tensorflow_hub as hub
import tensorflow as tf
# import numpy as np
# tf.logging.set_verbosity(tf.logging.INFO)
# tf.logging.info('This is a log')

# tf.enable_eager_execution()

# N = 100
# # dictionary of arrays:
# metadata = {'m1': np.zeros(shape=(N,2)), 'm2': np.ones(shape=(N,3,5))}
# num_samples = N
#
# def meta_dict_gen():
#     for i in range(num_samples):
#         ls = {}
#         label = 0
#         for key, val in metadata.items():
#             ls[key] = val[i]
#         yield ls, label
#
# dataset = tf.data.Dataset.from_generator(
#     meta_dict_gen,
#     output_types=({k: tf.float32 for k in metadata}, tf.int32))
# # iter = dataset.make_one_shot_iterator()
# # next_elem = iter.get_next()
# # print(next_elem)
# for i, batch in enumerate(dataset.batch(1)):
#     # print(batch)
#     print(batch)

# import scann
import numpy as np
import time

def scann_test():
    data_dir = '/data/hldai/data'
    retriever_module_path = os.path.join(data_dir, 'realm_data/cc_news_pretrained/embedder')
    var_name = "block_emb"
    with tf.device("/cpu:0"):
        question_emb_np = np.random.uniform(-1, 1, (4, 128))
        question_emb = tf.constant(question_emb_np, tf.float32)
        checkpoint_path = os.path.join(retriever_module_path, "encoded", "encoded.ckpt")
        np_db = tf.train.load_checkpoint(checkpoint_path).get_tensor(var_name)
        print(np_db.shape)
        retriever_beam_size = 5

        # block_emb = tf.constant(np_db)
        block_emb = tf.compat.v1.get_variable('block_emb_tf', initializer=np_db)

        searcher = scann.scann_ops.builder(block_emb, retriever_beam_size, "dot_product").tree(
            num_leaves=1000, num_leaves_to_search=100, training_sample_size=250000).score_ah(
            2, anisotropic_quantization_threshold=0.2).reorder(100).build()

        t = time.time()
        retrieved_block_ids, _ = searcher.search_batched(question_emb)
        print(retrieved_block_ids)
        print(time.time() - t)


def loss_debug():
    import tensorflow_probability as tfp

    n_types = 13
    n_neibors = 5
    batch_size = 4
    r = 1.0
    zx_logits = tf.constant(np.random.uniform(-r, r, (batch_size, n_neibors)))
    yzx_logits = tf.constant(np.random.uniform(-r, r, (batch_size, n_neibors, n_types)))
    # print(zx_logits)
    # print(yzx_logits)
    log_softmax_zx_logits = tf.nn.log_softmax(zx_logits, axis=1)
    # print(tf.reduce_sum(tf.math.exp(log_softmax_zx_logits), axis=1))
    # exit()
    log_sig_yzx_logits = tf.math.log_sigmoid(yzx_logits)
    # print(log_sig_yzx_logits + tf.expand_dims(log_sig_zx_logits, 2))
    z_log_probs = log_sig_yzx_logits + tf.expand_dims(log_softmax_zx_logits, 2)
    log_probs = tf.reduce_logsumexp(z_log_probs, axis=1)
    # print(log_probs)
    # print(tf.shape(log_probs))
    # print(z_log_probs)
    # log_neg_probs = tfp.math.log1mexp(log_probs)
    # print(log_neg_probs)
    # print(tf.shape(log_neg_probs))

    # print(tf.math.exp(log_probs) + tf.math.exp(log_neg_probs))
    # print(tf.math.exp(tfp.math.log1mexp(tf.constant([0.1, 0.3, 0.9], tf.float32))))
    # print(tfp.math.log1mexp(tf.constant([0.1, 0.3, 0.9], tf.float32)))
    # print()

    # log_probs = tf.math.log(tf.constant([0.1, 0.3, 0.9]))
    # print(tf.exp(tf.math.log(tf.constant(0.9))))
    # print(tf.math.log(1 - tf.exp(tf.math.log(tf.constant(0.9)))))
    print('probs', tf.math.exp(log_probs))
    log_neg_probs = tfp.math.log1mexp(log_probs)
    print(tf.math.exp(log_probs) + tf.math.exp(log_neg_probs))

# scann_test()
#
# from orqa.utils import bert_utils
#
# data_dir = '/data/hldai/data'
# block_records_path = os.path.join(data_dir, 'realm_data/blocks.tfr')
# retriever_module_path = os.path.join(data_dir, 'realm_data/cc_news_pretrained/embedder')
# reader_module_path = os.path.join(data_dir, 'realm_data/cc_news_pretrained/bert')
# num_block_records = 13353718
#
# # block_ids = [[8922907, 6052548, 10062955, 3353143, 1761062],
# #              [4329926, 2385692, 3212458, 4258115, 4555483]]
# #              [6885852, 11160934, 3541819, 11471241, 6999494],
# #              [8884514, 4336603, 12356131, 5319352, 2385659]]
# block_ids = [8922907, 6052548, 10062955, 3353143, 1761062]
#
# block_ids = tf.compat.v1.get_variable('block_ids', initializer=block_ids)
# # print(block_ids)
#
# blocks_dataset = tf.data.TFRecordDataset(
#     block_records_path, buffer_size=512 * 1024 * 1024)
# blocks_dataset = blocks_dataset.batch(
#     num_block_records, drop_remainder=True)
# blocks = tf.compat.v1.get_variable(
#     "blocks",
#     initializer=tf.data.experimental.get_single_element(blocks_dataset))
# # blocks = tf.get_variable(
# #     "blocks",
# #     initializer=tf.data.experimental.get_single_element(blocks_dataset))
# # blocks = tf.constant(tf.data.experimental.get_single_element(blocks_dataset))
# retrieved_blocks = tf.gather(blocks, block_ids)
# # print(retrieved_blocks)
#
# tokenizer, vocab_lookup_table = bert_utils.get_tf_tokenizer(reader_module_path)
# print('get tokenizer')
#
# (orig_tokens, block_token_map, block_token_ids, blocks) = (
#     bert_utils.tokenize_with_original_mapping(blocks, tokenizer))
# print(block_token_ids)

def pad_sep_to_tensor(tok_id_seqs):
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


sep_tok_id = 102
# vals = tf.constant([[2, 3, 0, 0, 0], [1, 2, 3, 8, 5], [2, 3, 7, 1, 0]], tf.int32)
# print(vals)
# print(pad_sep_to_tensor(vals))
labels = tf.constant([[0, 1, 1, 0], [1, 0, 0, 1]], tf.float32)
preds = tf.constant([[0.9, 0.7, 0.3, 0.8], [0.9, 0.1, 0.1, 0.7]], tf.float32)
# preds_for_true_pos = preds * labels
# print(preds_for_true_pos)
# correct_pos = tf.cast(tf.less(tf.constant(0.5), preds_for_true_pos), tf.float32)
# print(correct_pos)
# n_corrects = tf.reduce_sum(correct_pos, axis=1)
# print(n_corrects)

# 3-1. Choose which unit is used to use for the operation.
# import tensorflow as tf
# from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
# from tensorflow.python.training import py_checkpoint_reader
#
#
# latest_ckp = '/data/hldai/data/tmp/tmpmodels/model.ckpt-3'
# # print_tensors_in_checkpoint_file(latest_ckp, all_tensors=True, tensor_name='')
#
# # try:
# reader = py_checkpoint_reader.NewCheckpointReader(latest_ckp)
# var_to_shape_map = reader.get_variable_to_shape_map()
# var_to_dtype_map = reader.get_variable_to_dtype_map()
# for key, value in sorted(var_to_shape_map.items()):
#     print("tensor: %s (%s) %s" % (key, var_to_dtype_map[key].name, value))
#     print(type(reader.get_tensor(key)))

import config

dl = tf.keras.layers.Dense(10)
vals = tf.constant(np.random.uniform(-1, 1, (2, 5)), tf.float32)
print(dl(vals))
print(dl.get_weights())

with tf.compat.v1.variable_scope("layer_name", reuse=True):
    weights = tf.compat.v1.get_variable("kernel") # do not specify
    print(weights)
