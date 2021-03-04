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

import scann
import numpy as np
import time

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

    tf_db_vals = list()
    n_samples = np_db.shape[0]
    n_part = 5
    part_size = n_samples // n_part
    pbeg = 0
    for i in range(n_part):
        if i == n_part - 1:
            pend = n_samples
        else:
            pend = pbeg + part_size
        # print(pbeg, pend)
        tf_db_vals.append(tf.constant(np_db[pbeg:pend]))
        pbeg = pend
    # block_emb = tf.constant(np_db)
    block_emb = tf.concat(tf_db_vals, axis=0)

    # block_emb = tf.constant(np_db)

    searcher = scann.scann_ops.builder(block_emb, retriever_beam_size, "dot_product").tree(
        num_leaves=1000, num_leaves_to_search=100, training_sample_size=250000).score_ah(
        2, anisotropic_quantization_threshold=0.2).reorder(100).build()

    t = time.time()
    retrieved_block_ids, _ = searcher.search_batched(question_emb)
    print(retrieved_block_ids)
    print(time.time() - t)
