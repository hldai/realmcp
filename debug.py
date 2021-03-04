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
import tensorflow as tf
embedder_module_path = '/data/hldai/data/realm_data/cc_news_pretrained/embedder'
checkpoint_path = os.path.join(embedder_module_path, "encoded", "encoded.ckpt")
with tf.device("/cpu:0"):
    np_db = tf.train.load_checkpoint(checkpoint_path).get_tensor('block_emb')
    print(np_db.shape)
# blocks_dataset = tf.data.TFRecordDataset(
#     '/data/hldai/data/realm_data/blocks.tfr', buffer_size=512 * 1024 * 1024)
# cnt = 0
# for i, x in enumerate(blocks_dataset):
#     # print(x)
#     # if i > 1:
#     #     break
#     cnt += 1
#     if cnt % 100000 == 0:
#         print(cnt)
# print(cnt)
