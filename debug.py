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


import tensorflow as tf
import config

block_records_path = os.path.join(config.DATA_DIR, 'ultrafine/zoutput/webisa_full_uffilter.tfr')
blocks_dataset = tf.data.TFRecordDataset(block_records_path, buffer_size=512 * 1024 * 1024)
cnt = 0
for i, v in enumerate(blocks_dataset):
    cnt += 1
    # print(v)
    # if i > 3:
    #     break
print(cnt)
# tmp_blocks = tf.constant(['i you date', 'sh ij ko', 'day in day'])
# answers = tf.constant(["date", 'day'])
# result = orqa_ops.has_answer(blocks=tmp_blocks, answers=answers)
# print(tf.cast(result, tf.int32))
# print(result)

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
