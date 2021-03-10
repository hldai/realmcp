import tensorflow as tf
import time
import numpy as np
import os
import config
from utils import datautils, bert_utils


def save_ragged_vals_to_dataset(vals_list, output_path):
    # print(vals_list)
    def data_gen():
        # for i, vals in enumerate(vals_list):
        #     if i % 10000 == 0:
        #         print(i)
        yield tf.ragged.constant(vals_list, dtype=tf.int32)

    dataset = tf.data.Dataset.from_generator(
        data_gen, output_signature=tf.RaggedTensorSpec(ragged_rank=1, dtype=tf.int32))

    # for v in dataset:
    #     print(v)
    print('saving to', output_path)
    tf.data.experimental.save(dataset, output_path, shard_func=lambda x: np.int64(0))
    print('saved')


def save_ragged_tensors_dataset():
    blocks_list = datautils.load_pickle_data(os.path.join(config.DATA_DIR, 'realm_data/blocks_tok_id_seqs.pkl'))
    print('blocks list loaded', len(blocks_list))
    # # blocks_list = [[3, 4], [5, 3, 8], [8]]
    # for i, v in enumerate(blocks_list):
    #     print(v)
    #     if i > 3:
    #         break
    blocks_list = blocks_list[:10]
    save_ragged_vals_to_dataset(blocks_list, '/data/hldai/data/tmp/tmp.tfdata')

    # def data_gen():
        # for i, vals in enumerate(blocks_list):
        #     if i % 10000 == 0:
        #         print(i)
        #     yield tf.ragged.constant([vals], dtype=tf.int32)

    # dataset = tf.data.Dataset.from_generator(
    #     data_gen,
    #     output_signature=tf.RaggedTensorSpec(ragged_rank=1, dtype=tf.int32))

    # for v in dataset:
    #     print(v)
    # print('saving')
    # tf.data.experimental.save(dataset, '/data/hldai/data/tmp/tmp.tfdata', shard_func=lambda x: np.int64(0))
    # print('saved')


def load_rt_dataset_single_elem(dataset_path):
    dataset = tf.data.experimental.load(
        dataset_path,
        element_spec=tf.RaggedTensorSpec(ragged_rank=1, dtype=tf.int32))
    print('dataset loaded')
    # for i, v in enumerate(dataset):
    #     print(v)
    #     if i > 3:
    #         break
    # dataset = dataset.batch(n_records)
    print('batched')
    t = time.time()
    all_docs = tf.data.experimental.get_single_element(dataset)

    # print(type(all_docs))
    # print(all_docs[:3])
    print(time.time() - t)
    return all_docs


def load_dataset():
    dataset_path = '/data/hldai/data/tmp/tmp.tfdata'
    # dataset_path0 = os.path.join(config.DATA_DIR, 'realm_data/blocks_tok_id_seqs_l128/blocks_tok_id_seqs_l128_p0.tfd')
    # dataset_path = '/data/hldai/data/tmp/blocks_tok_id_seqs_l128.tfdata'
    # dataset_path1 = os.path.join(config.DATA_DIR, 'realm_data/blocks_tok_id_seqs_l128/blocks_tok_id_seqs_l128_p1.tfd')
    # docs_0 = load_rt_dataset_single_elem(dataset_path0)
    # docs_1 = load_rt_dataset_single_elem(dataset_path1)
    # print(docs_0.shape)
    # print(docs_1.shape)
    # docs = tf.concat((docs_0, docs_1), axis=0)
    # print(docs.shape)

    docs = load_rt_dataset_single_elem(dataset_path)
    print(docs)


def save_ragged_tensors_dataset_tmp():
    blocks_list = datautils.load_pickle_data(os.path.join(config.DATA_DIR, 'realm_data/blocks_tok_id_seqs.pkl'))
    print('blocks list loaded', len(blocks_list))
    # blocks_list = [[3, 4], [5, 3, 8], [8]]
    # for i, v in enumerate(blocks_list):
    #     print(v)
    #     if i > 3:
    #         break
    # blocks_list = blocks_list[:10000]

    def data_gen():
        yield tf.ragged.constant(blocks_list, dtype=tf.int32)
        # for i, vals in enumerate(blocks_list):
        #     if i % 10000 == 0:
        #         print(i)
        #     yield tf.ragged.constant([vals], dtype=tf.int32)

    dataset = tf.data.Dataset.from_generator(
        data_gen,
        output_signature=tf.RaggedTensorSpec(ragged_rank=1, dtype=tf.int32))

    # for v in dataset:
    #     print(v)
    print('saving')
    tf.data.experimental.save(
        dataset, '/data/hldai/data/tmp/blocks_tok_id_seqs_l128.tfdata', shard_func=lambda x: np.int64(0))
    print('saved')


def load_dataset_parts(dataset_path_prefix, n_parts):
    vals_list = list()
    for i in range(n_parts):
        dataset_path = '{}{}.tfd'.format(dataset_path_prefix, i)
        vals = load_rt_dataset_single_elem(dataset_path)
        vals_list.append(vals)
    return tf.concat(vals_list, axis=0)


n_records = 13353718
# save_ragged_tensors_dataset()
# save_ragged_tensors_dataset_tmp()
# load_dataset()

# dataset_path = os.path.join(config.DATA_DIR, 'realm_data/blocks_tok_id_seqs_l128_4m.tfd')
# blocks = load_rt_dataset_single_elem(dataset_path)
# print(blocks[:3])
# print(blocks.shape)

from locbert import tokenization

block_records_path = os.path.join(config.DATA_DIR, 'realm_data/blocks.tfr')
reader_module_path = '/data/hldai/data/realm_data/cc_news_pretrained/bert'
vocab_file = os.path.join(reader_module_path, 'assets/vocab.txt')
bert_tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case=True)
blocks_dataset = tf.data.TFRecordDataset(
    block_records_path, buffer_size=512 * 1024 * 1024)
blocks_dataset = blocks_dataset.batch(10)

tokenizer, vocab_lookup_table = bert_utils.get_tf_tokenizer(reader_module_path)
for i, v in enumerate(blocks_dataset):
    # print(v)
    v = tf.reshape(v, (-1, 5))
    question_token_ids = tokenizer.tokenize(v)
    # print(question_token_ids)
    # break

    question_token_ids = tf.cast(
        question_token_ids.merge_dims(2, 3).to_tensor(), tf.int32)
    print(question_token_ids)
    break
    question_token_ids = question_token_ids[:, :10]
    print(question_token_ids)

    tmp = tf.ragged.constant([[2, 3, 7], [1]], dtype=tf.int32)
    concat_inds = tf.concat([tmp, question_token_ids], axis=1)
    concat_inds_tensor = concat_inds.to_tensor()
    # print(concat_inds.to_tensor())
    print(concat_inds_tensor)
    print(concat_inds_tensor[:, -1])
    reach_max_len = tf.equal(concat_inds_tensor[:, -1], tf.constant(0, tf.int32))
    reach_max_len = 1 - tf.cast(reach_max_len, tf.int32)
    # print(tf.equal(concat_inds_tensor[:, -1], tf.constant(0, tf.int32)))
    inds_shape = tf.shape(concat_inds_tensor)
    print(reach_max_len)
    reach_max_len = tf.reshape(reach_max_len, (-1, 1))
    seps_tensor = tf.ones_like(reach_max_len) * reach_max_len
    print(seps_tensor)
    print(tf.concat((concat_inds_tensor, seps_tensor), axis=1))
    # print(concat_inds[])
    # print(bert_tokenizer.convert_tokens_to_ids(bert_tokenizer.tokenize(v.numpy().decode('utf-8'))))
    # print(bert_tokenizer.convert_ids_to_tokens(question_token_ids.numpy()[0]))
    if i > 1:
        break

# vals = tf.constant(np.random.uniform(-1, 1, (3, 5, 4)))
# print(vals)
# print(vals[:, 0, :])

# from tensorflow.keras import layers
# vals = tf.constant([[3, 4, 0], [1, 0, 0], [7, 7, 8], [4, 9, 0]], tf.float32)
# print(vals)
# vals_shape = tf.shape(vals)
# print(tf.range(vals_shape[0]))
# # bm = tf.sequence_mask([2, 1, 3, 2], vals_shape[1])
# bm = tf.sequence_mask(tf.range(vals_shape[0]), vals_shape[1])
# print(tf.cast(bm, tf.int32))

# blocks_list = [[3, 4], [7], [1, 8, 1]]
# blocks = tf.ragged.constant(blocks_list, dtype=tf.int32)
# tf.repeat(blocks, [2, 3, 2])
# print('blocks done')
# with tf.device('/cpu:0'):
#     retrieved_block_ids = tf.constant([[0, 2], [198, 10008]])
#     retrieved_blocks = tf.gather(blocks, retrieved_block_ids)
# print(retrieved_blocks)
