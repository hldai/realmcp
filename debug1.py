import tensorflow as tf
import time
import numpy as np
import os
import config
from utils import datautils


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


n_records = 13353718
# save_ragged_tensors_dataset()
# save_ragged_tensors_dataset_tmp()
load_dataset()

# blocks = tf.ragged.constant(blocks_list, dtype=tf.int32)
# print('blocks done')
# with tf.device('/cpu:0'):
#     retrieved_block_ids = tf.constant([[0, 2], [198, 10008]])
#     retrieved_blocks = tf.gather(blocks, retrieved_block_ids)
# print(retrieved_blocks)

# v = tf.constant([[-1], [-1], [-1]], tf.int32)
# print(tf.concat([blocks, v], axis=1))
# retrieved_block_ids = tf.constant([[0, 2], [0, 1]])
# # retrieved_block_ids = tf.constant([0, 2])
# retrieved_block_ids = tf.reshape(retrieved_block_ids, shape=(-1))
# print(tf.rank(retrieved_block_ids))
# print(retrieved_block_ids)
# # print(tf.rank(retrieved_block_ids))
# retrieved_blocks = tf.gather(blocks, retrieved_block_ids)
# # retrieved_blocks = blocks[retrieved_block_ids]
# print(retrieved_blocks)
