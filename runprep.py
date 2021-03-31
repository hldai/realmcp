import os
import tensorflow as tf
import numpy as np
from locbert import tokenization
import config
from utils import datautils


def doc_texts_to_token_ids():
    max_seq_len = 128
    output_file = os.path.join(config.DATA_DIR, 'realm_data/blocks_tok_id_seqs.pkl')
    tfr_text_docs_file = os.path.join(config.DATA_DIR, 'realm_data/blocks.tfr')
    reader_module_path = '/data/hldai/data/realm_data/cc_news_pretrained/bert'
    vocab_file = os.path.join(reader_module_path, 'assets/vocab.txt')
    tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case=True)

    blocks_dataset = tf.data.TFRecordDataset(tfr_text_docs_file, buffer_size=512 * 1024 * 1024)

    tok_id_seqs = list()
    for i, v in enumerate(blocks_dataset):
        # print(v)
        v = v.numpy()
        v = v.decode('utf-8')
        tokens = tokenizer.tokenize(v)
        # print(tokens)
        # print(len(tokens))
        token_ids = np.array(tokenizer.convert_tokens_to_ids(tokens), dtype=np.int32)
        # print(type(token_ids))
        if len(token_ids) > max_seq_len:
            token_ids = token_ids[:max_seq_len]
        tok_id_seqs.append(token_ids)
        # if i > 3:
        #     break
        if i % 10000 == 0:
            print(i)

    datautils.save_pickle_data(tok_id_seqs, output_file)


def save_ragged_vals_to_dataset(vals_list, output_path, concat_all=True):
    # print(vals_list)
    def data_gen():
        if concat_all:
            yield tf.ragged.constant(vals_list, dtype=tf.int32)
        else:
            for i, vals in enumerate(vals_list):
                if i % 10000 == 0:
                    print(i)
                yield tf.ragged.constant([vals], dtype=tf.int32)

    dataset = tf.data.Dataset.from_generator(
        data_gen, output_signature=tf.RaggedTensorSpec(ragged_rank=1, dtype=tf.int32))

    # for v in dataset:
    #     print(v)
    print('saving to', output_path)
    tf.data.experimental.save(dataset, output_path, shard_func=lambda x: np.int64(0))
    print('saved')


def save_doc_tok_id_seqs_singleall():
    n_parts = 5

    output_path = os.path.join(config.DATA_DIR, 'tmp/blocks_tok_id_seqs_l128_sa4m.tfd')
    blocks_list = datautils.load_pickle_data(os.path.join(config.DATA_DIR, 'realm_data/blocks_tok_id_seqs.pkl'))
    n_blocks = len(blocks_list)
    print('blocks list loaded', n_blocks)
    blocks_list = blocks_list[:4000000]
    save_ragged_vals_to_dataset(blocks_list, output_path, concat_all=False)


def save_doc_tok_id_seqs_to_parts():
    n_parts = 5

    output_path_prefix = os.path.join(config.DATA_DIR, 'realm_data/blocks_tok_id_seqs_l128/blocks_tok_id_seqs_l128_p')
    blocks_list = datautils.load_pickle_data(os.path.join(config.DATA_DIR, 'realm_data/blocks_tok_id_seqs.pkl'))
    n_blocks = len(blocks_list)
    print('blocks list loaded', n_blocks)

    # # blocks_list = [[3, 4], [5, 3, 8], [8]]
    # for i, v in enumerate(blocks_list):
    #     print(v)
    #     if i > 3:
    #         break
    # blocks_list = blocks_list[:10000]
    n_blocks_per_part = n_blocks // n_parts
    for i in range(n_parts):
        pbeg = i * n_blocks_per_part
        pend = (i + 1) * n_blocks_per_part if i < n_parts - 1 else n_blocks
        print(i, pbeg, pend)
        output_path = '{}{}.tfd'.format(output_path_prefix, i)
        save_ragged_vals_to_dataset(blocks_list[pbeg:pend], output_path, concat_all=True)
        # if i >= 1:
        #     break

    # def data_gen():
    #     for i, vals in enumerate(blocks_list):
    #         if i % 10000 == 0:
    #             print(i)
    #         yield tf.ragged.constant([vals], dtype=tf.int32)
    #
    # dataset = tf.data.Dataset.from_generator(
    #     data_gen,
    #     output_signature=tf.RaggedTensorSpec(ragged_rank=1, dtype=tf.int32))
    #
    # # for v in dataset:
    # #     print(v)
    # print('saving')
    # tf.data.experimental.save(dataset, '/data/hldai/data/tmp/tmp.tfdata', shard_func=lambda x: np.int64(0))
    # print('saved')


def save_doc_tok_id_seqs_whole():
    n_docs = 40000
    output_path = os.path.join(config.DATA_DIR, 'realm_data/blocks_tok_id_seqs_l128_4k.tfd')
    blocks_list = datautils.load_pickle_data(os.path.join(config.DATA_DIR, 'realm_data/blocks_tok_id_seqs.pkl'))
    blocks_list = blocks_list[:n_docs]
    save_ragged_vals_to_dataset(blocks_list, output_path)


def sample_from_realm_blocks():
    output_block_records_path = os.path.join(config.DATA_DIR, 'realm_data/blocks_2m.tfr')
    output_block_emb_path = os.path.join(config.DATA_DIR, 'realm_data/block_emb_2m.pkl')
    block_records_path = os.path.join(config.DATA_DIR, 'realm_data/blocks.tfr')
    retriever_module_path = os.path.join(config.DATA_DIR, 'realm_data/cc_news_pretrained/embedder')
    var_name = "block_emb"
    checkpoint_path = os.path.join(retriever_module_path, "encoded", "encoded.ckpt")
    np_db = tf.train.load_checkpoint(checkpoint_path).get_tensor(var_name)
    n_docs = np_db.shape[0]
    n_keep = 2000000
    idxs = np.random.permutation(np.arange(n_docs))[:n_keep]

    idxs = np.sort(idxs)
    datautils.save_pickle_data(np_db[idxs], output_block_emb_path)
    print(idxs[:10])
    # exit()

    idxs = set(idxs)
    # print(idxs[:10])
    # exit()

    blocks_dataset = tf.data.TFRecordDataset(
        block_records_path, buffer_size=512 * 1024 * 1024)
    with tf.io.TFRecordWriter(output_block_records_path) as file_writer:
        for i, v in enumerate(blocks_dataset):
            if i not in idxs:
                continue
            # # features = tf.train.Features(feature=by)
            # feature = {'': bytes_feature([v.numpy()])}
            # # print(feature)
            # # features = {'text': tf.train.Features(feature=[v.numpy()])}
            # features = tf.train.Features(feature=feature)
            # example = tf.train.Example(features=features)
            # print('TTTTTTTTTTTTTT', type(example.SerializeToString()))
            # exit()
            file_writer.write(v.numpy())
            # print(v)
            if i % 100000 == 0:
                print(i)
            #     break


# sample_from_realm_blocks()
# doc_texts_to_token_ids()
# save_doc_tok_id_seqs_to_parts()
# save_doc_tok_id_seqs_whole()
# save_doc_tok_id_seqs_singleall()


# blocks_dataset = tf.data.TFRecordDataset(
#     os.path.join(config.DATA_DIR, 'realm_data/blocks_2m.tfr'), buffer_size=512 * 1024 * 1024)
# # blocks_dataset = tf.data.TFRecordDataset(
# #     os.path.join(config.DATA_DIR, 'realm_data/blocks.tfr'), buffer_size=512 * 1024 * 1024)
# # blocks_dataset.batch(3)
# for i, v in enumerate(blocks_dataset.batch(3)):
#     print(v)
#     break
#     # if i > 5:
#     #     break

# import numpy as np
# example_path = '/data/hldai/data/tmp/tmp.tfr'
# with tf.io.TFRecordWriter(example_path) as file_writer:
#     # file_writer.write(tf.ragged.constant([[2], [3, 4]]))
#     # file_writer.write(tf.ragged.constant([[2, 7], [1, 3, 4]]))
#     for _ in range(4):
#         x, y = np.random.random(), np.random.random()
#
#         record_bytes = tf.train.Example(features=tf.train.Features(feature={
#             "x": tf.train.Feature(float_list=tf.train.FloatList(value=[x])),
#             "y": tf.train.Feature(float_list=tf.train.FloatList(value=[y])),
#         })).SerializeToString()
#         file_writer.write(record_bytes)
#
# dataset = tf.data.TFRecordDataset(example_path, buffer_size=512 * 1024 * 1024)
# dataset = dataset.batch(4)
# print(tf.data.experimental.get_single_element(dataset))
# # for v in dataset:
# #     print(v)

# data_path = 'd:/data/tmp/tmp.tfdata'
