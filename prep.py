import os
import tensorflow as tf
from locbert import tokenization
import config


def doc_texts_to_token_ids():
    tfr_text_docs_file = os.path.join(config.DATA_DIR, 'realm_data/blocks.tfr')
    reader_module_path = '/data/hldai/data/realm_data/cc_news_pretrained/bert'
    vocab_file = os.path.join(reader_module_path, 'assets/vocab.txt')
    tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case=True)

    blocks_dataset = tf.data.TFRecordDataset(tfr_text_docs_file, buffer_size=512 * 1024 * 1024)

    for i, v in enumerate(blocks_dataset):
        # print(v)
        v = v.numpy()
        v = v.decode('utf-8')
        tokens = tokenizer.tokenize(v)
        print(len(tokens))
        if i > 1:
            break


# doc_texts_to_token_ids()

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

data_path = 'd:/data/tmp/tmp.tfdata'

