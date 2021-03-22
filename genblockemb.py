import gzip
import tensorflow as tf


def select_texts():
    import random

    # 76397770 sentences
    n_keep = 2000000
    n_total = 76397770
    f = gzip.open('/data/hldai/data/ultrafine/enwiki-20151002-type-sents.txt.gz', 'rt', encoding='utf-8')
    texts = list()
    cnt = 0
    rand_rate = n_keep / n_total
    for i, line in enumerate(f):
        if i % 1000000 == 0:
            print(i)
        cnt += 1
    f.close()
    print(cnt)


# select_texts()

# print(tf.constant('foo').numpy())
# bl = tf.train.BytesList(value=['foo'.encode('utf-8')])
# print(type(bl))
output_block_records_path = 'd:/data/tmp/tmp.tfr'
# with tf.io.TFRecordWriter(output_block_records_path) as file_writer:
#     file_writer.write(tf.constant('foo').numpy())
#     file_writer.write(tf.constant('car').numpy())
# print(tf.io.serialize_tensor('foo'))


blocks_dataset = tf.data.TFRecordDataset(
    output_block_records_path, buffer_size=512 * 1024 * 1024)
for x in blocks_dataset:
    print(x)
# blocks_dataset = blocks_dataset.batch(
#     num_block_records, drop_remainder=True)
