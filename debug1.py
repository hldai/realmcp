import tensorflow as tf


blocks_list = [[3, 4], [2], [4, 5, 7]]
blocks = tf.ragged.constant(blocks_list, dtype=tf.int32)
v = tf.constant([[-1], [-1], [-1]], tf.int32)
print(tf.concat([blocks, v], axis=1))
# retrieved_block_ids = tf.constant([[0, 2], [0, 1]])
# # retrieved_block_ids = tf.constant([0, 2])
# retrieved_block_ids = tf.reshape(retrieved_block_ids, shape=(-1))
# print(tf.rank(retrieved_block_ids))
# print(retrieved_block_ids)
# # print(tf.rank(retrieved_block_ids))
# retrieved_blocks = tf.gather(blocks, retrieved_block_ids)
# # retrieved_blocks = blocks[retrieved_block_ids]
# print(retrieved_blocks)
