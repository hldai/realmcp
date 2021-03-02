import os
import tensorflow as tf
import tensorflow_hub as hub
from bert import tokenization, optimization
from exp import sessfetexp


# # run_train()
# reader_module_path = '/data/hldai/data/realm_data/cc_news_pretrained/bert'
# vocab_file = os.path.join(reader_module_path, 'assets/vocab.txt')
#
# # tokenizer, vocab_lookup_table = bert_utils.get_tf_tokenizer(reader_module_path)
# tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case=True)
# tokens = tokenizer.tokenize('He is a teacher.')
# print(tokens)
# tokens_full = ['[CLS]'] + tokens + ['[SEP]']
# print(tokenizer.convert_tokens_to_ids(tokens_full))
#
# # mode = tf.estimator.ModeKeys.TRAIN
# mode = tf.estimator.ModeKeys.PREDICT
# reader_module = hub.Module(
#     reader_module_path,
#     tags={"train"} if mode == tf.estimator.ModeKeys.TRAIN else {},
#     trainable=True)
#
# token_ids = tf.constant([[101, 2002, 2003, 1037, 3836, 1012, 102]], dtype=tf.int32)
# mask = tf.constant([[1, 1, 1, 1, 1, 1, 1]], dtype=tf.int32)
# segment_ids = tf.constant([[0, 0, 0, 0, 0, 0, 0]], dtype=tf.int32)
#
# concat_outputs = reader_module(
#     dict(
#         input_ids=token_ids,
#         input_mask=mask,
#         segment_ids=segment_ids),
#     signature="tokens",
#     as_dict=True)
#
# concat_token_emb = concat_outputs["sequence_output"]
#
# sess = tf.compat.v1.Session()
# init = tf.compat.v1.global_variables_initializer()
# sess.run(init)
# # Evaluate the tensor `c`.
# print(sess.run(concat_token_emb))  # prints 30.0

sessfetexp.train_fet_sess()
