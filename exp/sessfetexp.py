import os
import collections
import tensorflow as tf
import tensorflow_hub as hub
from bert import tokenization, optimization
from orqa.utils import scann_utils


RetrieverOutputs = collections.namedtuple("RetrieverOutputs", ["logits", "blocks"])


def retrieve(query_token_id_seqs, embedder_path, is_train, retriever_beam_size):
    """Do retrieval."""
    retriever_module = hub.Module(
        embedder_path,
        tags={"train"} if is_train else {},
        trainable=True)

    # [1, projection_size]
    question_emb = retriever_module(
        inputs=dict(
            input_ids=query_token_id_seqs,
            input_mask=tf.ones_like(query_token_id_seqs),
            segment_ids=tf.zeros_like(query_token_id_seqs)),
        signature="projected")
    return question_emb

    # block_emb, searcher = scann_utils.load_scann_searcher(
    #     var_name="block_emb",
    #     checkpoint_path=os.path.join(embedder_path, "encoded", "encoded.ckpt"),
    #     num_neighbors=retriever_beam_size)
    #
    # # [1, retriever_beam_size]
    # retrieved_block_ids, _ = searcher.search_batched(question_emb)
    #
    # # [1, retriever_beam_size, projection_size]
    # retrieved_block_emb = tf.gather(block_emb, retrieved_block_ids)
    #
    # # [retriever_beam_size]
    # retrieved_block_ids = tf.squeeze(retrieved_block_ids)
    #
    # # [retriever_beam_size, projection_size]
    # retrieved_block_emb = tf.squeeze(retrieved_block_emb)
    #
    # # [1, retriever_beam_size]
    # retrieved_logits = tf.matmul(question_emb, retrieved_block_emb, transpose_b=True)
    #
    # # [retriever_beam_size]
    # retrieved_logits = tf.squeeze(retrieved_logits, 0)

    # blocks_dataset = tf.data.TFRecordDataset(
    #     params["block_records_path"], buffer_size=512 * 1024 * 1024)
    # blocks_dataset = blocks_dataset.batch(
    #     params["num_block_records"], drop_remainder=True)
    # blocks = tf.get_local_variable(
    #     "blocks",
    #     initializer=tf.data.experimental.get_single_element(blocks_dataset))
    # retrieved_blocks = tf.gather(blocks, retrieved_block_ids)
    # return RetrieverOutputs(logits=retrieved_logits, blocks=retrieved_blocks)
    # return retrieved_logits


def train_fet_sess():
    # run_train()
    retriever_beam_size = 5
    embedder_module_path = '/data/hldai/data/realm_data/cc_news_pretrained/embedder'
    reader_module_path = '/data/hldai/data/realm_data/cc_news_pretrained/bert'
    vocab_file = os.path.join(reader_module_path, 'assets/vocab.txt')

    # tokenizer, vocab_lookup_table = bert_utils.get_tf_tokenizer(reader_module_path)
    tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case=True)
    tokens = tokenizer.tokenize('He is a teacher.')
    print(tokens)
    tokens_full = ['[CLS]'] + tokens + ['[SEP]']
    print(tokenizer.convert_tokens_to_ids(tokens_full))

    token_ids = tf.constant([[101, 2002, 2003, 1037, 3836, 1012, 102]], dtype=tf.int32)
    with tf.device("/cpu:0"):
        retriever_outputs = retrieve(token_ids, embedder_module_path, True, retriever_beam_size)

    # # mode = tf.estimator.ModeKeys.TRAIN
    # mode = tf.estimator.ModeKeys.PREDICT
    # reader_module = hub.Module(
    #     reader_module_path,
    #     tags={"train"} if mode == tf.estimator.ModeKeys.TRAIN else {},
    #     trainable=True)
    #
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
    # trainable_vars = tf.trainable_variables()
    sess = tf.compat.v1.Session()
    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)
    # print(sess.run(concat_token_emb))
    print(sess.run(retriever_outputs))
    for i in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES):
        print(i.name)  # i.name if you want just a name
