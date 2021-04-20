import os
import gzip
import tensorflow as tf
import config
from utils import datautils


def select_texts():
    import random

    # 76397770 sentences
    output_block_records_path = '/data/hldai/data/ultrafine/rlm_fet/enwiki-20151002-type-sents-2m-tmp.tfr'
    n_keep = 2000000
    n_total = 76397770
    rand_rate = n_keep / n_total
    print(rand_rate)
    with tf.io.TFRecordWriter(output_block_records_path) as file_writer:
        f = gzip.open('/data/hldai/data/ultrafine/enwiki-20151002-type-sents.txt.gz', 'rt', encoding='utf-8')
        # texts = list()
        cnt = 0
        for i, line in enumerate(f):
            if i % 1000000 == 0:
                print(i, cnt)
            v = random.uniform(0, 1)
            if v < rand_rate:
                file_writer.write(tf.constant(line.strip()).numpy())
                cnt += 1
            # if i > 1000:
            #     break
        f.close()
        print(cnt)


def pad_sep_to_tensor(tok_id_seqs, sep_tok_id):
    reach_max_len = tf.equal(tok_id_seqs[:, -1], tf.constant(0, tf.int32))
    reach_max_len = 1 - tf.cast(reach_max_len, tf.int32)
    reach_max_len = tf.reshape(reach_max_len, (-1, 1))
    seps_tensor = reach_max_len * sep_tok_id
    # print(seps_tensor)
    # print(tf.concat((q_doc_tok_id_seqs, seps_tensor), axis=1))
    tok_id_seqs = tf.concat((tok_id_seqs, seps_tensor), axis=1)

    is_zero = tf.cast(tf.equal(tok_id_seqs, tf.constant(0)), tf.int32)
    # print(is_zero)
    is_zero_cumsum = tf.cumsum(is_zero, axis=1)
    sep_tensor = tf.cast(tf.equal(is_zero_cumsum, tf.constant(1)), tf.int32) * sep_tok_id
    tok_id_seqs += sep_tensor
    return tok_id_seqs


def gen_emb_model_fn(features, labels, mode, params):
    import tensorflow_hub as hub
    from utils import bert_utils

    num_block_records = params['n_blocks']
    block_records_path = params['block_records_path']
    reader_module_path = params['reader_module_path']
    embedder_path = params['embedder_module_path']
    max_seq_len = 256

    blocks_dataset = tf.data.TFRecordDataset(
        block_records_path, buffer_size=512 * 1024 * 1024)
    blocks_dataset = blocks_dataset.batch(
        num_block_records, drop_remainder=True)
    blocks = tf.compat.v1.get_local_variable(
        "blocks",
        initializer=tf.data.experimental.get_single_element(blocks_dataset))
    retrieved_blocks = tf.gather(blocks, features['block_ids'])

    tokenizer, vocab_lookup_table = bert_utils.get_tf_tokenizer(reader_module_path)
    cls_token_id = tf.cast(vocab_lookup_table.lookup(tf.constant("[CLS]")), tf.int32)
    sep_token_id = tf.cast(vocab_lookup_table.lookup(tf.constant("[SEP]")), tf.int32)

    block_tok_id_seqs = tokenizer.tokenize(retrieved_blocks)
    block_tok_id_seqs = tf.cast(
        block_tok_id_seqs.merge_dims(1, 2).to_tensor(), tf.int32)
    batch_size = tf.shape(block_tok_id_seqs)[0]
    cls_tok_ids = tf.ones([batch_size, 1], tf.int32) * cls_token_id
    block_tok_id_seqs = tf.concat((cls_tok_ids, block_tok_id_seqs), axis=1)
    block_tok_id_seqs = block_tok_id_seqs[:, :max_seq_len - 1]
    block_tok_id_seqs = pad_sep_to_tensor(block_tok_id_seqs, sep_token_id)
    input_mask = 1 - tf.cast(tf.equal(block_tok_id_seqs, tf.constant(0)), tf.int32)

    retriever_module = hub.Module(
        embedder_path,
        tags={"train"} if mode == tf.estimator.ModeKeys.TRAIN else {},
        trainable=True)

    # [1, projection_size]
    block_emb = retriever_module(
        inputs=dict(
            input_ids=block_tok_id_seqs,
            # input_mask=tf.ones_like(query_token_id_seqs),
            input_mask=input_mask,
            segment_ids=tf.zeros_like(block_tok_id_seqs)),
        signature="projected")

    predictions = block_emb
    loss = tf.constant(1.0)
    logging_hook = tf.estimator.LoggingTensorHook({
        'id_seqs': block_emb,
        'rb': retrieved_blocks
    }, every_n_iter=1)
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=None,
        predictions=predictions,
        # prediction_hooks=[logging_hook],
        # training_hooks=[train_logging_hook],
        # evaluation_hooks=[logging_hook],
        # eval_metric_ops=eval_metric_ops
        )


def gen_embs():
    import numpy as np

    embedder_module_path = os.path.join(config.DATA_DIR, 'realm_data/cc_news_pretrained/embedder')
    reader_module_path = os.path.join(config.DATA_DIR, 'realm_data/cc_news_pretrained/bert')

    # output_emb_file = os.path.join(config.DATA_DIR, 'ultrafine/rlm_fet/enwiki-20151002-type-sents-2m-emb.pkl')
    # block_records_path = os.path.join(config.DATA_DIR, 'ultrafine/rlm_fet/enwiki-20151002-type-sents-2m.tfr')
    # params = {'n_blocks': 2000007, 'block_records_path': block_records_path, 'reader_module_path': reader_module_path,
    #           'embedder_module_path': embedder_module_path}

    output_emb_file = os.path.join(config.DATA_DIR, 'ultrafine/zoutput/webisa_full_uffilter_emb.pkl')
    block_records_path = os.path.join(config.DATA_DIR, 'ultrafine/zoutput/webisa_full_uffilter.tfr')
    params = {'n_blocks': 1671143, 'block_records_path': block_records_path, 'reader_module_path': reader_module_path,
              'embedder_module_path': embedder_module_path}

    logger = tf.get_logger()
    logger.setLevel('INFO')
    logger.propagate = False

    def input_fn():
        batch_size = 16

        def data_gen():
            block_ids = list()
            for block_id in range(params['n_blocks']):
                block_ids.append(block_id)
                if len(block_ids) >= batch_size:
                    yield {'block_ids': block_ids}
                    block_ids = list()
            if len(block_ids) > 0:
                yield {'block_ids': block_ids}

        dataset = tf.data.Dataset.from_generator(
            data_gen,
            output_signature=(
                {
                    'block_ids': tf.TensorSpec(shape=None, dtype=tf.int32),
                }))
        return dataset

    run_config = tf.estimator.RunConfig(
        model_dir=None,
        log_step_count_steps=100000,
        save_checkpoints_steps=None,
        save_checkpoints_secs=None,
        tf_random_seed=1973)
    estimator = tf.estimator.Estimator(
        config=run_config,
        model_fn=gen_emb_model_fn,
        params=params,
        model_dir=None)

    embs_list = list()
    for i, v in enumerate(estimator.predict(input_fn)):
        embs_list.append(v)
        # print(i, v)
        # if i > 10:
        #     break
        if i % 1000 == 0:
            print(i)

    datautils.save_pickle_data(np.array(embs_list, dtype=np.float32), output_emb_file)


def check_block_emb():
    emb_file = os.path.join(config.DATA_DIR, 'ultrafine/rlm_fet/enwiki-20151002-type-sents-2m-emb.pkl')
    block_emb = datautils.load_pickle_data(emb_file)
    print(block_emb.shape)


# select_texts()
gen_embs()
# check_block_emb()

# block_records_path = os.path.join(config.DATA_DIR, 'realm_data/realm_blocks/blocks_2m.tfr')
# block_records_path = os.path.join(config.DATA_DIR, 'ultrafine/rlm_fet/enwiki-20151002-type-sents-2m.tfr')
#
# blocks_dataset = tf.data.TFRecordDataset(
#     block_records_path, buffer_size=512 * 1024 * 1024)
# for i, x in enumerate(blocks_dataset):
#     print(x)
#     if i > 2:
#         break
# blocks_dataset = blocks_dataset.batch(
#     num_block_records, drop_remainder=True)
