import os
import collections
import tensorflow as tf
import tensorflow_hub as hub
from bert import tokenization, optimization
from orqa.utils import scann_utils


RetrieverOutputs = collections.namedtuple("RetrieverOutputs", ["logits", "blocks"])


def retrieve(query_token_id_seqs, embedder_path, mode, retriever_beam_size):
    print('RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR', mode)
    """Do retrieval."""
    retriever_module = hub.Module(
        embedder_path,
        tags={"train"} if mode == tf.estimator.ModeKeys.TRAIN else {},
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


def model_fn(features, labels, mode, params):
    embedder_module_path = '/data/hldai/data/realm_data/cc_news_pretrained/embedder'
    reader_module_path = '/data/hldai/data/realm_data/cc_news_pretrained/bert'
    retriever_beam_size = 5
    lr = 1e-5
    num_train_steps = 10

    token_ids = tf.constant([[101, 2002, 2003, 1037, 3836, 1012, 102]], dtype=tf.int32)
    with tf.device("/cpu:0"):
        retriever_outputs = retrieve(token_ids, embedder_module_path, mode, retriever_beam_size)

    predictions = retriever_outputs
    loss = tf.reduce_mean(retriever_outputs)
    eval_metric_ops = None

    logging_hook = tf.train.LoggingTensorHook({"pred": predictions}, every_n_iter=5)

    train_op = optimization.create_optimizer(
        loss=loss,
        init_lr=lr,
        num_train_steps=num_train_steps,
        num_warmup_steps=min(10000, max(100,
                                        int(num_train_steps / 10))),
        use_tpu=False)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        predictions=predictions,
        # training_hooks=[logging_hook],
        evaluation_hooks=[logging_hook],
        eval_metric_ops=eval_metric_ops)


def input_fn():
    # feats = tf.random_uniform((100, 3))
    # labels = tf.random_uniform(100)
    # return feats, labels
    return 0


def train_fet():
    # run_train()
    retriever_beam_size = 5
    num_train_steps = 1000
    num_eval_steps = 1000
    embedder_module_path = '/data/hldai/data/realm_data/cc_news_pretrained/embedder'
    reader_module_path = '/data/hldai/data/realm_data/cc_news_pretrained/bert'
    model_dir = '/data/hldai/data/tmp/tmpmodels'
    vocab_file = os.path.join(reader_module_path, 'assets/vocab.txt')
    params = {'batch_size': 4}

    var_name = "block_emb"
    checkpoint_path = os.path.join(embedder_module_path, "encoded", "encoded.ckpt")
    with tf.device("/cpu:13"):
        np_db = tf.train.load_checkpoint(checkpoint_path).get_tensor(var_name)
        init_db = tf.py_func(lambda: np_db, [], tf.float32)
        init_db.set_shape(np_db.shape)
        tf_db = tf.get_local_variable(var_name, initializer=init_db)
    print(type(tf_db))
    print(type(np_db))
    # print(tf.size(np_db))
    print(np_db.shape)
    print(tf.shape(tf_db))
    exit()

    # tokenizer, vocab_lookup_table = bert_utils.get_tf_tokenizer(reader_module_path)
    tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case=True)
    tokens = tokenizer.tokenize('He is a teacher.')
    print(tokens)
    tokens_full = ['[CLS]'] + tokens + ['[SEP]']
    print(tokenizer.convert_tokens_to_ids(tokens_full))

    run_config = tf.estimator.RunConfig(
        model_dir=model_dir,
        log_step_count_steps=5,
        save_checkpoints_steps=100,
        save_checkpoints_secs=None,
        tf_random_seed=1355)
    estimator = tf.estimator.Estimator(
        config=run_config,
        model_fn=model_fn,
        params=params,
        model_dir=model_dir)
    # estimator.train(input_fn)
    # estimator.evaluate(input_fn)

    train_spec = tf.estimator.TrainSpec(
        input_fn=input_fn,
        max_steps=num_train_steps)
    eval_spec = tf.estimator.EvalSpec(
        name="default",
        input_fn=input_fn,
        # exporters=exporters,
        # start_delay_secs=FLAGS.eval_start_delay_secs,
        # throttle_secs=FLAGS.eval_throttle_secs
        )

    # estimator.evaluate(input_fn)
    # tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

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
