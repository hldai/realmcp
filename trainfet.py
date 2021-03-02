import os
import tensorflow as tf
import tensorflow_hub as hub
from bert import tokenization, optimization
# from orqa.utils import bert_utils


def train_input_fn():
    return 0


def model_fn(features, labels, mode, params):
    reader_module_path = '/data/hldai/data/realm_data/cc_news_pretrained/bert'

    train_op = None
    loss = tf.constant(337)
    # a = tf.constant([2, 3, 4])
    # b = tf.constant([2, 5, 4])
    # predictions = a + b
    eval_metric_ops = None

    reader_module = hub.Module(
        reader_module_path,
        tags={"train"} if mode == tf.estimator.ModeKeys.TRAIN else {},
        trainable=True)

    token_ids = tf.constant([[101, 2002, 2003, 1037, 3836, 1012, 102]], dtype=tf.int32)
    mask = tf.constant([[1, 1, 1, 1, 1, 1, 1]], dtype=tf.int32)
    segment_ids = tf.constant([[0, 0, 0, 0, 0, 0, 0]], dtype=tf.int32)

    concat_outputs = reader_module(
        dict(
            input_ids=token_ids,
            input_mask=mask,
            segment_ids=segment_ids),
        signature="tokens",
        as_dict=True)

    concat_token_emb = concat_outputs["sequence_output"]
    predictions = concat_token_emb

    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = optimization.create_optimizer(
            loss=loss,
            init_lr=params["learning_rate"],
            num_train_steps=params["num_train_steps"],
            num_warmup_steps=min(10000, max(100,
                                            int(params["num_train_steps"] / 10))),
            use_tpu=False)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        predictions=predictions,
        eval_metric_ops=eval_metric_ops)


def run_train():
    model_dir = '/data/hldai/data/tmp/tmpmodels'
    batch_size = 2
    num_train_steps = 10
    save_checkpoints_steps = None
    tf_random_seed = 155
    keep_checkpoint_max = 5
    params = dict()
    params["batch_size"] = batch_size
    params["learning_rate"] = 1e-5
    params["num_train_steps"] = 10

    run_config = tf.estimator.RunConfig(
        model_dir=model_dir,
        tf_random_seed=tf_random_seed,
        save_checkpoints_steps=save_checkpoints_steps,
        keep_checkpoint_max=keep_checkpoint_max)

    estimator = tf.estimator.Estimator(
        config=run_config,
        model_fn=model_fn,
        params=params,
        model_dir=model_dir)

    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn,
        max_steps=num_train_steps)

    # estimator.train(input_fn=train_input_fn)
    predictor = estimator.predict(input_fn=train_input_fn)
    for i, p in enumerate(predictor):
        print(p)
        if i > 0:
            break
    # tf.estimator.train(
    #     estimator=estimator,
    #     train_spec=train_spec)


# run_train()
reader_module_path = '/data/hldai/data/realm_data/cc_news_pretrained/bert'
vocab_file = os.path.join(reader_module_path, 'assets/vocab.txt')

# tokenizer, vocab_lookup_table = bert_utils.get_tf_tokenizer(reader_module_path)
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case=True)
tokens = tokenizer.tokenize('He is a teacher.')
print(tokens)
tokens_full = ['[CLS]'] + tokens + ['[SEP]']
print(tokenizer.convert_tokens_to_ids(tokens_full))

# mode = tf.estimator.ModeKeys.TRAIN
mode = tf.estimator.ModeKeys.PREDICT
reader_module = hub.Module(
    reader_module_path,
    tags={"train"} if mode == tf.estimator.ModeKeys.TRAIN else {},
    trainable=True)

token_ids = tf.constant([[101, 2002, 2003, 1037, 3836, 1012, 102]], dtype=tf.int32)
mask = tf.constant([[1, 1, 1, 1, 1, 1, 1]], dtype=tf.int32)
segment_ids = tf.constant([[0, 0, 0, 0, 0, 0, 0]], dtype=tf.int32)

concat_outputs = reader_module(
    dict(
        input_ids=token_ids,
        input_mask=mask,
        segment_ids=segment_ids),
    signature="tokens",
    as_dict=True)

concat_token_emb = concat_outputs["sequence_output"]

sess = tf.compat.v1.Session()
init = tf.compat.v1.global_variables_initializer()
sess.run(init)
# Evaluate the tensor `c`.
print(sess.run(concat_token_emb))  # prints 30.0
