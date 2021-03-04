import os
import logging
import tensorflow as tf
import tensorflow_hub as hub
from bert import tokenization
from locbert import optimization
# from orqa.utils import bert_utils
from exp import fetexp
from absl import app
from absl import flags


def train_input_fn():
    return 0


def model_fn(features, labels, mode, params):
    reader_module_path = '/data/hldai/data/realm_data/cc_news_pretrained/locbert'

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


def init_universal_logging(logfile='main.log', mode='a', to_stdout=True):
    handlers = list()
    if logfile is not None:
        handlers.append(logging.FileHandler(logfile, mode=mode))
    if to_stdout:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(format='%(asctime)s %(filename)s:%(lineno)s %(levelname)s - %(message)s',
                        datefmt='%y-%m-%d %H:%M:%S', handlers=handlers, level=logging.INFO)


def main(_):
    init_universal_logging(None)
    fetexp.train_fet()


if __name__ == "__main__":
    # tf.disable_v2_behavior()
    # app.run(main)
    init_universal_logging(None)
    fetexp.train_fet()
