from locbert import optimization
import tensorflow as tf
import random


def get_ragged_vals(max_n_vals):
    all_vals = [[random.uniform(-1, 1) for _ in range(max_n_vals)]]
    for i in range(7):
        vals = list()
        rand_l = random.randint(1, max_n_vals)
        # print(rand_l)
        for _ in range(rand_l):
            vals.append(random.uniform(-1, 1))
        all_vals.append(vals)
    return all_vals


def input_fn():
    def data_gen():
        # print(all_vals)
        all_vals = get_ragged_vals(5)
        yield all_vals, -1

    dataset = tf.data.Dataset.from_generator(
        data_gen,
        output_types=(tf.float32, tf.int32))

    return dataset


def model_fn(features, labels, mode, params):
    num_train_steps = 100
    lr = 0.0001
    k = 5
    weights = tf.Variable(tf.random_normal_initializer()(shape=[k, k], dtype=tf.float32))
    vals_ragged = tf.ragged.constant(list(features))
    vals_tensor = vals_ragged.to_tensor()
    z = tf.matmul(vals_tensor, weights)
    predictions = z
    loss = tf.reduce_mean(z)
    eval_metric_ops = None

    # logging_hook = tf.estimator.LoggingTensorHook({"pred": predictions, 'feat': features}, every_n_iter=1)
    # logging_hook = tf.estimator.LoggingTensorHook(
    #     {"pred": predictions, 'labels': labels, 'feat': features['tok_id_seq_batch'],
    #      'ids': retrieved_block_ids}, every_n_iter=1)
    logging_hook = tf.estimator.LoggingTensorHook({'z': z, 'feat': features}, every_n_iter=1)

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


def train():
    params = dict()
    model_dir = '/data/hldai/data/tmp/tmpmodels'
    num_train_steps = 10

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

    estimator.evaluate(input_fn)


# train()
k = 5
vals = get_ragged_vals(k)
vals_ragged = tf.ragged.constant(vals)
print(vals_ragged)
print(tf.gather(vals_ragged, [2, 3]))
# vals_tensor = vals_ragged.to_tensor()
# weights = tf.Variable(tf.random_normal_initializer()(shape=[k, k], dtype=tf.float32))
# z = tf.matmul(vals_tensor, weights)
# print(weights)
# print(vals)
# print(vals_ragged)
# print(vals_tensor)
# print(z)
