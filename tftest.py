import os
from locbert import optimization
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import random
from utils import datautils
import config

global_vals = tf.constant(np.random.uniform(-5, -4, (3, 5)), tf.float32)


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


class DataGen:
    def __init__(self):
        pass

    def __iter__(self):
        return self.gen_val()

    def gen_val(self):
        for _ in range(3):
            yield tf.constant(np.random.uniform(-1, 1, (3, 5))), -1


# def input_fn():
#     print('CALLLLLLLLLLLLLLLLLLLL input fn')
#     def data_gen():
#         for _ in range(3):
#             yield np.random.uniform(-1, 1, (3, 5)), -1
#
#     dataset = tf.data.Dataset.from_generator(
#         data_gen,
#         output_types=(tf.float32, tf.int32))
#     return dataset


def get_padded_bert_input(tok_id_seqs):
    max_seq_len = max(len(seq) for seq in tok_id_seqs)
    tok_id_seq_batch = np.zeros((len(tok_id_seqs), max_seq_len), np.int32)
    input_mask = np.zeros((len(tok_id_seqs), max_seq_len), np.int32)
    for i, seq in enumerate(tok_id_seqs):
        tok_id_seq_batch[i][:len(seq)] = seq
        for j in range(len(seq)):
            input_mask[i][j] = 1
    return tok_id_seq_batch, input_mask


def to_one_hot(inds, vec_len):
    v = np.zeros(vec_len, dtype=np.float32)
    for ind in inds:
        v[ind] = 1
    return v


def input_fn():
    import json
    from locbert import tokenization

    batch_size = 4
    data_file = '/data/hldai/data/ultrafine/uf_data/crowd/test.json'
    type_vocab_file = '/data/hldai/data/ultrafine/uf_data/ontology/types.txt'
    reader_module_path = '/data/hldai/data/realm_data/cc_news_pretrained/bert'
    vocab_file = os.path.join(reader_module_path, 'assets/vocab.txt')
    tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case=True)

    types, type_id_dict = datautils.load_vocab_file(type_vocab_file)
    n_types = len(types)

    # texts = ['He is a teacher.',
    #          'He teaches his students.',
    #          'He is a lawyer.']
    texts = list()

    all_labels = list()
    with open(data_file, encoding='utf-8') as f:
        for i, line in enumerate(f):
            x = json.loads(line)
            text = '{} {} {}'.format(
                ' '.join(x['left_context_token']), x['mention_span'], ' '.join(x['right_context_token']))
            # print(text)
            texts.append(text)
            labels = x['y_str']
            tids = [type_id_dict.get(t, -1) for t in labels]
            tids = [tid for tid in tids if tid > -1]
            # if i > 5:
            all_labels.append(tids)
            if len(texts) >= 8:
                break
    print(len(texts), 'texts')

    def tok_id_seq_gen():
        tok_id_seqs = list()
        y_vecs = list()
        for i, text in enumerate(texts):
            tokens = tokenizer.tokenize(text)
            # print(tokens)
            tokens_full = ['[CLS]'] + tokens + ['[SEP]']
            tok_id_seq = tokenizer.convert_tokens_to_ids(tokens_full)
            # tok_id_seq = np.array([len(text)], np.float32)
            tok_id_seqs.append(tok_id_seq)
            y_vecs.append(to_one_hot(all_labels[i], n_types))
            if len(tok_id_seqs) >= batch_size:
                # tok_id_seq_batch, input_mask = get_padded_bert_input(tok_id_seqs)
                tok_id_seq_batch = tf.ragged.constant(tok_id_seqs)
                # y_vecs_tensor = tf.concat(y_vecs)
                yield {'tok_id_seq_batch': tok_id_seq_batch,
                       # 'input_mask': input_mask,
                       'vals': np.random.uniform(-1, 1, (3, 5))}, y_vecs
                tok_id_seqs = list()
                y_vecs = list()
        if len(tok_id_seqs) > 0:
            # tok_id_seq_batch, input_mask = get_padded_bert_input(tok_id_seqs)
            # y_vecs_tensor = tf.concat(y_vecs)
            tok_id_seq_batch = tf.ragged.constant(tok_id_seqs)
            yield {'tok_id_seq_batch': tok_id_seq_batch,
                   # 'input_mask': input_mask,
                   'vals': np.random.uniform(-1, 1, (3, 5))}, y_vecs

    # for v in iter(tok_id_seq_gen()):
    #     print(v)
    dataset = tf.data.Dataset.from_generator(
        tok_id_seq_gen,
        output_signature=(
            {
                'tok_id_seq_batch': tf.RaggedTensorSpec(dtype=tf.int32, ragged_rank=1),
                # 'tok_id_seq_batch': tf.TensorSpec(shape=None, dtype=tf.int32),
                # 'input_mask': tf.TensorSpec(shape=None, dtype=tf.int32),
                'vals': tf.TensorSpec(shape=None, dtype=tf.float32)
            },
            tf.TensorSpec(shape=None, dtype=tf.float32)))

    return dataset


def model_fn(features, labels, mode, params):
    num_train_steps = 100
    lr = 0.0001
    k = 5
    weights = tf.Variable(tf.random_normal_initializer()(shape=[k, k], dtype=tf.float32))
    cvals = tf.constant(np.random.uniform(-1, 1), tf.float32)
    reader_module_path = '/data/hldai/data/realm_data/cc_news_pretrained/bert'
    # vals_ragged = tf.ragged.constant(list(features))
    # vals_tensor = vals_ragged.to_tensor()
    # vals_tensor = features + global_vals
    vals_tensor = features['vals']
    # tok_id_seq_batch = features['tok_id_seq_batch'].to_tensor()
    # input_mask = features['input_mask']
    z = tf.matmul(vals_tensor, weights) + cvals
    predictions = z
    loss = tf.reduce_mean(z)
    eval_metric_ops = None

    # rand_vals = tf.constant(np.random.randint(100, 105, (4, 5)), dtype=tf.int32)
    # tok_id_seq_batch = tf.concat((features['tok_id_seq_batch'], rand_vals), axis=1).to_tensor()

    # with tf.device("/cpu:0"):
    #     blocks_np = datautils.load_pickle_data(os.path.join(config.DATA_DIR, 'realm_data/blocks_tok_id_seqs.pkl'))
    #     blocks = tf.ragged.constant(blocks_np)
    #     # blocks = tf.ragged.constant([[3, 4], [1], [2, 3, 7], [4]], dtype=tf.int32, ragged_rank=1)
    #     retrieved_block_ids = tf.constant([0, 2])
    #     retrieved_blocks = tf.gather(blocks, retrieved_block_ids).to_tensor()
    # logging_hook = tf.estimator.LoggingTensorHook({"pred": predictions, 'feat': features}, every_n_iter=1)
    # logging_hook = tf.estimator.LoggingTensorHook(
    #     {"pred": predictions, 'labels': labels, 'feat': features['tok_id_seq_batch'],
    #      'ids': retrieved_block_ids}, every_n_iter=1)
    ls = tf.reduce_sum(labels, axis=1)
    logging_hook = tf.estimator.LoggingTensorHook({
        'z': z, 'ls': ls}, every_n_iter=1)

    train_op = optimization.create_optimizer(
        loss=loss,
        init_lr=lr,
        num_train_steps=num_train_steps,
        num_warmup_steps=min(10000, max(100, int(num_train_steps / 10))),
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

    logger = tf.get_logger()
    logger.setLevel('INFO')
    logger.propagate = False

    run_config = tf.estimator.RunConfig(
        model_dir=model_dir,
        log_step_count_steps=5,
        save_checkpoints_steps=5,
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
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


train()
exit()

# digits = tf.ragged.constant([[3, 1, 4, 1], [], [5, 9, 2], [6], []])
# vals = tf.ragged.constant([[2, 3], [1], [], [1, 4, 9], [0]])

# print(tf.concat((digits, vals), axis=1))

# blocks_list = list()
# lens = [3, 1, 5, 2, 5, 3, 4, 2]
# # print(all_vals)
# # all_vals = get_ragged_vals(5)
# for length in lens:
#     vals = list()
#     for _ in range(length):
#         vals.append(random.uniform(-1, 1))
#     blocks_list.append(vals)
blocks_list = [[3, 4], [5, 3, 8], [8]]


def data_gen():
    for vals in blocks_list:
        yield tf.ragged.constant([vals], dtype=tf.float32)

dataset = tf.data.Dataset.from_generator(
    data_gen,
    output_signature=tf.RaggedTensorSpec(ragged_rank=1, dtype=tf.float32))

for v in dataset:
    print(v)
# dataset = dataset.apply(
#     tf.data.experimental.dense_to_ragged_batch(2))
# for v in dataset.batch(4):
#     print(v)
#     # print(v[1].to_tensor())
#     vt = v[1].to_tensor()
#     print(vt)
#     print(tf.squeeze(vt))
# for v in dataset:
#     print(v)

# tf.data.experimental.save(dataset, '/data/hldai/data/tmp/tmp.tfdata', shard_func=lambda x: np.int64(0))
# print('******************* save ************************')

# dataset = tf.data.experimental.load(
#     '/data/hldai/data/tmp/tmp.tfdata',
#     element_spec=tf.RaggedTensorSpec(ragged_rank=1, dtype=tf.float32))
# dataset = dataset.batch(8)
# all_docs = tf.data.experimental.get_single_element(dataset)
# print(all_docs)
# for v in dataset:
#     print(v)

# non_ragged_dataset = tf.data.Dataset.from_tensor_slices([1, 5, 3, 2, 8])
# non_ragged_dataset = non_ragged_dataset.map(tf.range)
# for v in non_ragged_dataset:
#     print(v)
# batched_non_ragged_dataset = non_ragged_dataset.apply(
#     tf.data.experimental.dense_to_ragged_batch(2))
# for element in batched_non_ragged_dataset:
#     print(element)

# vals_tensor = vals_ragged.to_tensor()
# weights = tf.Variable(tf.random_normal_initializer()(shape=[k, k], dtype=tf.float32))
# z = tf.matmul(vals_tensor, weights)
# print(weights)
# print(vals)
# print(vals_ragged)
# print(vals_tensor)
# print(z)
