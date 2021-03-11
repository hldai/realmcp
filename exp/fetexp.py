import os
import json
import collections
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
import tensorflow_probability as tfp
import scann
from locbert import tokenization
# from bert import tokenization
from locbert import tokenization, optimization
from orqa.utils import scann_utils
import config
from utils import datautils, bert_utils


RetrieverOutputs = collections.namedtuple("RetrieverOutputs", ["logits", "blocks"])
data_dir = config.DATA_DIR

num_block_records = 13353718
n_block_rec_parts = [2670743, 5341486, 8012229, 10682972, 13353718]
retriever_module_path = os.path.join(data_dir, 'realm_data/cc_news_pretrained/embedder')
var_name = "block_emb"
checkpoint_path = os.path.join(retriever_module_path, "encoded", "encoded.ckpt")
np_db = tf.train.load_checkpoint(checkpoint_path).get_tensor(var_name)
# blocks_list = list()
#
# def load_block_records():
#     print('LOADING blocks')
#     rand_lens = np.random.randint(5, 20, num_block_records)
#     for i, rand_len in enumerate(rand_lens):
#         vals = np.random.randint(0, 5000, rand_len)
#         blocks_list.append(vals)
#         if i % 1000000 == 0:
#             print(i)


def load_rt_dataset_single_elem(dataset_path):
    dataset = tf.data.experimental.load(
        dataset_path,
        element_spec=tf.RaggedTensorSpec(ragged_rank=1, dtype=tf.int32))
    print('dataset loaded from {}'.format(dataset_path))
    # for i, v in enumerate(dataset):
    #     print(v)
    #     if i > 3:
    #         break
    # dataset = dataset.batch(n_records)
    # print('batched')
    # t = time.time()
    all_docs = tf.data.experimental.get_single_element(dataset)

    # print(type(all_docs))
    # print(all_docs[:3])
    # print(time.time() - t)
    return all_docs


def load_dataset_parts(dataset_path_prefix, n_parts):
    vals_list = list()
    for i in range(n_parts):
        dataset_path = '{}{}.tfd'.format(dataset_path_prefix, i)
        vals = load_rt_dataset_single_elem(dataset_path)
        vals_list.append(vals)
    if n_parts == 1:
        return vals_list[0]
    return tf.concat(vals_list, axis=0)


def load_blocks_from_pkl():
    blocks_list = datautils.load_pickle_data(os.path.join(config.DATA_DIR, 'realm_data/blocks_tok_id_seqs.pkl'))
    blocks_list = blocks_list[:10000000]
    print('blocks list loaded', len(blocks_list))
    blocks = tf.ragged.constant(blocks_list, dtype=tf.int32)
    print('blocks list to ragged')
    return blocks


def load_blocks_from_ragged_list():
    dataset_path = os.path.join(config.DATA_DIR, 'tmp/blocks_tok_id_seqs_l128_all.tfdata')
    dataset = tf.data.experimental.load(
        dataset_path,
        element_spec=tf.RaggedTensorSpec(ragged_rank=1, dtype=tf.int32))
    print('dataset loaded')
    dataset = dataset.batch(num_block_records)
    print('batched')
    blocks = tf.data.experimental.get_single_element(dataset)
    return blocks


def retrieve(query_token_id_seqs, input_mask, embedder_path, mode, retriever_beam_size):
    """Do retrieval."""
    print('RRRRRRRRRRRRRRRRRetrieve', mode)
    block_records_path = os.path.join(data_dir, 'realm_data/blocks.tfr')
    # retriever_module_path = os.path.join(data_dir, 'realm_data/cc_news_pretrained/embedder')
    # var_name = "block_emb"
    # checkpoint_path = os.path.join(retriever_module_path, "encoded", "encoded.ckpt")

    retriever_module = hub.Module(
        embedder_path,
        tags={"train"} if mode == tf.estimator.ModeKeys.TRAIN else {},
        trainable=True)

    # [1, projection_size]
    question_emb = retriever_module(
        inputs=dict(
            input_ids=query_token_id_seqs,
            # input_mask=tf.ones_like(query_token_id_seqs),
            input_mask=input_mask,
            segment_ids=tf.zeros_like(query_token_id_seqs)),
        signature="projected")
    # return question_emb
    # question_emb_np = np.random.uniform(-1, 1, (4, 128))
    # question_emb = tf.constant(question_emb_np, tf.float32)

    # block_emb, searcher = scann_utils.load_scann_searcher(
    #     var_name="block_emb",
    #     checkpoint_path=os.path.join(embedder_path, "encoded", "encoded.ckpt"),
    #     num_neighbors=retriever_beam_size)
    # np_db = tf.train.load_checkpoint(checkpoint_path).get_tensor(var_name)
    n_docs = np_db.shape[0]
    # np_db = np_db[:n_docs // 2]

    # block_emb = tf.constant(np_db)
    block_emb = tf.compat.v1.get_variable('block_emb_tf', shape=np_db.shape)

    def init_fn(scaffold, sess):
        print('FFFFFFFFFFFFFFFFFFFFFF INIT BLOCK EMB')
        sess.run(block_emb.initializer, {block_emb.initial_value: np_db})

    scaffold = tf.compat.v1.train.Scaffold(init_fn=init_fn)

    searcher = scann.scann_ops.builder(block_emb, retriever_beam_size, "dot_product").tree(
        num_leaves=1000, num_leaves_to_search=100, training_sample_size=250000).score_ah(
        2, anisotropic_quantization_threshold=0.2).reorder(100).build()

    # [1, retriever_beam_size]
    retrieved_block_ids, _ = searcher.search_batched(question_emb)

    # [1, retriever_beam_size, projection_size]
    retrieved_block_emb = tf.gather(block_emb, retrieved_block_ids)

    # [retriever_beam_size]
    # retrieved_block_ids = tf.squeeze(retrieved_block_ids)
    # retrieved_block_ids = tf.constant(np.random.randint(0, 10000, (4, 5)))
    retrieved_block_ids = tf.reshape(retrieved_block_ids, shape=(-1, retriever_beam_size))
    # retrieved_block_ids = tf.reshape(retrieved_block_ids, shape=(-1))

    # # [retriever_beam_size, projection_size]
    # retrieved_block_emb = tf.squeeze(retrieved_block_emb)
    #
    # [1, retriever_beam_size]
    question_emb_ex = tf.expand_dims(question_emb, axis=1)
    retrieved_logits = tf.matmul(question_emb_ex, retrieved_block_emb, transpose_b=True)

    # # [retriever_beam_size]
    # retrieved_logits = tf.squeeze(retrieved_logits, 0)

    blocks_dataset = tf.data.TFRecordDataset(
        block_records_path, buffer_size=512 * 1024 * 1024)
    blocks_dataset = blocks_dataset.batch(
        num_block_records, drop_remainder=True)
    blocks = tf.compat.v1.get_local_variable(
        "blocks",
        initializer=tf.data.experimental.get_single_element(blocks_dataset))
    retrieved_blocks = tf.gather(blocks, retrieved_block_ids)
    # # blocks = tf.get_variable(
    # #     "blocks",
    # #     initializer=tf.data.experimental.get_single_element(blocks_dataset))
    # # blocks = tf.constant(tf.data.experimental.get_single_element(blocks_dataset))

    # blocks_list = list()
    # rand_lens = np.random.randint(5, 20, 100)
    # for i, rand_len in enumerate(rand_lens):
    #     vals = np.random.randint(0, 5000, rand_len)
    #     blocks_list.append(vals)
    #     if i % 1000000 == 0:
    #         print(i)
    # blocks = tf.ragged.constant(blocks_list, dtype=tf.int32)

    # dataset_path = '/data/hldai/data/tmp/tmp.tfdata'
    # dataset = tf.data.experimental.load(
    #     dataset_path,
    #     element_spec=tf.RaggedTensorSpec(ragged_rank=1, dtype=tf.int32))
    # print('dataset loaded')
    # print('batched')
    # blocks = tf.data.experimental.get_single_element(dataset)
    # blocks = load_dataset_parts(
    #     os.path.join(data_dir, 'realm_data/blocks_tok_id_seqs_l128/blocks_tok_id_seqs_l128_p'), 1)
    # retrieved_blocks = tf.gather(blocks, retrieved_block_ids).to_tensor()

    # blocks = load_rt_dataset_single_elem(os.path.join(config.DATA_DIR, 'realm_data/blocks_tok_id_seqs_l128_4k.tfd'))
    # blocks = load_blocks_from_ragged_list()
    # blocks = load_blocks_from_pkl()
    # retrieved_blocks = tf.gather(blocks, retrieved_block_ids).to_tensor()
    # retrieved_blocks = tf.squeeze(retrieved_blocks)

    print('blocks obtained')

    return scaffold, question_emb, retrieved_block_emb, retrieved_block_ids, retrieved_blocks, retrieved_logits
    # return scaffold, retrieved_block_emb, retrieved_block_ids


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


def model_fn(features, labels, mode, params):
    # print('MMMMMMMMMMMMMMMMMModel_fn', mode)
    embedder_module_path = os.path.join(config.DATA_DIR, 'realm_data/cc_news_pretrained/embedder')
    reader_module_path = os.path.join(config.DATA_DIR, 'realm_data/cc_news_pretrained/bert')
    retriever_beam_size = 5
    lr = 1e-5
    num_train_steps = 10
    max_seq_len = 256
    bert_dim = 768
    n_types = params['n_types']
    sep_tok_id = params['sep_tok_id']

    # token_ids = tf.constant([[101, 2002, 2003, 1037, 3836, 1012, 102]], dtype=tf.int32)
    tok_id_seq_batch_tensor = features['tok_id_seq_batch'].to_tensor()
    input_mask = 1 - tf.cast(tf.equal(tok_id_seq_batch_tensor, tf.constant(0)), tf.int32)
    # input_mask = features['input_mask']
    with tf.device("/cpu:0"):
        # retriever_outputs = retrieve(tok_id_seq_batch, input_mask, embedder_module_path, mode, retriever_beam_size)
        scaffold, question_emb, retrieved_block_emb, retrieved_block_ids, retrieved_blocks, zx_logits = retrieve(
            tok_id_seq_batch_tensor, input_mask, embedder_module_path, mode, retriever_beam_size)
        # scaffold, question_emb, retrieved_block_ids = retrieve(
        #     tok_id_seq_batch, input_mask, embedder_module_path, mode, retriever_beam_size)

    tokenizer, vocab_lookup_table = bert_utils.get_tf_tokenizer(reader_module_path)
    block_tok_id_seqs = tokenizer.tokenize(retrieved_blocks)
    block_tok_id_seqs = tf.cast(
        block_tok_id_seqs.merge_dims(2, 3).to_tensor(), tf.int32)
    # batch_size = tf.shape(tok_id_seq_batch_tensor)[0]
    blocks_max_seq_len = tf.shape(block_tok_id_seqs)[-1]
    block_tok_id_seqs_flat = tf.reshape(block_tok_id_seqs, (-1, blocks_max_seq_len))

    tok_id_seqs_repeat = features['tok_id_seqs_repeat']
    # tok_id_seqs_repeat = features['tok_id_seqs_repeat'].to_tensor()
    q_doc_tok_id_seqs = tf.concat((tok_id_seqs_repeat, block_tok_id_seqs_flat), axis=1).to_tensor()
    q_doc_tok_id_seqs = q_doc_tok_id_seqs[:, :max_seq_len - 1]
    q_doc_tok_id_seqs = pad_sep_to_tensor(q_doc_tok_id_seqs, sep_tok_id)

    # reach_max_len = tf.equal(q_doc_tok_id_seqs[:, -1], tf.constant(0, tf.int32))
    # reach_max_len = 1 - tf.cast(reach_max_len, tf.int32)
    # reach_max_len = tf.reshape(reach_max_len, (-1, 1))
    # seps_tensor = reach_max_len * sep_tok_id
    # # print(seps_tensor)
    # # print(tf.concat((q_doc_tok_id_seqs, seps_tensor), axis=1))
    # q_doc_tok_id_seqs = tf.concat((q_doc_tok_id_seqs, seps_tensor), axis=1)
    q_doc_input_mask = 1 - tf.cast(tf.equal(q_doc_tok_id_seqs, tf.constant(0, dtype=tf.int32)), tf.int32)

    reader_module = hub.Module(
        reader_module_path,
        tags={"train"} if mode == tf.estimator.ModeKeys.TRAIN else {},
        trainable=True)

    # input_mask = tf.sequence_mask(lengths, seqs_shape[1])
    # input_mask = tf.cast(input_mask, tf.int32)
    concat_outputs = reader_module(
        dict(
            # input_ids=tok_id_seq_batch,
            # input_mask=tf.ones_like(tok_id_seq_batch),
            # segment_ids=tf.zeros_like(tok_id_seq_batch)
            # segment_ids=concat_inputs.segment_ids
            input_ids=q_doc_tok_id_seqs,
            # input_mask=tf.ones_like(r_tok_id_seqs),
            input_mask=q_doc_input_mask,
            segment_ids=tf.zeros_like(q_doc_tok_id_seqs)
        ),
        signature="tokens",
        as_dict=True)
    # # predictions = retriever_outputs.logits
    #
    # concat_token_emb = concat_outputs["sequence_output"]
    qd_reps = concat_outputs['sequence_output'][:, 0, :]

    dense_layer = tf.keras.layers.Dense(n_types)
    yzx_logits = dense_layer(qd_reps)
    yzx_logits = tf.reshape(yzx_logits, (-1, retriever_beam_size, n_types))
    zx_logits = tf.reshape(zx_logits, (-1, retriever_beam_size))

    log_softmax_zx_logits = tf.nn.log_softmax(zx_logits, axis=1)
    log_sig_yzx_logits = tf.math.log_sigmoid(yzx_logits)
    z_log_probs = log_sig_yzx_logits + tf.expand_dims(log_softmax_zx_logits, 2)
    log_probs = tf.reduce_logsumexp(z_log_probs, axis=1)
    log_neg_probs = tfp.math.log1mexp(log_probs)
    # prob_sum = tf.math.exp(log_probs) + tf.math.exp(log_neg_probs)

    # kernel_initializer = tf.truncated_normal_initializer(stddev=0.02)
    # projection = tf.layers.dense(
    #     qd_reps,
    #     bert_dim,
    #     kernel_initializer=kernel_initializer)
    # yzx_logits =

    predictions = tf.exp(log_probs)
    # loss = tf.reduce_mean(predictions)
    loss_samples = -tf.reduce_sum(labels * log_probs + (1 - labels) * log_neg_probs, axis=1)
    loss = tf.reduce_mean(loss_samples)

    small_constant = tf.constant(0.00001)
    pos_preds = tf.cast(tf.less(tf.constant(0.5), predictions), tf.float32)
    n_pred_pos = tf.reduce_sum(pos_preds, axis=1) + small_constant
    n_true_pos = tf.reduce_sum(labels, axis=1) + small_constant
    n_corrects = tf.reduce_sum(pos_preds * labels, axis=1)
    precision = tf.reduce_mean(n_corrects / n_pred_pos)
    recall = tf.reduce_mean(n_corrects / n_true_pos)
    # logging_hook = tf.estimator.LoggingTensorHook({"pred": predictions, 'feat': features}, every_n_iter=1)
    # logging_hook = tf.estimator.LoggingTensorHook(
    #     {"pred": predictions, 'labels': labels, 'feat': features['tok_id_seq_batch'],
    #      'ids': retrieved_block_ids}, every_n_iter=1)
    train_logging_hook = tf.estimator.LoggingTensorHook({
        'loss': loss,
    }, every_n_iter=1)
    logging_hook = tf.estimator.LoggingTensorHook({
        'loss': loss,
        'pred': tf.reduce_mean(predictions)
    #     'pred': predictions,
    #     'labels': labels,
    #     'log_probs': tf.shape(log_probs),
    #     'yzx_logits': tf.shape(yzx_logits),
    #     'co': tf.shape(qd_reps),
    #     # 'prob_sum': prob_sum,
    #     '1b': tf.shape(question_emb),
    #     # 'qseq': q_doc_tok_id_seqs,
    #     'bk': tf.shape(zx_logits),
    #     'bemb': tf.shape(block_tok_id_seqs_flat),
    #     'bs': tf.shape(q_doc_tok_id_seqs),
    }, every_n_iter=1)
    # logging_hook = tf.estimator.LoggingTensorHook(
    #     {'ids': retrieved_block_ids, 'pred': predictions}, every_n_iter=1)
    # pred_mean = tf.keras.metrics.Mean(name='mean_1')(tf.reduce_mean(predictions))
    # pred_mean = tf.reduce_mean(predictions)
    # eval_metric_ops = None
    eval_metric_ops = {
        'precision': tf.compat.v1.metrics.mean(precision),
        'recall': tf.compat.v1.metrics.mean(recall)
    }

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
        # training_hooks=[train_logging_hook],
        # evaluation_hooks=[logging_hook],
        eval_metric_ops=eval_metric_ops,
        scaffold=scaffold)


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


class InputData:
    def __init__(self, tokenizer, types, type_id_dict, retriever_beam_size):
        self.batch_size = 2
        self.train_data_file = os.path.join(config.DATA_DIR, 'ultrafine/uf_data/crowd/train.json')
        self.dev_data_file = os.path.join(config.DATA_DIR, 'ultrafine/uf_data/crowd/dev.json')
        self.test_data_file = os.path.join(config.DATA_DIR, 'ultrafine/uf_data/crowd/test.json')
        self.tokenizer = tokenizer
        self.types = types
        self.type_id_dict = type_id_dict
        self.n_types = len(self.types)
        self.retriever_beam_size = retriever_beam_size

    def input_fn_train(self):
        return self.input_fn(self.train_data_file)

    def input_fn_dev(self):
        return self.input_fn(self.dev_data_file)

    def input_fn_test(self):
        return self.input_fn(self.test_data_file)

    def input_fn(self, data_file):
        batch_size = self.batch_size
        texts = list()
        all_labels = list()
        with open(data_file, encoding='utf-8') as f:
            for i, line in enumerate(f):
                x = json.loads(line)
                mstr = x['mention_span']
                text = '{} [SEP] {} {} {}'.format(
                    mstr, ' '.join(x['left_context_token']), mstr, ' '.join(x['right_context_token']))
                # print(text)
                texts.append(text)
                labels = x['y_str']
                tids = [self.type_id_dict.get(t, -1) for t in labels]
                tids = [tid for tid in tids if tid > -1]
                # if i > 5:
                all_labels.append(tids)
                if len(texts) >= 12:
                    break
        print(len(texts), 'texts')

        def tok_id_seq_gen():
            tok_id_seqs, tok_id_seqs_repeat = list(), list()
            y_vecs = list()
            for i, text in enumerate(texts):
                tokens = self.tokenizer.tokenize(text)
                # print(tokens)
                tokens_full = ['[CLS]'] + tokens + ['[SEP]']
                tok_id_seq = self.tokenizer.convert_tokens_to_ids(tokens_full)
                # tok_id_seq = np.array([len(text)], np.float32)
                tok_id_seqs.append(tok_id_seq)
                for _ in range(self.retriever_beam_size):
                    tok_id_seqs_repeat.append(tok_id_seq)
                y_vecs.append(to_one_hot(all_labels[i], self.n_types))

                if len(tok_id_seqs) >= batch_size:
                    # tok_id_seq_batch, input_mask = get_padded_bert_input(tok_id_seqs)
                    tok_id_seq_batch = tf.ragged.constant(tok_id_seqs)
                    tok_id_seqs_repeat_ragged = tf.ragged.constant(tok_id_seqs_repeat)
                    # y_vecs_tensor = tf.concat(y_vecs)
                    # yield {'tok_id_seq_batch': tok_id_seq_batch, 'input_mask': input_mask}, y_vecs
                    yield {'tok_id_seq_batch': tok_id_seq_batch, 'tok_id_seqs_repeat': tok_id_seqs_repeat_ragged}, y_vecs
                    tok_id_seqs, tok_id_seqs_repeat, y_vecs = list(), list(), list()
                    # y_vecs = list()
            if len(tok_id_seqs) > 0:
                # tok_id_seq_batch, input_mask = get_padded_bert_input(tok_id_seqs)
                tok_id_seq_batch = tf.ragged.constant(tok_id_seqs)
                tok_id_seqs_repeat_ragged = tf.ragged.constant(tok_id_seqs_repeat)
                # y_vecs_tensor = tf.concat(y_vecs)
                # yield {'tok_id_seq_batch': tok_id_seq_batch, 'input_mask': input_mask}, y_vecs
                yield {'tok_id_seq_batch': tok_id_seq_batch, 'tok_id_seqs_repeat': tok_id_seqs_repeat_ragged}, y_vecs

        # for v in iter(tok_id_seq_gen()):
        #     print(v)
        dataset = tf.data.Dataset.from_generator(
            tok_id_seq_gen,
            output_signature=({'tok_id_seq_batch': tf.RaggedTensorSpec(dtype=tf.int32, ragged_rank=1),
                              'tok_id_seqs_repeat': tf.RaggedTensorSpec(dtype=tf.int32, ragged_rank=1)},
                              tf.TensorSpec(shape=None, dtype=tf.float32)))

        return dataset


# def input_fn(self, data_file):
#     retriever_beam_size = 5
#     batch_size = self.batch_size
#     vocab_file = os.path.join(reader_module_path, 'assets/vocab.txt')
#     tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case=True)
#
#     types, type_id_dict = datautils.load_vocab_file(type_vocab_file)
#     n_types = len(types)
#
#     # texts = ['He is a teacher.',
#     #          'He teaches his students.',
#     #          'He is a lawyer.']
#     texts = list()
#
#     all_labels = list()
#     with open(data_file, encoding='utf-8') as f:
#         for i, line in enumerate(f):
#             x = json.loads(line)
#             text = '{} {} {}'.format(
#                 ' '.join(x['left_context_token']), x['mention_span'], ' '.join(x['right_context_token']))
#             # print(text)
#             texts.append(text)
#             labels = x['y_str']
#             tids = [type_id_dict.get(t, -1) for t in labels]
#             tids = [tid for tid in tids if tid > -1]
#             # if i > 5:
#             all_labels.append(tids)
#             if len(texts) >= 8:
#                 break
#     print(len(texts), 'texts')
#
#     def tok_id_seq_gen():
#         tok_id_seqs, tok_id_seqs_repeat = list(), list()
#         y_vecs = list()
#         for i, text in enumerate(texts):
#             tokens = tokenizer.tokenize(text)
#             # print(tokens)
#             tokens_full = ['[CLS]'] + tokens + ['[SEP]']
#             tok_id_seq = tokenizer.convert_tokens_to_ids(tokens_full)
#             # tok_id_seq = np.array([len(text)], np.float32)
#             tok_id_seqs.append(tok_id_seq)
#             for _ in range(retriever_beam_size):
#                 tok_id_seqs_repeat.append(tok_id_seq)
#             y_vecs.append(to_one_hot(all_labels[i], n_types))
#
#             if len(tok_id_seqs) >= batch_size:
#                 # tok_id_seq_batch, input_mask = get_padded_bert_input(tok_id_seqs)
#                 tok_id_seq_batch = tf.ragged.constant(tok_id_seqs)
#                 tok_id_seqs_repeat_ragged = tf.ragged.constant(tok_id_seqs_repeat)
#                 # y_vecs_tensor = tf.concat(y_vecs)
#                 # yield {'tok_id_seq_batch': tok_id_seq_batch, 'input_mask': input_mask}, y_vecs
#                 yield {'tok_id_seq_batch': tok_id_seq_batch, 'tok_id_seqs_repeat': tok_id_seqs_repeat_ragged}, y_vecs
#                 tok_id_seqs, tok_id_seqs_repeat, y_vecs = list(), list(), list()
#                 # y_vecs = list()
#         if len(tok_id_seqs) > 0:
#             # tok_id_seq_batch, input_mask = get_padded_bert_input(tok_id_seqs)
#             tok_id_seq_batch = tf.ragged.constant(tok_id_seqs)
#             tok_id_seqs_repeat_ragged = tf.ragged.constant(tok_id_seqs_repeat)
#             # y_vecs_tensor = tf.concat(y_vecs)
#             # yield {'tok_id_seq_batch': tok_id_seq_batch, 'input_mask': input_mask}, y_vecs
#             yield {'tok_id_seq_batch': tok_id_seq_batch, 'tok_id_seqs_repeat': tok_id_seqs_repeat_ragged}, y_vecs
#
#     # for v in iter(tok_id_seq_gen()):
#     #     print(v)
#     dataset = tf.data.Dataset.from_generator(
#         tok_id_seq_gen,
#         output_signature=({'tok_id_seq_batch': tf.RaggedTensorSpec(dtype=tf.int32, ragged_rank=1),
#                           'tok_id_seqs_repeat': tf.RaggedTensorSpec(dtype=tf.int32, ragged_rank=1)},
#                           tf.TensorSpec(shape=None, dtype=tf.float32)))
#
#     return dataset


def train_fet():
    logger = tf.get_logger()
    # logger.setLevel('ERROR')
    logger.setLevel('INFO')
    logger.propagate = False

    # run_train()

    retriever_beam_size = 5
    num_train_steps = 1000
    embedder_module_path = os.path.join(data_dir, 'realm_data/cc_news_pretrained/embedder')
    reader_module_path = os.path.join(data_dir, 'realm_data/cc_news_pretrained/bert')
    vocab_file = os.path.join(reader_module_path, 'assets/vocab.txt')
    model_dir = os.path.join(data_dir, 'tmp/tmpmodels')
    type_vocab_file = os.path.join(config.DATA_DIR, 'ultrafine/uf_data/ontology/types.txt')

    tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case=True)
    sep_tok_id = tokenizer.convert_tokens_to_ids(['[SEP]'])[0]
    types, type_id_dict = datautils.load_vocab_file(type_vocab_file)
    input_data = InputData(tokenizer, types, type_id_dict, retriever_beam_size)
    num_eval_steps = 10
    n_types = len(types)
    print('sep token id', sep_tok_id)
    # exit()
    params = {'batch_size': 1, 'retriever_beam_size': retriever_beam_size, 'n_types': n_types,
              'sep_tok_id': sep_tok_id}

    run_config = tf.estimator.RunConfig(
        model_dir=model_dir,
        log_step_count_steps=5,
        save_checkpoints_steps=10,
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
        input_fn=input_data.input_fn_train,
        max_steps=num_train_steps)
    eval_spec = tf.estimator.EvalSpec(
        name="default",
        input_fn=input_data.input_fn_test,
        # exporters=exporters,
        # start_delay_secs=FLAGS.eval_start_delay_secs,
        # throttle_secs=FLAGS.eval_throttle_secs
        )

    logger.info('DDDDDDDDDDDDDDDDDDDDD START')
    estimator.evaluate(input_data.input_fn_test)
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
