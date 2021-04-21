import os
import json
import config
from utils import datautils


def get_uf_sample_str(x):
    lcxt = ' '.join(x['left_context_token']).strip()
    rcxt = ' '.join(x['right_context_token']).strip()
    return '{} [[ {} ]] {}'.format(lcxt, x['mention_span'], rcxt).strip()

def check_retrieved_sents():
    import tensorflow as tf

    output_file = os.path.join(config.DATA_DIR, 'tmp/uf_wia_results_200_ins.txt')
    # results_file = os.path.join(config.DATA_DIR, 'realm_output/uf_wia_results_nm.txt')
    results_file = os.path.join(config.DATA_DIR, 'realm_output/uf_wia_results_200.txt')
    samples_file = os.path.join(config.DATA_DIR, 'ultrafine/uf_data/crowd/test.json')
    wia_block_records_path = os.path.join(config.DATA_DIR, 'ultrafine/zoutput/webisa_full_uffilter.tfr')
    wia_block_labels_file = os.path.join(config.DATA_DIR, 'ultrafine/zoutput/webisa_full_uffilter_labels.txt')

    samples = datautils.read_json_objs(samples_file)

    blocks_dataset = tf.data.TFRecordDataset(
        wia_block_records_path, buffer_size=512 * 1024 * 1024)

    sents = list()
    for i, sent in enumerate(blocks_dataset):
        sents.append(sent.numpy().decode('utf-8'))
        if i % 500000 == 0:
            print(i)
    with open(wia_block_labels_file, encoding='utf-8') as f:
        wia_labels = [line.strip() for line in f]

    fout = open(output_file, 'w', encoding='utf-8')
    f = open(results_file, encoding='utf-8')
    for i, line in enumerate(f):
        x = json.loads(line)
        # print(x)
        bids = x['block_ids']
        # print(samples[i])
        uf_sample_str = get_uf_sample_str(samples[i])
        fout.write('{}\n{}\n'.format(uf_sample_str, samples[i]['y_str']))
        # print(uf_sample_str)
        # print(x['y_str'])
        for bid in bids:
            fout.write('{}\n'.format(sents[bid]))
            fout.write('{}\n'.format(wia_labels[bid]))
            # print(sents[bid])
        fout.write('\n')
        # print()
        # if i > 3:
        #     break
    f.close()
    fout.close()


def check_qz_scores():
    import numpy as np

    qemb_file = os.path.join(config.DATA_DIR, 'realm_output/uf_test_qembs.pkl')
    results_file = os.path.join(config.DATA_DIR, 'realm_output/uf_wia_results_200.txt')
    wia_block_emb_file = os.path.join(config.DATA_DIR, 'ultrafine/zoutput/webisa_full_uffilter_emb.pkl')

    bids_list = list()
    f = open(results_file, encoding='utf-8')
    for i, line in enumerate(f):
        x = json.loads(line)
        # print(x)
        bids = x['block_ids']
        bids_list.append(bids)
    f.close()

    qembs = datautils.load_pickle_data(qemb_file)
    block_embs = datautils.load_pickle_data(wia_block_emb_file)
    print(qembs.shape)
    print(block_embs.shape)
    scores = list()
    qemb = qembs[0]
    for i, block_emb in enumerate(block_embs):
        scores.append(np.sum(qemb * block_emb))
        if i % 100000 == 0:
            print(i)
    bids = bids_list[0]
    scores = np.array(scores)
    print(scores[:100])
    print(bids)
    print([scores[bid] for bid in bids])
    idxs = np.argsort(-scores)[:10]
    print(idxs)
    print(scores[idxs])


# check_retrieved_sents()
check_qz_scores()
