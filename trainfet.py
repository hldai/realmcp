import os
import logging
import tensorflow_hub as hub
import tensorflow as tf
from bert import tokenization
from locbert import optimization
# from orqa.utils import bert_utils
from exp import fetexp
from absl import app
from absl import flags
from utils import utils
import config


if __name__ == "__main__":
    args = utils.parse_idx_device_args()

    output_dir = os.path.join(config.OUTPUT_DIR, 'realm_output')

    block_records_path = os.path.join(config.DATA_DIR, 'realm_data/realm_blocks/blocks_2m.tfr')
    block_labels_file = None
    block_emb_file = os.path.join(config.DATA_DIR, 'realm_data/realm_blocks/block_emb_2m.pkl')
    model_dir = os.path.join(output_dir, 'models')

    et_block_records_path = os.path.join(config.DATA_DIR, 'ultrafine/rlm_fet/enwiki-20151002-type-sents-2m.tfr')
    et_block_labels_file = None
    et_block_emb_file = os.path.join(config.DATA_DIR, 'ultrafine/rlm_fet/enwiki-20151002-type-sents-2m-emb.pkl')
    et_model_dir = os.path.join(output_dir, 'etdmodels')

    wia_block_records_path = os.path.join(config.DATA_DIR, 'ultrafine/zoutput/webisa_full_uffilter.tfr')
    wia_block_labels_file = os.path.join(config.DATA_DIR, 'ultrafine/zoutput/webisa_full_uffilter_labels.txt')
    wia_block_emb_file = os.path.join(config.DATA_DIR, 'ultrafine/zoutput/webisa_full_uffilter_emb.pkl')
    wia_model_dir = os.path.join(output_dir, 'wiamodels')
    lwia_model_dir = os.path.join(output_dir, 'lwiamodels')

    # tf.disable_v2_behavior()
    # app.run(main)
    # init_universal_logging(None)
    # tf.get_logger().setLevel('INFO')
    log_name = 'train_fet_{}'.format(args.idx)
    if args.idx == 0:
        fetexp.train_fet(block_records_path, block_emb_file, block_labels_file, model_dir, 'train', log_name)
    elif args.idx == 1:
        fetexp.train_fet(
            et_block_records_path, et_block_emb_file, et_block_labels_file, et_model_dir, 'train', log_name)
    elif args.idx == 2:
        fetexp.train_fet(
            wia_block_records_path, wia_block_emb_file, None, wia_model_dir, 'train', log_name)
    elif args.idx == 3:
        fetexp.train_fet(
            wia_block_records_path, wia_block_emb_file, wia_block_labels_file, lwia_model_dir, 'train', log_name)
    elif args.idx == 4:
        fetexp.train_fet(block_records_path, block_emb_file, model_dir, 'predict', None)
    elif args.idx == 5:
        fetexp.train_fet(et_block_records_path, et_block_emb_file, et_model_dir, 'predict', None)
    elif args.idx == 6:
        fetexp.train_fet(
            wia_block_records_path, wia_block_emb_file, wia_block_labels_file, lwia_model_dir, 'predict', None)
