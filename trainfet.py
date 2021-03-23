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


def main(_):
    # init_universal_logging(None)
    fetexp.train_fet()


if __name__ == "__main__":
    args = utils.parse_idx_device_args()

    output_dir = os.path.join(config.OUTPUT_DIR, 'realm_output')

    block_records_path = os.path.join(config.DATA_DIR, 'realm_data/realm_blocks/blocks_2m.tfr')
    block_emb_file = os.path.join(config.DATA_DIR, 'realm_data/realm_blocks/block_emb_2m.pkl')
    model_dir = os.path.join(output_dir, 'models')

    et_block_records_path = os.path.join(config.DATA_DIR, 'ultrafine/rlm_fet/enwiki-20151002-type-sents-2m.tfr')
    et_block_emb_file = os.path.join(config.DATA_DIR, 'ultrafine/rlm_fet/enwiki-20151002-type-sents-2m-emb.pkl')
    et_model_dir = os.path.join(output_dir, 'etdmodels')
    # tf.disable_v2_behavior()
    # app.run(main)
    # init_universal_logging(None)
    # tf.get_logger().setLevel('INFO')
    if args.idx == 0:
        fetexp.train_fet(block_records_path, block_emb_file, model_dir, 'train', 'train_fet_0')
    elif args.idx == 1:
        fetexp.train_fet(et_block_records_path, et_block_emb_file, model_dir, 'train', 'train_fet_1')
    elif args.idx == 2:
        fetexp.train_fet(block_records_path, block_emb_file, model_dir, 'predict', None)
