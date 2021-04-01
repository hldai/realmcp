import os
import gzip
import inflect
import config
from utils import datautils


def check_sents():
    type_vocab_file = os.path.join(config.DATA_DIR, 'ultrafine/uf_data/ontology/types.txt')
    type_sents_file = os.path.join(config.DATA_DIR, 'ultrafine/enwiki-20151002-type-sents.txt.gz')
    type_vocab, type_id_dict = datautils.load_vocab_file(type_vocab_file)
    # type_set = set(type_vocab)

    inflect_engine = inflect.engine()
    type_set = set()
    for i, t in enumerate(type_vocab):
        t = t.replace('_', ' ')
        type_set.add(t)
        plural_t = inflect_engine.plural(t)
        type_set.add(plural_t)

    f = gzip.open(type_sents_file, 'rt', encoding='utf-8')
    for i, line in enumerate(f):
        # print(line.strip())
        sent = line.strip()

        words = sent.split(' ')
        n_words = len(words)
        keep = False
        for j in range(n_words):
            cur_word = words[j].lower()
            # if cur_word in type_set:
            if cur_word in type_set:
                print(cur_word, '&', sent)

        if i > 3:
            break
    f.close()
