import os
import gzip
import inflect
import config
from utils import datautils


def ends_with(words, p, pattern_words):
    pattern_len = len(pattern_words)
    if p < pattern_len - 1:
        return False

    for i in range(pattern_len):
        p_absolute = p - pattern_len + 1 + i
        if words[p_absolute].lower() != pattern_words[i]:
            return False
    return True


def starts_with(words, p, pattern_words):
    pattern_len = len(pattern_words)
    if len(words) - p < pattern_len:
        return False

    for i in range(pattern_len):
        p_absolute = p + i
        if words[p_absolute].lower() != pattern_words[i]:
            return False
    return True


def get_type_str_dict(type_vocab):
    inflect_engine = inflect.engine()
    type_dict = dict()
    for i, t in enumerate(type_vocab):
        t_str = t.replace('_', ' ')
        type_dict[t_str] = t

    for i, t in enumerate(type_vocab):
        t_str = t.replace('_', ' ')
        plural_t = inflect_engine.plural(t_str)
        if plural_t not in type_dict:
            type_dict[plural_t] = t
    return type_dict


def check_sents():
    type_vocab_file = os.path.join(config.DATA_DIR, 'ultrafine/uf_data/ontology/types.txt')
    type_sents_file = os.path.join(config.DATA_DIR, 'ultrafine/res/enwiki-20151002-type-sents.txt.gz')
    output_file = os.path.join(config.DATA_DIR, 'ultrafine/res/enwiki-20151002-type-sents-filter.txt')
    type_vocab, type_id_dict = datautils.load_vocab_file(type_vocab_file)

    type_set = get_type_str_dict(type_vocab)

    f = gzip.open(type_sents_file, 'rt', encoding='utf-8')
    for i, line in enumerate(f):
        # print(line.strip())
        sent = line.strip()

        words = sent.split(' ')
        n_words = len(words)
        keep = False
        for j in range(n_words):
            cur_word = words[j].lower()
            if cur_word not in type_set:
                continue
            if ends_with(words, j - 1, ['a']) or ends_with(words, j - 1, ['the']) or ends_with(
                    words, j - 1, ['and', 'other']) or ends_with(words, j - 1, ['and', 'some', 'other']):
                keep = True
            if starts_with(words, j + 1, ['such', 'as']):
                keep = True
                # print(cur_word, '&', sent)
                # exit()

        if i > 10:
            break
    f.close()


def filter_not_noun_sents():
    output_types_file = os.path.join(config.DATA_DIR, 'ultrafine/res/enwiki-20151002-type-sents-s01-filter-types.txt')
    output_sents_file = os.path.join(config.DATA_DIR, 'ultrafine/res/enwiki-20151002-type-sents-s01-filter.txt')
    type_vocab_file = os.path.join(config.DATA_DIR, 'ultrafine/uf_data/ontology/types.txt')
    type_sents_file = os.path.join(config.DATA_DIR, 'ultrafine/res/enwiki-20151002-type-sents.txt.gz')
    pos_tags_file = os.path.join(config.DATA_DIR, 'ultrafine/res/enwiki-20151002-type-sents-postag-s01.txt')

    type_vocab, type_id_dict = datautils.load_vocab_file(type_vocab_file)
    type_dict = get_type_str_dict(type_vocab)

    ends_with_words = [['a'], ['the'], ['and', 'other'], ['and', 'some', 'other'], ['and', 'any', 'other']]
    starts_with_words = ['such', 'as']
    sent_id = -1
    sent = None
    keep_cnt = 0
    f = gzip.open(type_sents_file, 'rt', encoding='utf-8')
    f_pos = open(pos_tags_file, encoding='utf-8')
    for i, line in enumerate(f_pos):
        pos_tags = line.strip().split(' ')
        cur_sent_id = int(pos_tags[0])
        pos_tags = pos_tags[1:]
        # print(line.strip())

        while sent_id < cur_sent_id:
            sent = next(f).strip()
            # sent = line.strip()
            sent_id += 1

        words = sent.split(' ')
        n_words = len(words)
        assert n_words == len(pos_tags)
        # print(words)
        # print(pos_tags)
        # print()
        keep = False
        for j in range(n_words):
            cur_word = words[j].lower()
            t = type_dict.get(cur_word, None)
            if t is None:
                continue
            if pos_tags[j] not in {'NN', 'NNP', 'NNPS'}:
                continue

            # if any(ends_with(words, j - 1, e_words) for e_words in ends_with_words):
            #     keep = True
            # if not keep and any(starts_with(words, j + 1, s_words) for s_words in starts_with_words):
            #     keep = True

            for e_words in ends_with_words:
                if ends_with(words, j - 1, e_words):
                    # print(cur_word, '&', sent)
                    keep = True
                    break

            if keep:
                break

            for s_words in starts_with_words:
                if starts_with(words, j + 1, s_words):
                    # print(cur_word, '&', sent)
                    keep = True
                    break

            if keep:
                # print(t, '&', sent)
                break

        #     # if ends_with(words, j - 1, ['a']) or ends_with(words, j - 1, ['the']) or ends_with(
        #     #         words, j - 1, ['and', 'other']) or ends_with(words, j - 1, ['and', 'some', 'other']):
        #     #     keep = True
        #     # if starts_with(words, j + 1, ['such', 'as']):
        #     #     keep = True
        #         # print(cur_word, '&', sent)
        #         # exit()

        if keep:
            keep_cnt += 1

        if i % 10000 == 0:
            print(i, keep_cnt)

        # if i > 1000:
        #     break
        # if i > 10:
        #     break
    f.close()
    f_pos.close()
