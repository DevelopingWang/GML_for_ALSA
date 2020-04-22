from collections import namedtuple


Feature = namedtuple('Feature', ['index', 'word', 'polarity', 'score', 'pos', 'type'])
UniFeature = namedtuple('Feature', ['index', 'word', 'polarity', 'score', 'pos', 'type'])

AspectUniFeature = namedtuple('AspectUniFeature', ['index', 'word', 'polarity', 'score', 'pos', 'type',
                                             'aspect', 'entity','attribute', 'tid', 'aspect_clue'])

class Review(object):
    def __init__(self, sentences):
        self.sentences = sentences


class Sentence(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class Aspect(object):
    def __init__(self, category, tid, polarity):
        self.category = category
        self.tid = tid
        self.polarity = polarity


# class UniFeature(object):
#     def __init__(self, index, word, polarity, score, pos, type):
#         self.index = index
#         self.word = word
#         self.polarity = polarity
#         self.score = score
#         self.pos = pos
#         self.type = type


class NgramFeature(object):
    def __init__(self, indexs, tokens):
        self.indexs = indexs
        self.tokens = tokens

##############################################################################################
##  Class for factor graph
##############################################################################################
class AspectNode(object):
    def __init__(self, name, polarity, gold_polarity, prior_score=0):
        self.name = name
        self.polarity = polarity
        self.prior = prior_score
        self.isEvidence = True if (polarity== 'positive') or (polarity== 'negative') else False
        self.gold_polarity = gold_polarity

class FeatureNode(object):
    def __init__(self, name, polarity, word, prior_score, pos, type):
        self.name = name
        self.word = word
        self.polarity = polarity
        self.prior = prior_score
        self.pos = pos
        self.type = type
        self.isEvidence = True if (polarity== 'positive') or (polarity== 'negative') else False
        # self.isEvidence = True if type == 'strong_lexi_fea' else False

class BinaryRelation():
    def __init__(self, name1, name2, rel_type):
        self.name1 = name1
        self.name2= name2
        self.rel_type = rel_type


# def get_sentence_object(df_data_sid):
#     text, num_aspects, token, pos_tag, clean_token, clean_pos_tag, deepclean_token, deepclean_pos_tag = \
#         df_data_sid[['text', 'num_aspects', 'token', 'pos_tag', 'clean_token', 'clean_pos_tag',
#                         'deepclean_token', 'deepclean_pos_tag']].values[0]
#     tids = df_data_sid['t_id'].values
#     aspects = df_data_sid['category'].values
#     gold_polarities = df_data_sid['gold_polarity'].values
#     sent = Sentence(text=text,
#                     num_aspects=num_aspects,
#                     token=token,
#                     pos_tag=pos_tag,
#                     clean_token=clean_token,
#                     clean_pos_tag=clean_pos_tag,
#                     deepclean_token=deepclean_token,
#                     deepclean_pos_tag=deepclean_pos_tag,
#                     aspects=aspects,
#                     gold_polarities=gold_polarities,
#                     tids=tids)
#     return sent

def get_sentence_object(df_data_sid):
    text, num_aspects, token, pos_tag, clean_token, clean_pos_tag, deepclean_token, deepclean_pos_tag, \
    opinion_phrases, dep_relations, const_parse_tree = \
        df_data_sid[['text', 'num_aspects', 'token', 'pos_tag', 'clean_token', 'clean_pos_tag',
                        'deepclean_token', 'deepclean_pos_tag', 'opinion_phrases', 'dep_relations', 'const_parse_tree']].values[0]
    tids = df_data_sid['t_id'].values
    aspects = df_data_sid['category'].values
    gold_polarities = df_data_sid['gold_polarity'].values
    entities = df_data_sid['entity'].values
    attributes = df_data_sid['attribute'].values
    sent = Sentence(text=text,
                    num_aspects=num_aspects,
                    token=token,
                    pos_tag=pos_tag,
                    clean_token=clean_token,
                    clean_pos_tag=clean_pos_tag,
                    deepclean_token=deepclean_token,
                    deepclean_pos_tag=deepclean_pos_tag,
                    aspects=aspects,
                    gold_polarities=gold_polarities,
                    tids=tids,
                    entities = entities,
                    attributes = attributes,
                    opinion_phrases = opinion_phrases,
                    dep_relations = dep_relations,
                    const_parse_tree = const_parse_tree)
    return sent



