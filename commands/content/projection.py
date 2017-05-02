from clint.textui import progress
from pylab import rcParams
from gensim import corpora, models
import json
import matplotlib.pyplot as plt
import output
import pandas
import re
import requests
import spiderpig as sp
import subprocess


@sp.cached()
def get_majka_lemma(word):
    echo = subprocess.Popen(('echo', word), stdout=subprocess.PIPE)
    output = subprocess.check_output('./majka -f majka.w-lt'.split(), stdin=echo.stdout)
    echo.wait()
    output = output.decode("utf-8").split(':')
    if len(output) > 0:
        return output[0]
    return word


def _get_words(text):
    for word in re.split('[ \.,;/\(\)\-\\n"\[\]]', text):
        word = word.strip().lower()
        if len(word) <= 2 or word.isdigit():
            continue
        yield word


@sp.cached()
def get_morphodita_words(text):
    response = requests.post('http://lindat.mff.cuni.cz/services/morphodita/api/tag?output=json', data={'data': text})
    return [word['lemma'] for sentence in json.loads(response.text)['result'] for word in sentence]


@sp.cached()
def get_majka_words(text):
    result = []
    for word in _get_words(text):
        result.append(get_majka_lemma(word))
    return result


def get_words(filename, method='majka'):
    with open(filename, 'r') as f:
        text = f.read()
        if method == 'raw':
            result = []
            for word in _get_words(text):
                result.append(word)
            return result
        elif method == 'majka':
            return get_majka_words(text)
        elif method == 'morphodita':
            return get_morphodita_words(text)


@sp.cached()
def load_records(data_dir='data', decision_type=None, limit=None):
    data = pandas.read_csv('{}/data.csv'.format(data_dir), index_col=False, delimiter=';')
    if decision_type is not None:
        data = data[data["Forma rozhodnutí"] == decision_type]
    return data if limit is None else data.head(n=limit)


@sp.cached()
def load_normalized_texts(data_dir='data', decision_type=None, limit=None, method='raw'):
    documents = {}
    for ecli, filename in progress.bar(load_records(decision_type=decision_type, limit=limit)[['Identifikátor evropské judikatury', 'txt_file']].values):
        documents[ecli] = get_words(filename.replace('./target', data_dir), method=method)
    return documents


@sp.cached()
def get_corpus(decision_type=None, limit=None, method='raw'):
    normalized_texts = load_normalized_texts(decision_type=decision_type, limit=limit, method=method)
    sorted_eclis, sorted_texts = zip(*sorted(normalized_texts.items()))
    dictionary = corpora.Dictionary([words for words in sorted_texts])
    corpus = [dictionary.doc2bow(text) for text in sorted_texts]
    return sorted_eclis, corpus, dictionary


@sp.cached()
def get_transformation(groups, decision_type=None, limit=None, method='raw'):
    _, corpus, dictionary = get_corpus(decision_type=decision_type, limit=limit, method=method)
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=groups)
    return lsi


@sp.cached()
def get_projection(decision_type=None, limit=None, method='raw'):
    ecli_judge = load_records(decision_type=decision_type, limit=limit).set_index('Identifikátor evropské judikatury')['Soudce zpravodaj'].to_dict()
    transformation = get_transformation(2, decision_type=decision_type, limit=limit, method=method)
    eclis, corpus, _ = get_corpus(decision_type=decision_type, limit=limit, method=method)
    result = []
    for ecli, data in zip(eclis, transformation[corpus]):
        judge = ecli_judge[ecli]
        result.append({
            'judge': judge,
            'ecli': ecli,
            'x': data[0][1],
            'y': data[1][1],
        })
    return pandas.DataFrame(result)


def get_topics(decision_type=None, limit=None, method='raw'):
    transformation = get_transformation(2, decision_type=decision_type, limit=limit, method=method)
    result = []
    for t in transformation.print_topics():
        result.append(t[1])
    return result


def plot_judge_projection(projection, topics):
    projection = projection[['judge', 'x', 'y']].groupby(['judge']).mean().reset_index().sort_values(by='judge')
    projection['judge_acronym'] = projection['judge'].apply(lambda name: '{}{}'.format(name.split()[0][:3], name.split()[1][0]))
    rcParams['figure.figsize'] = 10, 10
    for judge, acronym, x, y in projection[['judge', 'judge_acronym', 'x', 'y']].values:
        plt.scatter(x, y, label='{}: {}'.format(acronym, judge), color='white', linewidth=0)
        plt.scatter(x, y, color='black')
        plt.text(x, y, acronym, fontsize='small', )
    plt.xlabel(' + '.join([t for t in topics[0].split() if t != '+'][:3]) + ' + ...')
    plt.ylabel(' + '.join([t for t in topics[1].split() if t != '+'][:3]) + ' + ...')
    plt.legend(loc='center left', fontsize='xx-small', bbox_to_anchor=(0.95, 0.5))
    output.savefig('content_projection')


def execute(limit=None, method='raw'):
    projection = get_projection(decision_type='Nález', limit=limit, method=method)
    topics = get_topics(decision_type='Nález', limit=limit, method=method)
    plot_judge_projection(projection, topics)
