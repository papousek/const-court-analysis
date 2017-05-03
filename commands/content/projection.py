from clint.textui import progress
from pylab import rcParams
from gensim import corpora, models, matutils
from sklearn.manifold import TSNE, SpectralEmbedding
import json
import matplotlib.pyplot as plt
import output
import pandas
import re
import requests
import seaborn as sns
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
def get_morphodita_words(filename):
    with open(filename, 'r') as f:
        text = f.read()
        response = requests.post('http://lindat.mff.cuni.cz/services/morphodita/api/tag?output=json', data={'data': text})
        return [word['lemma'] for sentence in json.loads(response.text)['result'] for word in sentence]


@sp.cached()
def get_majka_words(filename):
    with open(filename, 'r') as f:
        text = f.read()
        result = []
        for word in _get_words(text):
            result.append(get_majka_lemma(word))
        return result


def get_words(filename, method='majka'):
    if method == 'raw':
        with open(filename, 'r') as f:
            text = f.read()
            result = []
            for word in _get_words(text):
                result.append(word)
            return result
    elif method == 'majka':
        return get_majka_words(filename)
    elif method == 'morphodita':
        return get_morphodita_words(filename)


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
def get_corr(decision_type=None, limit=None, method='raw'):
    _, corpus, dictionary = get_corpus(decision_type=decision_type, limit=limit, method=method)
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    matrix = pandas.DataFrame(matutils.corpus2dense(corpus_tfidf, num_terms=len(dictionary)))
    return matrix.corr()



@sp.cached()
def get_gensim_model(name, groups, decision_type=None, limit=None, method='raw'):
    _, corpus, dictionary = get_corpus(decision_type=decision_type, limit=limit, method=method)
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    mapping = {
        'lsi': models.LsiModel,
        'lda': models.LdaModel,
        'tfidf': models.TfidfModel,
    }
    kwargs = {}
    if name != 'tfidf':
        kwargs['num_topics'] = groups
    model = mapping[name](corpus_tfidf, id2word=dictionary, **kwargs)
    topics = None
    if hasattr(model, 'print_topics'):
        topics = []
        for t in model.print_topics():
            topics.append(t[1])
    return model, topics


@sp.cached()
def get_projection(decision_type=None, limit=None, method='raw', trans_method='lda'):
    ecli_judge = load_records(decision_type=decision_type, limit=limit).set_index('Identifikátor evropské judikatury')['Soudce zpravodaj'].to_dict()
    eclis, corpus, dictionary = get_corpus(decision_type=decision_type, limit=limit, method=method)
    result = []
    if trans_method in ['lda', 'lsi', 'tfidf']:
        transformation, _ = get_gensim_model(trans_method, 2, decision_type=decision_type, limit=limit, method=method)
        for ecli, data in zip(eclis, transformation[corpus]):
            data = dict(data)
            judge = ecli_judge[ecli]
            result.append({
                'judge': judge,
                'ecli': ecli,
                'x': data.get(0, 0),
                'y': data.get(1, 0),
            })
    else:
        matrix = get_corr(decision_type=decision_type, limit=limit)
        if trans_method == 'tsne':
            model = TSNE(n_components=2)
        elif trans_method == 'spectral':
            model = SpectralEmbedding(n_components=2)
        for ecli, data in zip(eclis, model.fit_transform(matrix)):
            judge = ecli_judge[ecli]
            result.append({
                'judge': judge,
                'ecli': ecli,
                'x': data[0],
                'y': data[1],
            })
    return pandas.DataFrame(result)


def plot_judge_projection(projection):
    projection = projection[['judge', 'x', 'y']].groupby(['judge']).mean().reset_index().sort_values(by='judge')
    projection['judge_acronym'] = projection['judge'].apply(lambda name: '{}{}'.format(name.split()[0][:3], name.split()[1][0]))
    rcParams['figure.figsize'] = 10, 10
    for judge, acronym, x, y in projection[['judge', 'judge_acronym', 'x', 'y']].values:
        plt.scatter(x, y, label='{}: {}'.format(acronym, judge), color='white', linewidth=0)
        plt.scatter(x, y, color='black')
        plt.text(x, y, acronym, fontsize='small', )
    plt.legend(loc='center left', fontsize='xx-small', bbox_to_anchor=(0.98, 0.5))
    output.savefig('judge_projection')


def plot_projection(projection):
    projection = projection.sort_values(by='judge')
    xmin, xmax = projection['x'].quantile(0.01), projection['x'].quantile(0.99)
    ymin, ymax = projection['y'].quantile(0.01), projection['y'].quantile(0.99)
    g = sns.FacetGrid(projection, col='judge', col_wrap=5, ylim=(ymin, ymax), xlim=(xmin, xmax))
    g.map(plt.scatter, 'x', 'y', alpha=0.5).set_titles('{col_name}')
    output.savefig('projection')


def execute(limit=None, method='raw', trans_method='lda'):
    projection = get_projection(decision_type='Nález', limit=limit, method=method, trans_method=trans_method)
    if trans_method in ['lsi', 'lda', 'tfidf']:
        _, topics = get_gensim_model(trans_method, 2, decision_type='Nález', limit=limit, method=method)
        if topics is not None:
            print('X', topics[0])
            print('Y', topics[1])
    plot_projection(projection)
    plot_judge_projection(projection)
