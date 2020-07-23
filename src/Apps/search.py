import requests

import dash
import dash_dangerously_set_inner_html
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output


import plotly.express as px
import pandas as pd

#######

##### Takes a search query as input and get the vectors from the whole dataset to compare.
import spacy
import pickle
import numpy as np
from scipy import spatial
import sys
import unidecode
#from sklearn.decomposition import PCA
#QUERY  Neighbours Ids_and_Score_bool
directory='../'
argv=sys.argv
nlp = spacy.load("fr_core_news_lg")
pca = pickle.load(open(directory+'models/pca_30.pkl','rb'))
pca_space= np.load(directory+'models/vectors_pca_30.npy', allow_pickle=True)
id_table=list(np.load(directory+'../data/id_table.npy', allow_pickle=True))
data=pd.read_csv('../../data/Catalogue_locs.csv', sep=',',error_bad_lines=False, encoding='latin-1')
tree = spatial.KDTree(pca_space)
from spacy.lang.fr.stop_words import STOP_WORDS
from spacy.lang.fr import French
parser=French()
stopwords = list(STOP_WORDS)

def process_query(search_query):
    query=str(search_query).lower()
    clean_query = unidecode.unidecode(query)
    tokens=parser(clean_query)
    tokens = [ word.lower_ for word in tokens ]
    tokens = [ word for word in tokens if word not in stopwords]
    tokens = " ".join([i for i in tokens])
    return (tokens)

def query2vec(search_query):
    x=nlp(search_query).vector #spacy 300d
    y=pca.transform([x])[0] #pca 30d
    return(y)

def get_id(idx):
    dataset_id=id_table[idx-1]
    return(dataset_id)

def get_idx(ids):
    dataset_idx=id_table.index(ids)
    return(dataset_idx)

def id2vec(ids):
    return(pca_space[get_idx(ids)])

def neighbours(vector, n):
    n_ids=[]
    score=[]
    dist, pos=tree.query(vector, k=n)
    for j in range(len(pos)):
        n_ids.append(get_id(pos[j]))
        score.append(1-dist[j]/50) ##very approximate metric
    return(n_ids, score)

def Search(search_query, n):
    n_ids, score=neighbours(query2vec(process_query(search_query)), n)
    #print(n_ids, score)
    return(n_ids, score)

def Similarity(ids, n):
    n_ids, score=neighbours(id2vec(ids), n)
    return(n_ids, score)
#######

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div(children=[
    html.Div(["Input: ", dcc.Input(id='my-input', value='association', type='text')], width={"size": 6, "offset": 3}),
    html.H3("Résultats :"),
    dbc.Spinner(html.Div(id='my-output')),
])

def dataset(ids):
    #res = requests.get(f'https://www.data.gouv.fr/api/1/datasets/{ids}')
    #return res.json()
    idx=get_idx(ids)
    #print(idx)
    df=data.loc[idx]
    return(df)
def embed(ids, score):
    # html = dash_dangerously_set_inner_html.DangerouslySetInnerHTML(f"""
    #     <div data-udata-dataset="{id}"></div>
    #     <script data-udata="https://www.data.gouv.fr/" src="https://static.data.gouv.fr/static/oembed.js" async defer></script>
    # """)

    ds = dataset(ids)
    print(ds)
    output = html.Tr([
        html.Td(html.A(ds['id'], href=f"https://www.data.gouv.fr/fr/datasets/{ids}")),
        html.Td(html.A(ds['title'], href=f"https://www.data.gouv.fr/fr/datasets/{ids}")),
        html.Td(ds['description']),
        html.Td(ds['pred locs']),
        html.Td(score),
    ])

    return output

@app.callback(
    Output(component_id='my-output', component_property='children'),
    [Input(component_id='my-input', component_property='value')]
)
def update_output_div(input_value):
    knns=Search(input_value, 10)
    results = [ embed(i, j) for i,j in zip(knns[0], knns[1] ) ]
  
    # print(f" 👀 {input_value}")
    # print(f" results: {results}")
#    results = html.Div([ embed(r) for r in Search(input_value, 10) ])
#    return 'Output: {}'.format(Search(input_value, 10))

    header = [
        html.Thead(html.Tr([
            html.Th("id"),
            html.Th("titre"),
            html.Th("description"),
            html.Th("Localisation"),
            html.Th("Distance"),
        ]))
    ]

    return dbc.Table(header + [html.Tbody(results)], bordered=True, hover=True, responsive=True,  striped=True)

if __name__ == '__main__':
    app.run_server(port=8050, debug=True)