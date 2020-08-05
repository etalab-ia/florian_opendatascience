import opendatascience_module as opd
import pandas as pd
import sys
import numpy as np
argv=sys.argv
def load_queries(path='../../data/querys.csv'):
    df=pd.read_csv('../../data/querys.csv', sep=',',error_bad_lines=False, encoding='latin-1')
    ids=df['expected']
    queries=np.array(df['query'], dtype=str)
    return ids, queries

global embeddings
embeddings= opd.get_large_files('../../data/test')[0]

def first_found_pos(queries):
    positions=[101]*len(queries)
    for i in range(len(queries)):
        results=opd.Search(queries[i],100)[0]
        k=0
        for r in results:
            k=k+1
            if r==ids[i]:
                positions[i]=k
    return positions

def classify_results(queries, positions):
    acceptable=[]
    rejected=[]
    bof=[]
    for q in range(len(queries)):
        if positions[q]==100:
            rejected.append((queries[q], positions[q]))
        else:
            if positions[q]<=5:
                 acceptable.append((queries[q], positions[q]))
            else:
                bof.append((queries[q], positions[q]))
    return acceptable, bof, rejected

def bench_score(queries):
    positions=first_found_pos(queries)
    score=0
    for i in positions:
        score+=101-i
    score=score/len(queries)
    retrun(score)
    
if __name__ == "__main__":
    """Main function
    Produces the embeddings from the given CSV file location
    
    Keyword argument:
    file_dir (str): File location of the csv (catalog standard)
    new_location (str): new location folder for saving data
    df_save_opt (bool): to activate or not saving the dataframe

    Returns:
    embeddings (npy of floats) : The 1024 embedding of each dataset as a "big sentence"
    id_table (python list of ints): list with the ids in the order of the dfx 
    """
    bench_score(load_queries(argv[0])[1])