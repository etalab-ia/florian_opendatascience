##
import sys
import re
import pandas as pd
import numpy as np
#
import torch
from sentence_transformers import SentenceTransformer, models
##
cp_1252_chars = {
    # from http://www.microsoft.com/typography/unicode/1252.htm
    u"\x80": u"\u20AC", # EURO SIGN
    u"\x82": u"\u201A", # SINGLE LOW-9 QUOTATION MARK
    u"\x83": u"\u0192", # LATIN SMALL LETTER F WITH HOOK
    u"\x84": u"\u201E", # DOUBLE LOW-9 QUOTATION MARK
    u"\x85": u"\u2026", # HORIZONTAL ELLIPSIS
    u"\x86": u"\u2020", # DAGGER
    u"\x87": u"\u2021", # DOUBLE DAGGER
    u"\x88": u"\u02C6", # MODIFIER LETTER CIRCUMFLEX ACCENT
    u"\x89": u"\u2030", # PER MILLE SIGN
    u"\x8A": u"\u0160", # LATIN CAPITAL LETTER S WITH CARON
    u"\x8B": u"\u2039", # SINGLE LEFT-POINTING ANGLE QUOTATION MARK
    u"\x8C": u"\u0152", # LATIN CAPITAL LIGATURE OE
    u"\x8E": u"\u017D", # LATIN CAPITAL LETTER Z WITH CARON
    u"\x91": u"\u2018", # LEFT SINGLE QUOTATION MARK
    u"\x92": u"\u2019", # RIGHT SINGLE QUOTATION MARK
    u"\x93": u"\u201C", # LEFT DOUBLE QUOTATION MARK
    u"\x94": u"\u201D", # RIGHT DOUBLE QUOTATION MARK
    u"\x95": u"\u2022", # BULLET
    u"\x96": u"\u2013", # EN DASH
    u"\x97": u"\u2014", # EM DASH
    u"\x98": u"\u02DC", # SMALL TILDE
    u"\x99": u"\u2122", # TRADE MARK SIGN
    u"\x9A": u"\u0161", # LATIN SMALL LETTER S WITH CARON
    u"\x9B": u"\u203A", # SINGLE RIGHT-POINTING ANGLE QUOTATION MARK
    u"\x9C": u"\u0153", # LATIN SMALL LIGATURE OE
    u"\x9E": u"\u017E", # LATIN SMALL LETTER Z WITH CARON
    u"\x9F": u"\u0178", # LATIN CAPITAL LETTER Y WITH DIAERESIS
} 
def fix_1252_codes(text):
    """
    Replace non-standard Microsoft character codes from the Windows-1252 character set in a unicode string with proper unicode codes.
    Code originally from: http://effbot.org/zone/unicode-gremlins.htm
    """
    if re.search(u"[\x80-\x9f]", text):
        def fixup(m):
            s = m.group(0)
            return cp_1252_chars.get(s, s)
        if isinstance(text, type("")):
            text = str(text)
        text = re.sub(u"[\x80-\x9f]", fixup, text)
    return text 

#
argv=sys.argv

def process_file(file_dir):
    """
       CSV file loaded and concatenated to a dataframe with the usefull columns:
       'title', 'organization', 'description','tags'

    Keyword argument:
    file_dir (str): File location of the csv (catalog standard)

    Returns:
    dfx (pandas dataframe of str) :Aggregated text
    id_table (python list of ints): list with the ids in the order of the dfx
    """
    df=pd.read_csv(file_dir, sep=';',error_bad_lines=False, encoding='latin-1') #le fichier original
    id_table=np.array(list(df['id']), dtype=object)
    df=df.astype(str)
    dfx=df[['title', 'organization', 'description','tags']].agg('. '.join, axis=1)
    return dfx, id_table
    
def filter_df(dff, table, new_location):
    too_short=[]
    
    for i in range(len(dff)):
        if len(dff[i])<45:
            too_short.append(i)
    
    dff.drop(dff.index[too_short], inplace=True)
    table=np.delete(table, too_short, axis=0)
    np.save(new_location+'/id_table.npy', table, allow_pickle=True, fix_imports=True)
    
    return dff, table
    
def super_cleaning(dfx):
    """dataframe is cleaned of the unicode errors, separators, spaces extensions links etc..
    objective is to keep clean text.

    Keyword argument:
    dfx (pandas dataframe of str): df file with 1 column only ad text inside

    Returns:
    dfx (pandas dataframe of str) :the same df but the text inside is 'cleaned'
   """
    for i in range(len(dfx)):
        dfx[i]=fix_1252_codes(dfx[i])
        dfx[i]=re.sub(r'\S*.(fr|org|com|map|jpeg|jpg|zip|png)\b','', dfx[i])
        dfx[i]=re.sub(r"""
               [,;@#/\\?*[\]\r\n\<\>\_\{\}\»\«\\\(\)!&$]+  # Accept one or more copies of punctuation
               \ *           # plus zero or more copies of a space,
               """,
               " ",          # and replace it with a single space
               dfx[i], flags=re.VERBOSE)
        
        dfx[i]=re.sub(r'\\x\w+',' ', dfx[i])
        dfx[i]=re.sub(r'\bx\w+', '', dfx[i])
        dfx[i]=re.sub(r'\S+\d+\w+', " ", dfx[i])
        dfx[i]=re.sub(r'(?:^| )\w(?:$| )',' ',dfx[i])
        dfx[i]=re.sub(' +', ' ', dfx[i])
    return(dfx)


#dataset_in=np.load('../../data/cleaned2.npy', allow_pickle=True) en gros
#path_out='../../data/multilingual_embeddings.npy'
def embedder(dataset_in, path_out, save):
    """
    Multilingual SBERT embedding of a pandas 1d dataset 

    Keyword argument:
    dataset_in (npy array of str): the cleanest and most filtered

    Returns:
    embeddings (npy of floats) : The 512 embedding of each dataset as a "big sentence"
    """
    model = SentenceTransformer('distiluse-base-multilingual-cased')
    dataset_in=dataset_in.to_numpy()
    embeddings= model.encode(dataset_in, batch_size=8, show_progress_bar=True)
    if save:
        np.save(path_out+'/embeddings.npy', embeddings, allow_pickle=True)
    return(embeddings)

def save_to(df_clean, new_location):
    """
     Save the dataframe to new_location folder both in pkl and npy format

    Keyword argument:
    df_clean (pandas dataframe): stored for further use
    new_location (str): new location folder for saving data

    Returns:
    none 
    """
    df_clean.to_pickle(new_location+'/df_clean.pkl')
    df_clean_np=df_clean.to_numpy(dtype=object)
    np.save(new_location+'/df_clean.npy', df_clean_np, allow_pickle=True, fix_imports=True)

def clean(file_dir, new_location, df_save_opt ):
    """
    Main function
    Produces the embeddings from the given CSV file location
    

    Keyword argument:
    file_dir (str): File location of the csv (catalog standard)
    new_location (str): new location folder for saving data
    df_save_opt (bool): to activate or not saving the dataframe

    Returns:
    embeddings (npy of floats) : The 1024 embedding of each dataset as a "big sentence"
    id_table (python list of ints): list with the ids in the order of the dfx
    """
    processed, table=process_file(file_dir)
    df_clean=super_cleaning(processed)
    df_clean_filtered, id_table=filter_df(df_clean, table, new_location)
    embeddings=embedder(df_clean_filtered, new_location, True) 
                          
    if df_save_opt:
        save_to(df_clean, new_location)
    return embeddings, id_table 

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
    clean(argv[1], argv[2], argv[3])