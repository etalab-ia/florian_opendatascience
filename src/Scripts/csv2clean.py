##
import sys
import re
import pandas as pd
import numpy as np
#
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

#
argv=sys.argv

def process_file(file_dir):
    df=pd.read_csv(file_dir, sep=';',error_bad_lines=False, encoding='latin-1') #le fichier original
    df=df.astype(str)
    dfx=df[['title', 'organization', 'description','tags']].agg('. '.join, axis=1)
    return(dfx)

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

def super_cleaning(dfx):
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

def save_to(df_clean, new_location):
    df_clean.to_pickle(new_location+'/df_clean.pkl')
    df_clean_np=df_clean.to_numpy()
    np.save(new_location+'/df_clean.npy', df_clean_np, allow_pickle=True, fix_imports=True)

def clean(file_dir, new_location ):
    df_clean=super_cleaning(process_file(file_dir))
    if new_location != 'none':
        save_to(df_clean, new_location)
    return(df_clean)

if __name__ == "__main__":
    clean(argv[1], argv[2])
