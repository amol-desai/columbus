from nltk.tokenize import sent_tokenize
#need to download the punkt nltk corpus for sentence tokenizer to work
#from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
#from sklearn.feature_extraction.text import TfidfVectorizer
import re,string,math

def pagerank(graph,d):
    """
    Args:
        graph - A similarity matrix that acts as a graph adjacency matrix giving edge weights i.e. probability of moving between two nodes or sentences (Array of Arrays)
        d - page rank probability of taking the edge vs jumping to a new node

    Returns:
        page rank of each node in the graph
    """
    prev_pr,new_pr = np.zeros(len(graph)),np.ones(len(graph))
    #compute page rank until convergence
    while np.sum(np.abs(new_pr-prev_pr)) > 0.001:
        prev_pr = new_pr.copy()
        for i in range(len(graph)):
            new_pr[i] = np.sum(graph[:,i]*d+np.ones(len(graph))/len(graph)*(1-d))
    #normalize the page rank
    return new_pr/sum(new_pr)

def highlight_doc(doc,query='',n=3):
    """
    Args:
        doc - Document to be highlighted (string)
        query - The search query (string,optional)
        n - number of top sentences

    Returns:
        The most relevant snippet with the query terms highlighted (string)
    """
    #this implementation picks out the top n sentences. This might be a problem
    #for reviews with long sentences. An improvement might be to truncate based on num of words or characters
    doc = re.sub('\n+',' ',re.sub(r'([a-z|A-Z|0-9]) *\n',r'\1. ',doc))
    sentences = sent_tokenize(doc)
    v= TfidfVectorizer(decode_error='ignore',strip_accents='unicode',
                       ngram_range=(1,3),stop_words='english',
                       max_df=0.5,min_df=0.0,norm='l2',use_idf=True)
    tfidf = v.fit_transform(sentences)
    similarity = cosine_similarity(tfidf)
    
    pg_rank = pagerank(similarity,0.8)
    #pick the first three most important sentences and put them in order of appearance
    picked_indices = np.sort(np.argsort(pg_rank)[::-1][0:20])
    toret_str = ''
    #append ... to either end of snippet when needed to indicate snips
    #put the summary together
    for i in picked_indices:
        if (i-1) not in picked_indices and i != 0:
            toret_str = toret_str+'...'
        toret_str = toret_str+sentences[i]
    if i+1<len(sentences):
        toret_str = toret_str+'...'
    #do the highlights
    #toret_str = highlight(query.split(),toret_str)
    return toret_strs
