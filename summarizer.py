from nltk.tokenize import sent_tokenize
#need to download the punkt nltk corpus for sentence tokenizer to work
#from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import numpy as np
#from sklearn.feature_extraction.text import TfidfVectorizer
import re,string,math
from scipy.sparse import csr_matrix,lil_matrix

def pagerank_undirected(graph,d):
    """
    Args:
        graph - Unnormalized(or normalized) transition matrix in csr format
        d - page rank probability of taking the edge vs jumping to a new node

    Returns:
        page rank of each node in the graph
    """
    normalize(graph,norm='l1',axis=1,copy=False) #normalize as transition probability per row(node)

    #create matrix for which to find principal eigenvector
    M = d*graph #weighted transition probability
    try:
        jmp_temp = (1-d)/graph.shape[0] #teleportation weighting
        J = np.ones(graph.shape) 
        J = jmp_temp*J #teleportation probability matrix
        M = M+J
    except ValueError:
        pass

    #intialize normalized rankings
    R = np.matrix(np.random.random(graph.shape[0])).transpose()
    normalize(R,norm='l1',axis=0,copy=False)

    Rp = R*np.Inf

    #iterate until convergence
    while (np.square(R-Rp).sum() > 0.001):
        Rp = R
        R = M.dot(R)

    #convert R to numpy array instead of matrix before returning.
    return np.array(R).reshape(len(R),)

def text_cleanup(text):
    #posts are separated by newline in addition to any newlines within the post.
    #If lines were not truncated by a punctuation, truncate with period to form sentence
    #boundary and remove the newline.
    #Convert entire text into single str
    cleanText = re.sub('\n+',' ',re.sub(r'([a-z|A-Z|0-9]) *\n',r'\1. ',text))
    return cleanText


def gen_sentence_vector(text,query='',removeQuestions=True):
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
    cleanText = text_cleanup(text)
        
    sentences = sent_tokenize(cleanText)
    toRem = []
    for i,sentence in enumerate(sentences):
        if len(sentence.split())<4 or (sentence.endswith('?') and removeQuestions):
            toRem.append(i)
    sentences=np.delete(sentences,toRem)
    
    if query is not '':
        sentences=np.append(sentences,query)

    return sentences

def cosine_similarity_sparse(spMat):
    spMat_n = normalize(spMat,norm='l2',axis=1)
    cos_sim = spMat_n*spMat_n.transpose()
    return cos_sim
    

def gen_similarity_matrix(sent_vect,ngrams=(3,4)):

    for ng in range(ngrams[0],ngrams[1]+1):
        try:
            v= TfidfVectorizer(decode_error='ignore',strip_accents='unicode',
                               ngram_range=(ng,ngrams[1]),stop_words='english',
                               max_df=0.5,min_df=0.0,norm='l2',use_idf=True)

            tfidf = v.fit_transform(sent_vect)

            #this creates a dense matrix which blows up
            #similarity = lil_matrix(cosine_similarity(tfidf))

            #compute cosine similarity
            similarity = lil_matrix(cosine_similarity_sparse(tfidf))
            
            similarity.setdiag(np.zeros(similarity.shape[0])) #remove self directed links from all nodes
            similarity=csr_matrix(similarity)
        except ValueError:
            pass
        else:
            break
        
    return similarity

def get_indices_for_similar_sentences(similarity,idxSentToMatch):
    relevant_indices = np.array(similarity[idxSentToMatch].todense()
                                    .nonzero()[1])
    relevant_indices = relevant_indices.reshape(relevant_indices.shape[1],)
    return relevant_indices

def summarize(text,query='',removeQuestions=True,topNT=3,topNS=3,jumpProb = 0.2,inOrder=False):

    sentences = gen_sentence_vector(text,query,removeQuestions=True)
    if query is not '':
        similarity = gen_similarity_matrix(sentences,(1,len(query.split())))
    else:
        similarity = gen_similarity_matrix(sentences)

    pgRank = 0
    for i in range(5):    
        pgRank = pgRank+pagerank_undirected(similarity,1-jumpProb)
    pgRank = pgRank/5

    if query is not '':
        relevant_indices = get_indices_for_similar_sentences(similarity,len(sentences)-1)
        #only look at sentences similar to the query
        picked_indices = relevant_indices[np.argsort(pgRank[relevant_indices])
                                          [::-1][0:topNS]]
    else:
        #pick the first n most important sentences and put them in order of appearance
        picked_indices = np.argsort(pgRank)[::-1][0:topNT]
        for i in picked_indices:
            relevant_indices = get_indices_for_similar_sentences(similarity,i)
            picks_from_relevant = relevant_indices[np.argsort(pgRank[relevant_indices])
                                                   [::-1][0:topNS]]
            for j in picks_from_relevant:
                if j not in picked_indices:
                    picked_indices = np.insert(picked_indices,np.where(picked_indices==i)
                                               [0][0]+1,picks_from_relevant)
            
    toret_str = ''
    picked_indices = np.unique(picked_indices)
    if inOrder:
        picked_indices = np.sort(picked_indices)
    #put the summary together in order of importance
    #print picked_indices
    for i in picked_indices:
        toret_str = toret_str+'\n\n'+sentences[i]
    return toret_str
