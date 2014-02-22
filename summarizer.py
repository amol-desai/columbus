def s_tokenize(text):
    """
    Args:
        text - Document to be tokenized (string)

    Returns:
        An array tokenized by sentences
        Another array where each sentence is tokenized the words in the sentence
    """
    temp = re.split('(\.|!|\?|\n)',text)
    p = PorterStemmer()

    sentences = []
    words = []
    #create the two arrays to be returned. Stem the words that are tokenized
    for i,item in enumerate(temp):
        if re.match('\.|!|\?|\n',item):
            if i != 0 and len(temp[i-1])>1:
                sentences.append(temp[i-1]+item)
        else:
            if len(item) > 1:
                if i==len(temp)-1:
                    sentences.append(item)
                temp1 = re.sub('[%s]' %re.escape(string.punctuation),
                               ' ',item.lower()).split()
                temp2 = []
                if len(temp1)>0:
                    for word in temp1:
                        temp2.append(p.stem(word))
                    words.append(temp2)
    return sentences, words

def tfidf(words):
    """
    Args:
        words - Array of sentences where each sentence is tokenized by word i.e. Array or arrays

    Returns:
        TFIDF matrix of sentences x words in array of dictionary format (Array of dict)
    """
    #term frequency. This is the data structure that will eventually be updated to be tfidf and returned
    tf = []
    #document frequency
    df = {}

    #populate tf for each sentence and count of sentences containing term in idf
    for i,sentence in enumerate(words):
        tf.append({})
        for w in sentence:
            curr_w = ''
            if w not in tf[i]:
                tf[i][w] = 1
            else:
                tf[i][w] += 1
            if w not in df:
                curr_w = w
                df[w] = 1
            elif w != curr_w:
                df[w] += 1
    
    #compute tfidf and do not normalize by length of the sentence
    for sentence in tf:
        for key in sentence.keys():
            sentence[key] = sentence[key]*math.log10(float(len(words))/df[key])

    return tf

def compute_similarity(tfidf):
    """
    Args:
        tfidf - Array of dictionaries that provide a TFIDF matrix

    Returns:
        Similarity matrix of sentences x sentences obtained by using cosine similarity
    """
    
    similarity = np.array(np.zeros((len(tfidf),len(tfidf))))

    #create inverted dict of word to sentence mapping to
    #avoid going through sentence dict multiple times
    inv_dict = {}
    for i,sentence in enumerate(tfidf):
        for word in sentence:
            if word in inv_dict:
                inv_dict[word].append(i)
            else:
                inv_dict[word] = [i]

    #compute the cosine similariity and populate the similarity matrix                
    for word in inv_dict:
        for i,j in combinations(inv_dict[word],2):
            similarity[i,j] = similarity[i,j]+((tfidf[i][word]*tfidf[j][word])/(math.sqrt(np.sum(np.array(tfidf[i].values())**2))*math.sqrt(np.sum(np.array(tfidf[j].values())**2))))
            similarity[j,i] = similarity[i,j]

    return similarity

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
    
    sentences,words_per_sentence = s_tokenize(doc)
    term_weighting = tfidf(words_per_sentence)
    similarity_matrix = compute_similarity(term_weighting)
    pg_rank = pagerank(similarity_matrix,0.8)
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
    return toret_str
