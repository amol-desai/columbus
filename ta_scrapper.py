def find_forum_for_loc(searchStr):
    searchStr = "_".join(searchStr.split())
    searchUrl = baseUrl+"/Search?q="+searchStr

    pg = urlopen(searchUrl).read()
    soup = bs(pg)

    forumLink = soup.findAll("a",text="Forums")[0]
    forumLink = baseUrl+forumLink.get("href")

    return forumLink #main forum link

def visit_topic_pages(forumLink):
    if not forumLink:
        return
    
    global topic_count
    topic_count += 10

    try:
        pg = urlopen(forumLink).read()
    except IncompleteRead,e:
        pg = e.partial
        
    soup = bs(pg)

    #topics = soup.findAll(href=re.compile("/ShowTopic-.*"))
    #gives two instances of the same topic
    topics = soup.findAll("b")
    for topic in topics:
        t = topic.find(href=re.compile("/ShowTopic-.*"))
        if t:
            topicLink = baseUrl+t.get("href")
            scrape_topic_page(topicLink,True)
            #debug by looking at just the first topic
            #break

    visit_topic_pages(get_next_page(soup))

def get_next_page(soup):
    nextPg = soup.find("a",{'class':'guiArw sprite-pageNext'})
    if nextPg:
        nextPg = baseUrl+nextPg.get("href")
    return nextPg

def scrape_topic_page(topicLink,isFirstPage):
    if not topicLink:
        return
    
    global text
    global post_count

    try:
        pg = urlopen(topicLink).read()
    except IncompleteRead, e:
        pg = e.partial
    
    soup = bs(pg)
    
    posts = soup.findAll("div",{'class':'postBody'})
    if not isFirstPage:
        posts.remove(posts[0])
    for p in posts:
        t = p.get_text().encode('utf8')
        m = re.match('(.*)Removed',t)
        if not m:
            post_count += 1
            text = text+re.sub('\n+',' ',re.sub(r'([a-z|A-Z|0-9]) *\n',r'\1. ',t))
    
    scrape_topic_page(get_next_page(soup),False)
