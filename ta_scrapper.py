from urllib2 import urlopen
from httplib import IncompleteRead
from bs4 import BeautifulSoup as bs
import re
#from nltk.stem.porter import PorterStemmer
import numpy as np
#import string,math
#from itertools import combinations

text = ''
baseUrl = "http://www.tripadvisor.com"
topic_count = 0
post_count = 0

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
        m = re.match('(.*)Removed',t.encode('string-escape'))
        if not m:
            post_count += 1
            text = text+'\n'+t
    
    scrape_topic_page(get_next_page(soup),False)

##f = open('t.txt','r')
##text = f.read()
##f.close()

visit_topic_pages(find_forum_for_loc("ahmedabad"))
##print "# of topics",topic_count
##print "# of Posts",post_count
#print highlight_doc(text)
f = open('t.txt','w')
f.write(text)
f.close()
