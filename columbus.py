import summarizer
import ta_scraper

def main(loc,query='',nTopics=3,nSentPerTopic=3)
  ta_scraper.visit_topic_pages(ta_scraper.find_forum_for_loc(loc))
  print summarize(text,query,removeQuestions=True,topNT=nTopics,topNS=nSentPerTopic,jumpProb = 0.2,inOrder=False)
  
  
