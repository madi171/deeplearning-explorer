import urllib2

url = "http://crd.opt.ifeng.com:8080/icrawlms/news_queryNewsInfoByDocid.action?docid=%s&searchType=1"
vid = "video_01b57642-fb55-4de2-8cb3-7c22774dd767"
req = urllib2.urlopen(url % vid)
print req.read()
