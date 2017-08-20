import sys
import logging
from gensim.models import Word2Vec
import urllib2

url = "http://crd.opt.ifeng.com:8080/icrawlms/news_queryNewsInfoByDocid.action?docid=%s&searchType=1"
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

input_file = "datasets/news_click_seq.out"
output = "output/news_click_seq.result"
output_w2v = output + ".vec"
output_w2v_voc = output + ".vec.voc"

id_seqs = []
print "loading..."
cnt = 0
for line in open(input_file):
    id_seq = line.strip().split('\t')
    id_seqs.append(id_seq)
    if cnt % 10000 == 0:
        print " load %d" % cnt
    cnt += 1

print "load complete"

# model = Word2Vec(id_seqs, size=64, window=4, min_count=1, workers=8, hs=0, iter=80, )
model = Word2Vec(id_seqs, size=128, window=5, min_count=2, workers=16, hs=0, iter=160, )
# best param size=128, window=5, min_count=2, workers=16, hs=0, iter=100
# model.save(output)
# model.save_word2vec_format(output_w2v, fvocab=output_w2v_voc)

import json


def get_title(vid):
    ret = urllib2.urlopen(url % vid).read()
    try:
        obj = json.loads(ret, 'utf-8')
        return obj['title']
    except:
        return ""


import random

for i in xrange(1, 50):
    sent_id = random.randint(1, len(id_seqs))
    target_vid = id_seqs[sent_id][0]
    target_title = get_title(target_vid)
    lens = len(target_title)
    if lens > 350 or lens < 14:
        continue
    print "# %d target news = %s, %s" % (i, target_title, target_vid)
    target_vec = model[target_vid]
    sim_list = model.most_similar(target_vid)
    # print sim_list
    for ss in sim_list:
        vid = ss[0]
        title = get_title(vid)
        print "  %d sim %.2f news = %s, %s" % (i, ss[1], title, vid)
    print "\n\n"
