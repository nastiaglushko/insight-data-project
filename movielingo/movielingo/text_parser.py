import sys
sys.path.append("../movielingo/")

from nltk.tokenize import RegexpTokenizer

TOKENIZER = RegexpTokenizer(r'\w+')

def division(x,y):
    if float(x)==0 or float(y)==0:
        return 0
    return float(x)/float(y)

#sentence (S)
s="'ROOT'"

#verb phrase (VP)
vp="'VP > S|SINV|SQ'"
vp_q="'MD|VBZ|VBP|VBD > (SQ !< VP)'"

#clause (C)
c="'S|SINV|SQ [> ROOT <, (VP <# VB) | <# MD|VBZ|VBP|VBD | < (VP [<# MD|VBP|VBZ|VBD | < CC < (VP <# MD|VBP|VBZ|VBD)])]'"

#T-unit (T)
t="'S|SBARQ|SINV|SQ > ROOT | [$-- S|SBARQ|SINV|SQ !>> SBAR|VP]'"

#dependent clause (DC)
dc="'SBAR < (S|SINV|SQ [> ROOT <, (VP <# VB) | <# MD|VBZ|VBP|VBD | < (VP [<# MD|VBP|VBZ|VBD | < CC < (VP <# MD|VBP|VBZ|VBD)])])'"

#complex T-unit (CT)
ct="'S|SBARQ|SINV|SQ [> ROOT | [$-- S|SBARQ|SINV|SQ !>> SBAR|VP]] << (SBAR < (S|SINV|SQ [> ROOT <, (VP <# VB) | <# MD|VBZ|VBP|VBD | < (VP [<# MD|VBP|VBZ|VBD | < CC < (VP <# MD|VBP|VBZ|VBD)])]))'"

#coordinate phrase (CP)
cp="'ADJP|ADVP|NP|VP < CC'"

#complex nominal (CN)
cn1="'NP !> NP [<< JJ|POS|PP|S|VBG | << (NP $++ NP !$+ CC)]'"
cn2="'SBAR [<# WHNP | <# (IN < That|that|For|for) | <, S] & [$+ VP | > VP]'"
cn3="'S < (VP <# VBG|TO) $+ VP'"

#fragment clause
fc="'FRAG > ROOT !<< (S|SINV|SQ [> ROOT <, (VP <# VB) | <# MD|VBZ|VBP|VBD | < (VP [<# MD|VBP|VBZ|VBD | < CC < (VP <# MD|VBP|VBZ|VBD)])])'"

#fragment T-unit
ft="'FRAG > ROOT !<< (S|SBARQ|SINV|SQ > ROOT | [$-- S|SBARQ|SINV|SQ !>> SBAR|VP])'"

#list of patterns to search for
patternlist=[s,vp,c,t,dc,ct,cp,cn1,cn2,cn3,fc,ft,vp_q]

from nltk.parse import CoreNLPParser
TREGEX = "../../tregex"
TEMP = "./"
import os
import subprocess

raw_text = "I put the book in the box on the table where it belonged and it was fine.\nI put the book in the box on the table where it belonged and it was fine."
# output = '2455' # should be text_id
parser = CoreNLPParser()
parsed = parser.parse_text(raw_text)

#print(str(next(parsed)))

f = open("temp.parsed", "w")
for elem in parsed:
    f.write(str(elem))
f.close()

patterncount = []
for pattern in patternlist:
    command = TREGEX + "/tregex.sh "  + pattern + " " + TEMP+ "/temp.parsed -C -o"  
    direct_output = subprocess.check_output(command, shell=True)
    count = direct_output.decode('utf-8')[:-1]
    patterncount.append(count)

print(patterncount)

patterncount[7]=patterncount[-4]+patterncount[-5]+patterncount[-6]
patterncount[2]=patterncount[2]+patterncount[-3]
patterncount[3]=patterncount[3]+patterncount[-2]
patterncount[1]=patterncount[1]+patterncount[-1]

w = len(TOKENIZER.tokenize(raw_text))
#add frequencies of words and other structures to output string
# output+=","+str(w)
# for count in patterncount[:8]:
#     output+=","+str(count)
    
#list of frequencies of structures other than words
[s,vp,c,t,dc,ct,cp,cn]=patterncount[:8]

#compute the 14 syntactic complexity indices
mls=division(w,s)
mlt=division(w,t)
mlc=division(w,c)
c_s=division(c,s)
vp_t=division(vp,t)
c_t=division(c,t)
dc_c=division(dc,c)
dc_t=division(dc,t)
t_s=division(t,s)
ct_t=division(ct,t)
cp_t=division(cp,t)
cp_c=division(cp,c)
cn_t=division(cn,t)
cn_c=division(cn,c)

ratio_output = [mls,mlt,mlc,c_s,vp_t,c_t,dc_c,dc_t,t_s,ct_t,cp_t,cp_c,cn_t,cn_c]
print(ratio_output)
from nltk.tokenize import sent_tokenize

def reformat_sentences_for_parser(raw_text):
    new_lines = [x + '\n' for x in sent_tokenize(raw_text)]
    formatted_text = ' '.join(new_lines)
    return formatted_text

