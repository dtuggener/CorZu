# -*- coding: utf-8 -*- 

"""
Convert CoNLL output of parzu/corzu to HTML
"""

import sys, re, random, colorsys

def gen_color(colors):
    """ Generate color hex """
    while True:
        colorcode=random.randint(0, 16777215)
        color='#%x'%colorcode
        if not color in colors: 
            return color

# Print HTML header and stuff
print '<meta http-equiv="Content-Type" content="text/html; charset=utf-8">'
print '<html> \
<head> \
<style> \
html * \
{\
   font-size: 1em; \
   font-family: Arial, Helvetica, sans-serif; \
   line-height: 150%;\
}\
</style>\
</head>\
<body>  \
<p>'
print '<font family="Arial, Helvetica, sans-serif">'


# Read and prepare the input
dok=open(sys.argv[1],'r').read()
lines=dok.split('\n')
ids=[]
coref={}    # colors dict
coref_ids=[]
sent=0

# Collect all coreference ids
for line in lines:
    if not line=='':
        if not line.strip().endswith('_') and not line.strip().endswith('-'):
            l=line.split('\t')
            ms=l[-1].split('|')
            for m in ms:
                if not m=='': ids.append(re.search('\d+',m).group())

for line in lines:
    
    if line=='\n' or line =='\t\n' or line=='':
        print '<br>'
        sent+=1

    else:
        line=re.split('\t| +',line)
        
        # No coreference
        if line[-1].strip()=='_' or line[-1].strip()=='-': 
            print line[1],
            
        else:
            coref_start=re.findall('\(\d+',line[-1])
            coref_start.reverse()
            for id in coref_start:
                idint=re.search('\d+',id).group()
                if len([x for x in ids if x==idint])>1: # no singeltons
                    coref_ids.append(id)
                    if coref.has_key(id):   # open cset
                        print '<font color="'+coref[id]+'"><sup>'+id.replace('(','[<')+'</sup>',
                    else:                   # new cset
                        color=gen_color(coref.values()) #generate random color
                        print '<font color="'+color+'"><sup>'+id.replace('(','[>')+'</sup>',
                        coref[id]=color
            # Insert word ID here?
            print line[1],
            coref_end=re.findall('\d+\)',line[-1])
            for id in coref_end:
                idint=re.search('\d+',id).group()
                if len([x for x in ids if x==idint])>1:
                    last_id=coref_ids.pop()
                    print '<sup>]</sup></font>',
print '</p></body></html>'        
