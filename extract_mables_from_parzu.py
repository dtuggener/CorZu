# -*- coding: utf-8 -*-

"""
Extract markables from ParZu output (CoNLL format)
Perform string matching (we do it here, not in the coreference resolver)
Example usage: python extract_mables_from_conll.py examples/parsed.conll > examples/markables.txt
"""

import os,re,sys,pdb,cPickle,operator,copy
from collections import defaultdict
from get_subcat_frame import *

global align_gold_boundaries, real_preprocessing
align_gold_boundaries=False                             #align extracted markables to gold mention boundaries
real_preprocessing=True                                 #switch for gold morphology

# make script work from arbitrary directory
corzu_dir = os.path.dirname(os.path.realpath(__file__))

""" functions """
def nn_str_matching(ante,anaph):    
    """ string matching for common nouns; return True/False """
    if len(anaph[-3])==1: return 0                      #don't resolve bare NPs
    #if ante[1]==anaph[1] or anaph[1]-ante[1]>10: return 0                      #not in the same sentence, not more than 10 sentences away
    if ante[6]!=anaph[6]: return 0                      #morph match: gender
    if ante[7]!=anaph[7]: return 0                      #morph match: number
    #if len(ante[-3])==1 and len(anaph[-3])==1 and ante[7]=='PL' and ante[9]==anaph[9]: return 1    # allow plural NPs of length one to match; no substring match
    #if len(ante[-3])==1 and len(anaph[-3])==1: return 1    #no modifier in anaphor and antecedent: ok, plurals mostly, might be tricky though...    
    #if len(ante[-3])>1 and ante[-3][0].startswith('all'): return 0  
    if len(anaph[-3])>1 and anaph[-3][0].lower().startswith('all'): return 0    #all-quantified NPs can't be anaphoric
    if len(anaph[-3])>1 and anaph[-3][0].lower().startswith('ein'): return 0    #indefinite NPs can't be anaphoric
    if len(anaph[-3])>1 and anaph[-3][0].lower().startswith('ander'): return 0    #indefinite NPs can't be anaphoric
    if ante[-3][0].lower().startswith('kein') or anaph[-3][0].lower().startswith('kein'): return 0 #negative quantification
    #if ante[-3][0].lower().startswith('einig') or anaph[-3][0].lower().startswith('einig'): return 0 #fuzzy quantification        
    #if ante[-3][0].lower().startswith('manche') or anaph[-3][0].lower().startswith('manche'): return 0 #fuzzy quantification
    if ante[-3]==anaph[-3]: return 1                    #full match

    #"""
    #longer NPs
    if len(ante[-3])>1 or len(anaph[-3])>1:

        if not ante[9] in ante[-3]: #conjunctions
            if ante[9]==anaph[9]:
                return 1
            else:
                return 0

        # both NPs have determiners
        if (ante[1],ante[2]) in determiners and not determiners[ante[1],ante[2]]=='*' and (anaph[1],anaph[2]) in determiners and not determiners[anaph[1],anaph[2]]=='*':
            if ante[9]==ante[-3][-1] and anaph[9]==anaph[-3][-1]:    #matches are the head, i.e. at end of string, no postmodifiers
                #explicitly allow "ein.* .. NN" -> "die.* .. NN" pattern, disregarding modifiers -> too lax? #TODO: Sentence distance limit?
                if determiners[ante[1],ante[2]].startswith('ein') and re.match('[DdZz](er|ie.*|as|em|es|u.*)|[Bb]eide.*',determiners[anaph[1],anaph[2]]):
                    return 1     
                if determiners[ante[1],ante[2]].startswith('dies') and re.match('[DdZz](er|ie.*|as|em|es|u.*)|[Bb]eide.*',determiners[anaph[1],anaph[2]]):
                    return 1                         
                #ante is modified; everything between determiner and head matches, i.e. everything in the anaph str is in the ante str
                if len(ante[-3])>2 and [x for x in anaph[-3][1:] if not x==anaph[9] and not x in ante[-3][1:]]==[]:
                    return 1
            """
            elif determiners[anaph[1],anaph[2]].startswith('dies') and anaph[1]-ante[1]<4:
                pdb.set_trace()
                return 0
            """
            
        #if the ante head is not the end of the string (i.e. there are postmodifiers), at least ~60% of the anaph tokens must overlap
        else:
            if len(ante[-3]) > 1:
                if (ante[1],ante[2]) in determiners and not determiners[ante[1],ante[2]]=='*':  #remove determiners
                    ante_str=ante[-3][1:]
                else:
                    ante_str=ante[-3]
            else: ante_str = ante[-3]
            if len(anaph[-3]) > 1:
                if (anaph[1],anaph[2]) in determiners and not determiners[anaph[1],anaph[2]]=='*':
                    anaph_str=anaph[-3][1:]
                else:
                    anaph_str=anaph[-3]      
            else: anaph_str = anaph[-3]
            #60% of the mentions in must be shared; TODO: reduce this to ADJD, ADJA, NE, NN? Match numbers
            if len([x for x in anaph_str if not x==anaph[9] and x in ante_str])/float(len(anaph_str))>.6 or len([x for x in ante_str if not x==ante[9] and x in anaph_str])/float(len(ante_str))>.6:
                return 1
                
        #ante is modified, anaph isn't
        if (anaph[1],anaph[2]) in determiners and not determiners[anaph[1],anaph[2]]=='*' and not determiners[anaph[1],anaph[2]]=='CARD':  #remove determiners
            anaph_str=anaph[-3][1:]
        else:
            anaph_str=anaph[-3]  
        if len(anaph_str)==1: 
            return 1      

def str_match(mables):
    """ string matching for nouns; return a list of lists containig string matching noun markables """
    orig_mables=copy.deepcopy(mables)    
    matches=[]
    while not mables==[]:
        mable=mables[0]                                     #first markable in markables
        match=[]

        if mable[4]=='NE':                                  #named entitiy
            #match=[np for np in mables if np[9]==mable[9] and np[0]>mable[0] and np[4]=='NE'] #named entity: head matching and succeeding
            match=[]
            for np in mables:
                if np[0]>mable[0] and np[9]==mable[9] and np[4]=='NE':
                    """
                    if not np[6]==mable[6] or not np[7]==mable[7]: 
                        if [x for x in np[6:8] if x=='*']==[] and [x for x in mable[6:8] if x=='*']==[]:
                            pdb.set_trace()
                    """
                    if len(np[12])==1:
                        match.append(np)
                    else:
                        if mable[-1]=='PER':    #for person entities we require the gender to match
                            if np[6]==mable[6]:
                                match.append(np)
                            else:
                                #check if we have first names and if they are the same
                                try:
                                    name_mable=mable[12][mable[12].index(mable[9])-1]
                                    name_np=np[12][np[12].index(np[9])-1]
                                    if (name_mable in female_names or name_mable in male_names) and (name_np in female_names or name_np in male_names):
                                        if name_np==name_mable:
                                            match.append(np)
                                except:
                                    match.append(np)
                        else:
                            match.append(np)
            if (mable[1],mable[2]) in nominal_mods: #match NEs to re-occurring nominal modifiers
                for n in nominal_mods[mable[1],mable[2]]:
                    for np in mables:
                        if np[4]=='NN' and np[1]>mable[1] and n.lower().endswith(np[9].lower()) and mable[7]==np[7]:  #EU-Umwelkommissarin -> Kommissarin
                            if (np[1],np[2]) in determiners and not determiners[np[1],np[2]] in ['eine','ein','*']:
                                match.append(np)
                    
        elif mable[4]=='NN':                                #common noun, require more specific matching
            for np in mables:
                #if np[0]<mable[0]: #use this with reversed order
                if np[0]>mable[0]:
 
                    if np[9]==mable[9] and nn_str_matching(mable,np):
                        match.append(np)  
                               
                    #Partial Matching, de-hyphenate                        
                    elif len(np[9])<len(mable[9]) and '-' in mable[9] and re.sub('.*-','',mable[9]).lower().endswith(np[9].lower()) and nn_str_matching(mable,np):
                        match.append(np)                                 
                    elif len(np[9])>len(mable[9]) and '-' in np[9] and re.sub('.*-','',np[9]).lower().endswith(mable[9].lower()) and nn_str_matching(mable,np):
                        match.append(np)                        
                    #Look for matching nominal modifiers 
                    elif (mable[1],mable[2]) in nominal_mods and np[9].isupper() and np[4] in ['NE','NN']:
                        for n in nominal_mods[mable[1],mable[2]]:
                            if n.lower().endswith(np[9].lower()) or np[9].lower().endswith(n.lower()):   #EU-Umwelkommissarin -> Kommissarin
                                if (np[1],np[2]) in determiners:
                                    if determiners[np[1],np[2]] not in ['ein','eine']:
                                        match.append(np)  
                                        break
                                else:
                                    match.append(np)     
                                    break

                    #TODO: only when one of them is decompoundable; i.e. not Umwelt, but Hauptpumpwerk <-> Pumpwerk                                            
                    """                        
                    #Partial Matching, string share same ending. Problem: Umwelt <-> Welt; too fuzzy
                    elif len(np[9])<len(mable[9]) and mable[9].lower().endswith(np[9].lower()) and nn_str_matching(mable,np):
                        match.append(np)                         
                    elif len(np[9])>len(mable[9]) and np[9].lower().endswith(mable[9].lower()) and nn_str_matching(mable,np):
                        match.append(np)  
                    """                        
                    
                    #TODO: if mable[9] is all uppercase, assume abbreviation, look for decompoundable noun
                    # VS = Verfassungs-Schutz = Verfassungsschutz
                    """
                    elif mable[9].isupper() and np[9].startswith(mable[9][0]):
                        decomp=compsplit.split_compound(np[0])
                        try: 
                            next([x for x in if x[0]>0 and x[2].startswith(mable[9][-1])])  # Schutz
                            match.append(np)
                        except StopIteration:
                            pass                        
                    """     
                               
            #Demonstrative common noun NPs                                
            if match==[] and (mable[1],mable[2]) in determiners and determiners[mable[1],mable[2]].startswith('dies'):
                try:
                    #ante=next(m for m in reversed(orig_mables) if m[1]<mable[1] and m[9]==mable[9] and mable[1]-m[1]<4)
                    ante=next(m for m in reversed(orig_mables) if m[1]<mable[1] and m[9].lower().endswith(mable[9].lower()) and mable[1]-m[1]<4 and m[7]==mable[7])
                    try:
                        cset=next(c for c in matches if ante in c)  #ante is in string match cset
                        cset.append(mable)
                        cset.sort()
                    except StopIteration:
                        matches.append([ante,mable])    #add new cset
                except StopIteration:
                    pass                                      
                        
        if len(match)>0:                                    #we have matches
            match.insert(0,mable)                           #insert the markable at the beginning of the match list
            for np in match: 
                if np in mables:                            #very rare case where nominal descriptor and NE have the same head lemma: Der Richter Klaus Richter
                    mables.remove(np)                       #remove the matched markables, don't process them again
            match.sort()
            matches.append(match)
        else: 
            mables.remove(mable)                            #remove processed markable if no matches are found        
    return matches


#extract token morphology; return [person,genus,numerus]
if real_preprocessing:  #Parzu parsed file
    def get_morph(tok):
        morph=[]
        if tok[4] in ['NN','NE','PRELS','PRELAT','PDS']:
            if tok[5]=='_': return [3,'*','*']
            morph=[3]                                       #person
            morph_in=tok[5].split('|')
            morph.append(morph_in[0])                       #gender
            morph.append(morph_in[2])                       #number
        elif tok[4]=='PPOSAT':                              #manually set morphology of possessive pronouns as their morph. match their heads                            
            if re.match('mein.*',tok[2].lower()): morph=[1,'*','Sg']
            elif re.match('uns.*',tok[2].lower()): morph=[1,'*','Pl']
            elif re.match('dein.*',tok[2].lower()): morph=[2,'*','Sg']
            elif re.match('euer.*',tok[2].lower()): morph=[2,'*','Pl']
            elif re.match('sein.*',tok[2].lower()): morph=[3,'*','Sg']
            elif re.match('ihr.*',tok[2].lower()): morph=[3,'*','*']
            else: morph=[3,'*','*']                         #we shouldn't get here, the above should cover everything
        elif tok[4]=='PPER':
            if tok[2].lower()=='es': morph=[3,'Neut','Sg']  #we extract it (but don't resolve it)
            elif tok[2].lower()=='er': morph=[3,'Masc','Sg']#to be sure, we do it manually; sometimes the parser outputs faulty morphology
            elif tok[1] in ['Sie','Ihnen'] and int(tok[0])>1: morph=[2,'*','*'] #Uppercase "Sie" und "Ihnen" not at beginning of sentence is 2nd person
            elif tok[5]=='_': return [3,'*','*']
            #elif tok[2].lower()=='sie': return [3,'*','*']          #leave it underspecified
            else:
                morph_in=tok[5].split('|')
                if not re.match('\d',morph_in[0]):morph.append(3)         #person
                else: morph.append(int(morph_in[0]))
                if not morph_in[2] in ['Fem','Masc','Neut','_']: morph.append('*')
                else: morph.append(morph_in[2])                   #gender
                if not morph_in[1] in ['Pl','Sg','_']: morph.append('*')
                else: morph.append(morph_in[1])                   #number
        morph=['*' if m=='_' else m for m in morph]         #replace '_' with '*' for unspecified values
        return morph
else:   #tuebadz extracted conll file
    def get_morph(tok):
        morph=[]
        if tok[4] in ['NN','NE','PRELS','PRELAT','PDS']:
            morph=[3]
            morph.append(re.search('(.)$',tok[5]).group(1))  #genus
            morph.append(re.search('(.).$',tok[5]).group(1))  #number
        elif tok[4]=='PRF': morph=[3,'*','*']
        elif tok[4]=='PPOSAT':                            
            if re.match('mein.*',tok[2].lower()): morph=[1,'*','s']
            elif re.match('uns.*',tok[2].lower()): morph=[1,'*','p']
            elif re.match('dein.*',tok[2].lower()): morph=[2,'*','s']
            elif re.match('euer.*',tok[2].lower()): morph=[2,'*','p']
            elif re.match('sein.*',tok[2].lower()): morph=[3,'*','s']
            elif re.match('ihr.*',tok[2].lower()): morph=[3,'*','*']
            else: morph=[3,'*','*']            
        elif tok[4]=='PPER':
            if tok[2]=='es': morph=[3,'n','s']
            else:
                morph.append(int(re.search('(.)$',tok[5]).group(1)))   #person
                morph.append(re.search('(.).$',tok[5]).group(1))  #genus
                morph.append(re.search('(.)..$',tok[5]).group(1))  #number
        return morph

#search markable extension, i.e. traverse parse tree recursively; return list of tokens (daughters)
def get_extension(head,token,sent,ext):
    for m in sent:
        if m[6]==token[0]:
            if m[4] in ['PRELS','PRELAT','PWAV','KOUS','PROAV','ADJD','KOKOM'] and int(m[0])>int(head[0]):
                return ext
            if m[7] in ['-unknown-','par']:
                return ext
            if m[7]=='kon' and not m[4]=='KON':
                return ext
            if not m[1]=='"' and not m[4].startswith('V') and not m[4]=='KON':
                ext.append(m)
            get_extension(head,m,sent,ext)        
            """
            if m[7]=='kon' and not m[0]=='1':   #only allow coordination at sentence beginning
                get_extension(m,sent,ext)
            elif m[4].startswith('V'): 
                get_extension(m,sent,ext)
            else: 
                if not m[1]=='"': #and not m[4]=='ADV':   #TODO:allow APPR? Test what works better
                    ext.append(m)
                get_extension(m,sent,ext)
            """                
    return ext


verbs=defaultdict(dict)
all_verbs={}
haben={}
gmods={}
preds={}

""" main """
#we process line by line, aggregate all tokens of a sentence and then extract the markables
sentence=[]                                             #list to which tokens from a sentence are appended to
mables=[]                                               #list of extracted markables
sent_nr=1                                               #sentence counter

"""
path=os.path.dirname(sys.argv[0])
if path.startswith('..'):
    path+='/'
elif not path=='':
    path='/'+path
person=eval(open(path+'data/mensch.txt','r').read())  #Person descriptions extracted from Germanet 7 nomen.Mensch.xml
male_names=eval(open(path+'data/male_names.txt','r').read())      #male first names, used for gender disambiguation of named entities
female_names=eval(open(path+'data/female_names.txt','r').read())  #female first names
"""

if os.path.isfile(corzu_dir + os.sep + 'mensch.txt'): person=eval(open(corzu_dir + os.sep + 'mensch.txt','r').read())  #Person descriptions extracted from Germanet 7 nomen.Mensch.xml
else: 
    print >> sys.stderr,'Not using mensch.txt; consider using it for improved pronoun resolution performance (see README).'
    person=[]
male_names=eval(open(corzu_dir + os.sep + 'male_names.txt','r').read())      #male first names, used for gender disambiguation of named entities
female_names=eval(open(corzu_dir + os.sep + 'female_names.txt','r').read())  #female first names

doc_counter=0

sentences={}
sentence=[]
mables=[]
koords=[]
coref={}
aggr=[]
prepositions={}
pposat_heads={}
nominal_mods=defaultdict(list)
verbs=defaultdict(dict)
all_verbs={}
preds={}
haben={}
gmods={}
determiners={}

for line in open(sys.argv[1],'r').readlines():
        
    if line=='\n' or line=='\t\t\t\t\t\t\t\t\t\n':    #newline is sentence boundary, start processing the aggregated sentence
    
        if not sentence==[]:
            sentences[str(sent_nr)]=sentence    
        for tok in sentence:
        
            #find predicatives: "A is a B" etc.
            if tok[4] in ['NN'] and tok[7]=='pred':
                try:
                    v_gov=next(t for t in sentence if t[4].startswith('V') and t[0]==tok[6] and t[2]=='sein')
                    n_head=next(t for t in sentence if t[4] in ['NE','PPER'] and t[6]==v_gov[0] and int(t[0])<int(tok[0]) and not t[2]=='es')                   
                    matching_mable=next(m for m in reversed(mables) if m[0]==sent_nr and int(n_head[0]) in range(int(m[1]),int(m[2])+1))    #find the matching mable
                    nominal_mods[tuple(matching_mable[:2])].append(tok[2])  #store the predication as a nominal_mod
                except StopIteration: True
        
            if tok[4] in ['PPER','PRELS','PRELAT','PPOSAT','PDS'] and not tok[2] in ['es','was']:     #pronouns                     
                mable=[sent_nr,int(tok[0]),int(tok[0])]           #sentence number, markable extension start token, markable extension end token
                mable.append(tok[4])                    #PoS-tag
                ext=[tok]                               #extension, all tokens in the markable. here it, is only one token.
                morph=get_morph(tok)                    #morphological features
                mable+=morph
                if tok[7]=='cj':    #Konjunktionen: GF ersetzen durch die des Kopfs
                    try:
                        konj=next(t for t in sentence if t[0]==tok[6])
                        head=next(t for t in sentence if t[0]==konj[6])
                        tok[7]=head[7]
                    except StopIteration: True                
                if tok[7].upper()=='PN':
                    if sentence[int(tok[6])-1][7].upper()=='OBJP':
                        mable.append('OBJP')
                    else:
                        mable.append(tok[7].upper())            #gram. function 
                else:
                    mable.append(tok[7].upper())            #gram. function 
                mable.append(tok[2])                    #lemma
           
                try:
                    gov,mode=get_gov(tok,sentence)               #(full) verb governing the token, returns verb token id and lemma
                    if mable[3]=='PPOSAT' and not gov is None:
                        mable.append(int(gov[0]))
                        mable.append(gov[2].replace('#','').replace('-',''))
                    elif not gov is None and not mable[3]=='PPOSAT':
                        mable.append(int(gov[0]))
                        mable.append(gov[2].replace('#','').replace('-',''))
                        if not mable[7]=='OBJP': 
                            if mode=='passive' and mable[7]=='SUBJ':                       
                                mable[7]='OBJA'
                    verbs[sent_nr,int(gov[0])][mable[7].lower()]=tok
                except TypeError:
                    mable.append(0)
                    mable.append('*')   
                    
                mable.append([tok[2]])                  #full markable string
                conn='noconn'                           #check wheter the markable is preceded by a discourse connective
                if not ext[0][0]==1:                    #not the first token of a sentence
                    for i in range(sentence.index(ext[0])-1,-1,-1):   #look backwards
                        if sentence[i][4]=='$,': break  #don't cross commas
                        elif sentence[i][7] in ['subj','obja','objd'] or sentence[i][4].startswith('V'): break #don't cross these GFs
                        elif sentence[i][4]=='KOUS' and not sentence[i][0]==1: 
                            conn='conn'
                            break
                mable.append(conn)
                mable.append('-')  #NE type
                if tok[4]=='PPOSAT':
                    try:
                        pposat_head=next(t for t in sentence if t[0]==tok[6])
                        pposat_heads[mable[0],mable[1]]=pposat_head
                    except StopIteration: True            
                
                doit=True
                if tok[4]=='PDS':
                    #criterion for extracting PDS: either masculine, feminine, or plural. Not *jenige* and *jene*                
                    if tok[1].lower()=='dessen' or 'jene' in tok[1].lower() or 'jenige' in tok[1].lower() or 'all' in tok[1].lower():
                        doit=False
                    if real_preprocessing:
                        if not tok[5].endswith('Pl') and not tok[5].endswith('_') and not tok[5].startswith('Fem') and not tok[5].startswith('Masc'):
                            doit=False
                    else:
                        if not tok[5].endswith('*') and not tok[5].endswith('m') and not tok[5].endswith('f'):
                            doit=False    
                if tok[7].upper()=='PN':
                    prepositions[sent_nr,int(tok[0])]=sentence[int(tok[6])-1][2]                                            
                if doit:                        
                    mables.append(mable)                    #append markable to the list of markables
                #determiners[(mable[0],mable[1])]='*'                

            elif tok[4] in ['NN','NE']:                 #nouns
                """
                Apposition handling: 
                1. if the preceding markable is a named entity, shift the head to the current token
                [Lothar] Koring -> Lothar [Koring]
                2. if the preceding markable is an apposition, shift the head to the current token
                Landesvorsitzende [Ute] Wedemeier -> Landesvorsitzende Ute [Wedemeier]
                3. the preceding markable must be the immediate predecessor of the current token
                Problemtic: [Staatsanwaltschaft] Bremen -> Staatsanwaltschaft [Bremen]
                -> Only do it if the apposition is a NE with NER tag PER?
                """
                if tok[7]=='app' and not mables==[]:      
                    head=sentence[int(tok[6])-1]
                    try:
                        head_mable=next(m for m in reversed(mables) if sent_nr==m[0] and int(head[0]) in range(m[1],m[2]+1))
                        if tok[4]=='NE':    #Die Kanzlerin, Angela Merkel
                            if head_mable[3]=='NN': #store the nominal descriptor: Die [Kanzlerin], Angela Merkel, ... as we override it below
                                nominal_mods[tuple(head_mable[:2])].append(head_mable[8]) 
                            #else:                                                                
                            head_mable[8]=tok[2]            #shift the head lemma
                            head_mable[3]=tok[4]            #override PoS-tag        
                            head_mable[-1]=tok[-2]          #NE tag
                            if real_preprocessing==False:
                                if not tok[5][-1]=='*' and head_mable[5]=='*' and tok[5][1]==head_mable[6]:   #gender match?                                 
                                    head_mable[5]=tok[5][-1]
                                    
                        elif tok[4]=='NN' and head_mable[3]=='NE':# and head_mable[-1] in ['PER','ORG']:    #Angela Merkel, die Kanzlerin
                            nominal_mods[tuple(head_mable[:2])].append(tok[2])  #store the nominal descriptor: Angela Merkel, die [Kanzlerin], ...   
                        elif tok[4]=='NN' and head_mable[3]=='NN' and tok[2].isupper() and int(tok[0])<len(sentence) and sentence[int(tok[0])][1]==')':   # Umweltministerium (BMU)
                            nominal_mods[tuple(head_mable[:2])].append(tok[2])
                            
                        """
                        if tok[4]=='NE':    #Die Kanzlerin, Angela Merkel
                            if tok[-2] in ['PER','ORG']:    #or only !='LOC' ?
                                if head_mable[3]=='NN': #store the nominal descriptor: Die [Kanzlerin], Angela Merkel, ... as we override it below
                                    nominal_mods[tuple(head_mable[:2])].append(head_mable[8]) 
                                if head_mable[-1]=='PER' and not tok[-2]=='PER':    #Otto Schily (SPD) -> don't shift head to SPD
                                    pass
                                else:                                                                
                                    head_mable[8]=tok[2]            #shift the head lemma
                                    head_mable[3]=tok[4]            #override PoS-tag        
                                    head_mable[-1]=tok[-2]          #NE tag
                                    if real_preprocessing==False:
                                        if not tok[5][-1]=='*' and head_mable[5]=='*' and tok[5][1]==head_mable[6]:   #gender match?                                 
                                            head_mable[5]=tok[5][-1]
                                    
                            
                        elif tok[4]=='NN' and head_mable[3]=='NE' and head_mable[-1] in ['PER','ORG']:    #Angela Merkel, die Kanzlerin
                            nominal_mods[tuple(head_mable[:2])].append(tok[2])  #store the nominal descriptor: Angela Merkel, die [Kanzlerin], ...   


                        elif tok[4]=='NN' and head_mable[3]=='NN' and tok[2].isupper() and int(tok[0])<len(sentence) and sentence[int(tok[0])][1]==')':   # Umweltministerium (BMU)
                            nominal_mods[tuple(head_mable[:2])].append(tok[2])
                        """            
                                        
                        if int(tok[0])>head_mable[2]:
                            head_mable[2]=int(tok[0])            #expand the token extension end                            
                    except StopIteration: pass

                else:
                    ext_borders=get_extension(tok,tok,sentence,[])  #search recursively for daughter tokens
                    ext_borders.append(tok)
                    ext_borders=sorted(ext_borders, key=lambda x: int(x[0]))
                    ext=[m for m in sentence if int(m[0]) in range(int(ext_borders[0][0]),int(ext_borders[-1][0])+1)]                           
                    try:
                        #cut off conjunctions and relative sentences etc.
                        border=next(m for m in ext if m[4] in ['PRELS','PRELAT','PWAV','KOUS','PROAV','ADJD','KON'] and int(m[0])>int(tok[0]))
                        ext=ext[:ext.index(border)]
                    except StopIteration:                                                   
                        pass   
                    while ext[-1][4] in ['APPR','$,','$.','KOUS','PTKNEG'] or ext[-1][4].startswith('V') or ext[-1][2]=='-': #cut extension end
                        ext=ext[:-1]  
                    while ext[0][2]=='/' or ext[0][4]=='PTKNEG': #cut extension start
                        ext=ext[1:]                                                                
                    mable=[sent_nr,int(ext[0][0]),int(ext[-1][0])] #sentence nr, token id start, token id end             
                    mable.append(tok[4])                #PoS-tag       
                    morph=get_morph(tok)                #morphological features
                    mable+=morph
                    if tok[7].upper()=='PN':
                        if sentence[int(tok[6])-1][7].upper()=='OBJP':
                            mable.append('OBJP')
                        else:
                            mable.append(tok[7].upper())            #gram. function                             
                    else:
                        mable.append(tok[7].upper())            #gram. function 
                    mable.append(tok[2])                #lemma


                    #determine the determiner
                    try:
                        det=next(m for m in ext if m[6]==tok[0] and int(m[0])<int(tok[0]) and m[4] in ['ART','PIAT','PDAT','CARD','APPRART','PPOSAT'])
                        if det[4] in ['ART','PIAT','PDAT']:
                            determiners[(mable[0],mable[1])]=det[2]
                        elif det[4] in ['CARD','APPRART','PPOSAT']:
                            determiners[(mable[0],mable[1])]=det[4]
                        if not det in ext:pdb.set_trace()
                    except StopIteration:
                        determiners[(mable[0],mable[1])]='*'
                    #(full) verb governing the token, returns verb token id & lemma
                    try:
                        gov,mode=get_gov(tok,sentence)           
                        mable.append(int(gov[0]))
                        mable.append(gov[2].replace('#','').replace('-',''))
                        if not mable[7]=='OBJP': 
                            if mode=='passive' and mable[7]=='SUBJ':                       
                                mable[7]='OBJA'  
                        verbs[sent_nr,int(gov[0])][mable[7].lower()]=tok                                                              
                    except TypeError:   # get_gov returned None
                        mable.append(0)
                        mable.append('*')                        
                                        
                    #sort the extension on the token id, first element. it's a string
                    mable.append([m[2] for m in ext]) #markable extension string, needed for string matching
                    conn='noconn'                       #check wheter the markable is preceded by a discourse connective
                    if not ext[0][0]=='1':
                        for i in range(sentence.index(ext[0]),-1,-1):     #look backwards from mable                   
                            if sentence[i][4]=='$,': break                  #don't cross commas
                            elif sentence[i][7] in ['subj','obja','objd'] or sentence[i][4].startswith('V'): break #don't cross these GFs
                            elif sentence[i][4]=='KOUS' and not sentence[i][0]==1: 
                                conn='conn'
                                break
                    mable.append(conn)
                    mable.append(tok[-2])   #NE type
                    if tok[7].upper()=='PN':
                        prepositions[sent_nr,int(ext[0][0])]=sentence[int(tok[6])-1][2]                      
                    mables.append(mable)                #append markable to the list of markables
                    if tok[7]=='gmod':
                        try:
                            gmod_head=next(t for t in sentence if t[0]==tok[6])
                            #gmods[str(sent_nr)+'-'+tok[0]]=gmod_head
                            gmods[str(sent_nr)+'-'+str(mable[1])]=gmod_head
                        except StopIteration:
                            True          
                    #Koordinierte NPen
                    if tok[7]=='cj':
                        try:                        
                            und=next(k for k in sentence if k[2] in ['und','&'] and k[0]==tok[6])   #"und" regiert die NP
                            #maybe: if the tok is NN and sing, require determiner?
                            koord_head=next(k for k in sentence if k[0]==und[6] and k[4] in ['NN','NE'])   #the coordination head is a noun
                            if len(mables)>2:
                                if koord_head[2]==mables[-2][8] and sent_nr == mables[-2][0]:
                                    koord=copy.deepcopy(mables[-2]) #copy the coordination head                                   
                                    koord[2]=int(mable[2])   #extension
                                    koord[5]='*'    #gender
                                    koord[6]='PL'    #number
                                    #koord[8]=koord[8]+' '+und[2]+' '+str(mable[8]) #head string: Detlev und Karin
                                    koord[8]=koord[8]+' und '+str(mable[8])
                                    #koord[11].append(und[2])
                                    koord[11].append('und')
                                    koord[11]+=list(mable[11])
                                    koords.append(koord)      
                                elif koord_head[2]==mables[-3][8] and sent_nr == mables[-3][0]:    
                                    koord=copy.deepcopy(mables[-3]) #copy the coordination head
                                    koord[2]=int(mable[2])   #extension
                                    koord[5]='*'    #gender
                                    koord[6]='PL'    #number
                                    #koord[8]=koord[8]+' '+und[2]+' '+str(mable[8])    #head string: Detlev und Karin
                                    koord[8]=koord[8]+' und '+str(mable[8]) #head string: Detlev und Karin                                                    
                                    koord[11].append('und')
                                    koord[11]+=list(mable[11])
                                    koords.append(koord)      
                        except StopIteration: True                                      
      

            #coref info
            if not tok[-1].endswith('-') and not tok[-1].endswith('_'): #and not tok[4]=='PRF': #no reflexives
                for id in tok[-1].split('|'):
                    cid=int(re.search('\d+',id).group())
                    if re.match('\(\d+\)',id):
                        if coref.has_key(cid): coref[cid].append([sent_nr,int(tok[0]),int(tok[0])])
                        else: coref[cid]=[[sent_nr,int(tok[0]),int(tok[0])]]
                    elif re.match('\(\d+',id):
                        aggr.insert(0,[cid,sent_nr,int(tok[0])])
                    elif re.match('\d+\)',id):
                        for ext in aggr:
                            if ext[0]==cid:
                                aggr.remove(ext)
                                ext=ext[1:]
                                ext.append(int(tok[0]))
                                break
                        if coref.has_key(cid): coref[cid].append(ext)
                        else: coref[cid]=[ext]
                    else: pdb.set_trace()

        #if not verbs=={}:pdb.set_trace()      
        verbs=get_subcat(verbs,sentence)         
        all_verbs.update(verbs)
        verbs=defaultdict(dict)
        sentence=[]
        sent_nr+=1
        
    else:                                               #aggregate sentence tokens
        line=line.strip().split('\t')
        if not line==['']:
            sentence.append(line)
            #sentence.append([int(line[0])]+line[1:])
                    

""" output """
print 'docid= 1'
if not koords==[]:
    mables+=koords  #include coordinated nps
mables.sort()                                           #sort by sentence number and markable extension
mables2=[]                                              #final markable list (some transformation below)
mable_nr=0                                              #markable id counter
                
#some transformations in the markable feature vectors
for m in mables:
    m.insert(0,mable_nr)                                #insert markable ID
    if not real_preprocessing:
        if m[6]=='f': m[6]='FEM'
        if m[6]=='m': m[6]='MASC'
        if m[6]=='n': m[6]='NEUT'
        if m[7]=='s': m[7]='SG'
        if m[7]=='p': m[7]='PL'                        
    m2=m[:9]
    m2[6]=m2[6].upper()                                 #uppercase gender                
    m2[7]=m2[7].upper()                                 #uppercase number
    if m2[7]=='PL':
        m2[6]='*'                                      #Plural NPs don't have gender, ignore parse output
    m2.append(m[10])
    m2.append(m[11])
    m2.append(m[13])
    m2.append(m[9])
    m2.insert(11,m[-1])                  
    if m[4] =='NE' and not m[7]=='PL':                  #override gender using list of first names, but not plural, e.g. conjunctions
        for t in m[-3]:                                 #m[-3] is markable full string
            if t in male_names and t in female_names: 
                m2[6]='*'
                m2[7]='SG'
                m2[11]='PER'
                break
            elif t in male_names: 
                m2[6]='MASC'
                m2[7]='SG'
                m2[11]='PER'               
                break         
            elif t in female_names:
                m2[6]='FEM'
                m2[7]='SG'
                m2[11]='PER'
                break 
        """
        else:
            #TODO if there is a noun in the NP, assume gender is correct, if it is an NE only NP, leave it underspecified
            if real_preprocessing:
                if not (m[1],m[2]) in determiners or determiners[m[1],m[2]]=='*':
                    m2[6]='*'                                #don't take morphology of parser, leave it underspecified for NEs
        """                            
    #add animacy feature here
    if m[4] =='NE':
        if m[2]==m[3]:  #single word token
            if m[9] in male_names or m[9] in female_names or m[-1]=='PER': m2.insert(9,'ANIM')
            else: m2.insert(9,'*')
        elif m[-1]=='PER': m2.insert(9,'ANIM')  #mwt                  
        else:
            found=0
            for tok in m[-2]:
                if tok[0].isupper(): 
                    if tok in person or tok in male_names or tok in female_names:
                        m2.insert(9,'ANIM')
                        found=1
                        m2[12]='PER'
                        break
            if found==0: m2.insert(9,'*')
    elif m[4]=='NN':
        if m[9] in person: m2.insert(9,'ANIM')
        elif '-' in m[9]: 
            found=0
            lex=re.search('.*-(.*)',m[9]).group(1)
            if lex in person: 
                m2.insert(9,'ANIM')
                found=1
            if found==0: m2.insert(9,'*')
        elif '|' in m[9]: 
            found=0
            lex=re.search('(.*?)\|',m[9]).group(1)
            if lex in person: 
                m2.insert(9,'ANIM')
                found=1
            if found==0: m2.insert(9,'*')
        else: m2.insert(9,'*')
    else: m2.insert(9,'*')                                          
    mables2.append(m2)
    mable_nr+=1

print 'mables=',mables2
print 'coref=',coref        
str_matches=str_match(mables)
#str_matches=str_match(list(reversed(mables)))
str_matches2=[]
for i in str_matches: str_matches2.append([j[0] for j in i])
print 'str_matches=',str_matches2  
print 'pposat_heads=',pposat_heads
print 'nominal_mods=',dict(nominal_mods)
print 'verbs=',all_verbs
print 'preds=',preds
print 'haben=',haben
print 'gmods=',gmods
#print 'sentences=',sentences
print 'definite=[]'
print 'demonstrative=[]'
print 'determiners=',determiners
print 'prepositions=',prepositions
print '####'          
#sent_nr-=1                 



sys.stderr.write('\n')
