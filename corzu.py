# -*- coding: utf-8 -*-

"""
# ================================================================ #

CorZu - Incremental Entity-mention Coreference Resolution for German
Author: don.tuggener@gmail.com

# ================================================================ #

Usage:

train:
python corzu.py ../tueba_files/train.mables.parzu       #real preprocessing
python corzu.py ../tueba_files/train_gold_prep.mables   #gold preprocessing

test:
python corzu.py ../tueba_files/test.mables.parzu ../tueba_files/test_9.1.conll main.res

when loaded as module main:
train:
main.main('../tueba_files/train_ner.mables.parzu')
test:
main.main('../tueba_files/dev_ner.mables.parzu','../tueba_files/dev_9.1.conll','mle_dev_tmp.res')   #real preprocessing
main.main('../tueba_files/dev_gold_prep.mables','../tueba_files/dev_9.1.conll','mle_dev_tmp.res')   #gold preprocessing

"""

# ============================================= #

''' IMPORTS '''

import copy, cPickle, operator, re, sys, pdb, os, subprocess
from math import ceil,log
from itertools import combinations
from collections import defaultdict,Counter
from verbadicendi import vdic
import random

# ============================================= #

''' SETTINGS '''

global mode, output_format
mode='test'             #train or test
output_format='semeval'   #semeval or tueba; take semeval for runnning CorZu on raw text

global preprocessing
preprocessing='real'    # gold or real

global classifier
classifier='mle'        #mle, thebeast, wapiti
if classifier=='mle':
    global ante_counts,pronouns_counts
    ante_counts,pronouns_counts={'PPER':0,'PPOSAT':0,'PRELS':0,'PRELAT':0,'PDS':0},{'PPER':0,'PPOSAT':0,'PRELS':0,'PRELAT':0,'PDS':0}
if classifier=='wapiti':
    global wapiti_mode,wapiti_path
    wapiti_mode='sequence'    #sequence one instance per pronoun and all ante_cands
    #wapiti_mode='single'     #single is one training instance per ante_cand-pper-pair, 
    if wapiti_mode=='single': wapiti_path='wapiti'
    if wapiti_mode=='sequence': wapiti_path='wapiti_seq'    
if classifier=='thebeast':
    global train_file
    train_file='mln/thebeast.train.tueba'
    if mode=='train' and os.path.exists(train_file): 
        os.remove(train_file)  #delete the training instances file if we train, as we append to it  

global avg_ante_counts,single_ante_counts
avg_ante_counts=defaultdict(list)
single_ante_counts=defaultdict(list)
global output_pronoun_eval, trace_errors, trace_nouns
trace_errors=False  #trace pronoun resolution errors
trace_nouns=False   #trace noun resolution errors
if mode=='train':
    trace_errors=False
    trace_nouns=False
output_pronoun_eval=False    #print pronoun classifier accuracy for cases where the true antecedent is actually available (i.e. among the candidates)    

global link_ne_nn_sets
link_ne_nn_sets=False    #rule-based concatenation of person NE entitiy sets to certain animate common noun sets
if link_ne_nn_sets:
    global eval_link_ne_nn_sets  
    eval_link_ne_nn_sets=defaultdict(int)

global verb_postfilter, twin_mode, verb_postfilter_context_w2v, verb_postfilter_context_graph
verb_postfilter_context_w2v=False
verb_postfilter_context_graph=False
verb_postfilter='off' #train, test, or off; re-ranks the top two candidates based on verb semantics
if verb_postfilter=='train':
    global verb_postfilter_str,verb_postfilter_feature_names
    verb_postfilter_str,verb_postfilter_feature_names=[],[]
if verb_postfilter!='off' or verb_postfilter_context_w2v or verb_postfilter_context_graph:    
    global verb_postfilter_cnt, accuracy
    accuracy=defaultdict(lambda: defaultdict(lambda:defaultdict(int)))    
    verb_postfilter_cnt=0
twin_mode='off'   # train, test, or off; re-ranks the top two candidates; no real improvement
if twin_mode!='off' or verb_postfilter!='off':    # can only have twin mode with ranked candidates, i.e. test mode
    mode='test'
if verb_postfilter!='off' or verb_postfilter_context_w2v or verb_postfilter_context_graph:
    import numpy as np
    import compsplit
    import gensim
    import networkx as nx
    import graph_access as graph  
    from scipy.spatial.distance import cosine
    from scipy.sparse import csr_matrix     
  
# ============================================= #        

''' FUNCTION DEFINITIONS '''

def morph_comp(m1,m2):  #m1=[person,number,gender]
    """compare morphology values"""
    if m1[1]==m2[1] and m1[2]==m2[2]: return 1  #exact match
    if m1[1]=='*' and m1[2]=='*': return 1  #ante underspecified
    if m2[1]=='*' and m2[2]=='*': return 1  #anaph underspecified
    if m1[1]=='*' and m1[2]==m2[2]: return 1
    if m2[1]=='*' and m1[2]==m2[2]: return 1
    if m1[2]=='*' and m1[1]==m2[1]: return 1
    if m2[2]=='*' and m1[1]==m2[1]: return 1
    
def pper_comp(ante,pper):
    """morphological compatibility check for personal and relative pronouns"""
    #if pper[5]==1: return 0 #don't resolve 1st and 2nd person pronouns
    #if pper[5]==1 and ante[9]=='ANIM' and ante[7]=='SG': return 1 #1st person pronoun to animate antes
    if ante[-1]=='es': return 0
    if pper[1]-ante[1] >3: return 0 #sentence distance    
    if pper[5]!=ante[5]: return 0   #Person agreement
    if pper[-1].lower() in ['sie','ihre','ihr'] and ante[7]=='SG' and ante[6] in ['MASC','NEUT']: return 0 
    if pper[-1].lower() in ['sie','ihre','ihr'] and ante[-1].lower() in ['er','seine','sein']: return 0 
    if pper[-1].lower() in ['er','seine','sein'] and ante[-1].lower() in ['sie','ihre','ihr']: return 0            
    if pper[-1].lower() in ['er','seine','sein'] and ante[6] in ['FEM']: return 0        
    if not ante[4]=='PPOSAT' and not pper[4] in ['PRELS','PRELAT'] and not ante[5]==1: #same head exclusive, except PPOSAT as ante and PRELS as anaph
        if pper[1]==ante[1] and pper[10]==ante[10] and pper[11]==ante[11]!='*': return 0
    if not ante[4]=='PPOSAT' and not pper[4] in ['PRELS','PRELAT']:        
        if pper[1]==ante[1] and pper[2]>=ante[2] and pper[3]<=ante[3]: return 0 # pper not in ante extension
    if morph_comp(ante[5:8],pper[5:8]): return 1
    
def pposat_comp(ante,pper):
    """morphological compatibility check for possessive pronouns"""  
    if pper[5]!=ante[5]: return 0   #Person
    if pper[1]-ante[1] >3: return 0 #sentence distance
    if ante[6]=='*' and ante[7]=='*': return 1  #ante underspecified
    if re.match('[Mm]ein.*',pper[-1]):
        if ante[5]==1 and ante[7]=='SG': return 1
    if re.match('[Dd]ein.*',pper[-1]):
        if ante[5]==2 and ante[7]=='SG': return 1
    if re.match('[Ss]ein.*',pper[-1]) and not ante[-1].lower() in ['sie','ihre','ihr']:
        if ante[6] in ['MASC','NEUT','*'] and ante[7]=='SG' and ante[5]==3: return 1
    if re.match('[Ii]hr.*',pper[-1]) and not ante[-1].lower() in ['er','seine','sein']:
        if ante[6] in ['FEM','*'] and ante[7]=='SG' and ante[5]==3: return 1        
    if re.match('[Ii]hr.*',pper[-1]) and not ante[-1].lower() in ['er','seine','sein']:
        if ante[7]=='PL' and ante[5]==3: return 1        
    if re.match('[Ee]ur.*',pper[-1]):
        if ante[7]=='PL': return 1   
    if re.match('[Uu]ns.*',pper[-1]):
        if ante[7]=='PL' and ante[5]==1: return 1        

def update_csets(ante,mable):
    """disambiguate pronoun and update coreference partition"""
    mable[9]=ante[9]                                #animacy projection of ante to mable
    mable[12]=ante[12]                              #ne_type    
    if mable[4]=='PRF':                             #override morpho and gf of reflexives by ante
        mable[5:10]=ante[5:10]
        
    #if ante[4] in ['NN','NE'] and mable[4] in ['NN','NE']: mable[6:8]=ante[6:8]
    
    if ante[6]!='*' and mable[6]=='*': mable[6]=ante[6]
    elif ante[6]=='*' and mable[6]!='*': ante[6]=mable[6]
    if ante[7]!='*' and mable[7]=='*': mable[7]=ante[7]              
    elif ante[7]=='*' and mable[7]!='*': ante[7]=mable[7]

    if mable[4]=='PPOSAT':
        if ante[1]==mable[1]: mable[8]=ante[8]      #same sentence: keep gf/salience of ante
            
    if ante in wl:  #ante is from wl: open new cset
        csets.append([ante,mable])
        wl.remove(ante) 
    else:           #ante is from cset: append to set
        for cset in csets:
            if ante in cset:
                cset.append(mable)
                break
                
def get_true_ante(mable,ante_cands_wl,ante_cands_csets): 
    """return true antecedent if amongst antecedent candidates"""   
    try:
        #find cset containing the markable. markable is not the first mention in the cset (otherwise no antecedent)
        coref_id_anaphor=next(x for x,y in coref.items() if mable[1:4] in y and sorted(y).index(mable[1:4])!=0)
        all_antes=ante_cands_wl+ante_cands_csets
        all_antes.sort(reverse=True)    #reverse sort to find most recent ante
        true_ante=next(a for a in all_antes if a[1:4] in coref[coref_id_anaphor])         
    except:
        true_ante=[]    
    return true_ante

def get_mable(tm):
    try: 
        return next(m for m in mables if m[1:4]==tm)
    except StopIteration:
        return tm
        
# ============================================= # 

''' VERB SEMANTICS RELATED STUFF '''

def load_verb_res():    
    ''' Load verb semantics related resources. '''
    global G, w2v_model, w2v_model_gf
    sys.stderr.write('Loading graph...')
    G=nx.read_gpickle('../../sdewac_graph/verbs_and_args_no_subcat.gpickle') 
    sys.stderr.write(' done.\nLoading word2vec models...')
    w2v_model_gf=gensim.models.Word2Vec.load('../../word2vec/vectors_sdewac_gf_skipgram_min50_new.gensim')    
    sys.stderr.write(' done.\n')        
    
def get_lex(m,res,gf=None):
    ''' Return a common noun for a markable '''
    lex=m[-1]
    if m[4]=='NE': #handle coref set info here: try to get a nominal desciptor
        if (m[1],m[2]) in nominal_mods:
            lex=nominal_mods[m[1],m[2]][0]
        else: 
            try:
                cset_m=next(c for c in csets if m in c)
                ne_mentions=[mention for mention in cset_m if mention[4]=='NE']
                for mention in ne_mentions:
                    if (mention[1],mention[2]) in nominal_mods:
                        lex=nominal_mods[mention[1],mention[2]][0]
                        break
            except StopIteration:
                lex=m[-1]                               
    elif not m[4]=='NN':    #get noun ante for non-NN candidates, i.e. pronouns            
        try:
            cset_m=next(c for c in csets if m in c)
            try:
                nn_mention=next(mention for mention in cset_m if mention[4]=='NN')
                lex=nn_mention[-1]
            except StopIteration:
                ne_mentions=[mention for mention in cset_m if mention[4]=='NE']
                for mention in ne_mentions:
                    if (mention[1],mention[2]) in nominal_mods:
                        lex=nominal_mods[mention[1],mention[2]][0]
                        break
                else:
                    if not ne_mentions==[]:
                        lex=ne_mentions[0][-1]
                    else:
                        lex=m[-1]
        except StopIteration:
            lex=m[-1] 
    if '-' in lex:  #take hyphenated head
        lex=lex.split('-')[-1]
    if '|' in lex:  #ambiguous lemma, take the one that is in the w2c model
        lex=lex.split('|')[0]   #crudely take the first lemma
    if ' und ' in lex:
        lex=lex.split(' und ')[0]  #conjunction head: 'man and wife' -> 'man'                     
    try:
        if not lex.decode('utf8').isupper(): #Don't change DDR to Ddr
            try: lex=lex.decode('utf8').title()   #PolitikerIn -> Politikerin         
            except: pass
    except: pass        
    if res=='w2v_model_gf': #need unicode
        try:
            lex_unicode=lex.decode('utf8')
        except:
            lex_unicode=lex     
        if lex_unicode+'^'+gf in w2v_model_gf:
            return lex_unicode+'^'+gf
        else:
            if unichr(223) in lex_unicode and lex_unicode.replace(unichr(223),'ss')+'^'+gf in w2v_model_gf:
                return lex_unicode.replace(unichr(223),'ss')+'^'+gf             
            if gf=='root' and lex_unicode+'^subj' in w2v_model_gf:
                return lex_unicode+'^subj'
            #only split if it is not an NE
            if not (m[4]=='NE' and lex_unicode==m[-1]):
                lex_split=compsplit.split_compound(lex_unicode)
                if lex_split[0][0]>0:
                    lex_unicode=lex_split[0][2]+'^'+gf
                    if lex_unicode in w2v_model_gf: 
                        return lex_unicode
                    if unichr(223) in lex_unicode and lex_unicode.replace(unichr(223),'ss')+'^'+gf in w2v_model_gf:
                        return lex_unicode.replace(unichr(223),'ss')+'^'+gf       
        return lex_unicode+'^'+gf
    if res=='w2v_model': #need unicode
        try:
            lex_unicode=lex.decode('utf8')
        except:
            lex_unicode=lex     
        if lex_unicode in w2v_model:
            return lex_unicode
        else:
            if unichr(223) in lex_unicode and lex_unicode.replace(unichr(223),'ss') in w2v_model:
                return lex_unicode.replace(unichr(223),'ss')             
            #only split if it is not an NE
            if not (m[4]=='NE' and lex_unicode==m[-1]):                
                lex_split=compsplit.split_compound(lex_unicode)
                if lex_split[0][0]>0:
                    lex_unicode=lex_split[0][2]
                    if lex_unicode in w2v_model: 
                        return lex_unicode
                    if unichr(223) in lex_unicode and lex_unicode.replace(unichr(223),'ss') in w2v_model:
                        return lex_unicode.replace(unichr(223),'ss')
        return lex_unicode        
    if res=='graph':    #need utf8
        try:
            lex=lex.encode('utf8')
        except:
            pass     
        if lex in G:
            return lex   
        else:
            try:
                lex=lex.decode('utf8')
                if unichr(223) in lex and lex.replace(unichr(223),'ss') in G:
                    return lex.replace(unichr(223),'ss').encode('utf8')        
                if not (m[4]=='NE' and lex==m[-1].decode('utf8')):                    
                    lex_split=compsplit.split_compound(lex)
                    if lex_split[0][0]>0:
                        lex=lex_split[0][2].encode('utf8')
            except: pass
        try:
            return lex.encode('utf8')
        except:
            return lex

def get_lex_token(lex,res,gf=None):
    """
    Try to map a noun to an instance in either the graph, word2vec
    For word2_vec return unicode, for graph utf8
    res: 'graph', 'w2v_model_gf'
    """        
    if '-' in lex:  #take hyphenated head
        lex=lex.split('-')[-1]
    if '|' in lex:  #ambiguous lemma, take the one that is in the w2c model
        lex=lex.split('|')[0]   #crudely take the first lemma            
    if ' und ' in lex:
        lex=lex.split(' und ')[0]  #conjunction head: 'man and wife' -> 'man'                     
    try:
        if not lex.decode('utf8').isupper(): #Don't change DDR to Ddr
            try: lex=lex.decode('utf8').title()   #PolitikerIn -> Politikerin         
            except: pass
    except: pass     
    if res=='w2v_model_gf': 
        try:
            lex_unicode=lex.decode('utf8')
        except:
            lex_unicode=lex     
        if lex_unicode+'^'+gf in w2v_model_gf:
            return lex_unicode+'^'+gf
        else:
            if unichr(223) in lex_unicode and lex_unicode.replace(unichr(223),'ss')+'^'+gf in w2v_model_gf:
                return lex_unicode.replace(unichr(223),'ss')+'^'+gf             
            if gf=='root' and lex_unicode+'^subj' in w2v_model_gf:
                return lex_unicode+'^subj'
            lex_split=compsplit.split_compound(lex_unicode)
            if lex_split[0][0]>-0.5:
                lex_unicode=lex_split[0][2]
                if lex_unicode+'^'+gf in w2v_model_gf: 
                    return lex_unicode+'^'+gf
                if unichr(223) in lex_unicode and lex_unicode.replace(unichr(223),'ss')+'^'+gf in w2v_model_gf:
                    return lex_unicode.replace(unichr(223),'ss')+'^'+gf       
        return lex_unicode+'^'+gf
    if res=='w2v_model': #need unicode
        try:
            lex_unicode=lex.decode('utf8')
        except:
            lex_unicode=lex     
        if lex_unicode in w2v_model:
            return lex_unicode
        else:
            if unichr(223) in lex_unicode and lex_unicode.replace(unichr(223),'ss') in w2v_model:
                return lex_unicode.replace(unichr(223),'ss')             
            lex_split=compsplit.split_compound(lex_unicode)
            if lex_split[0][0]>0:
                lex_unicode=lex_split[0][2]
                if lex_unicode in w2v_model: 
                    return lex_unicode
                if unichr(223) in lex_unicode and lex_unicode.replace(unichr(223),'ss') in w2v_model:
                    return lex_unicode.replace(unichr(223),'ss')
        return lex_unicode                
    if res=='graph':
        try:
            lex=lex.encode('utf8')
        except:
            pass                 
        if lex in G:
            return lex   
        else:
            try:
                lex=lex.decode('utf8')
                if unichr(223) in lex and lex.replace(unichr(223),'ss') in G:
                    return lex.replace(unichr(223),'ss').encode('utf8')        
                lex_split=compsplit.split_compound(lex)
                if lex_split[0][0]>0:
                    lex=lex_split[0][2].encode('utf8')
            except: pass
        try:
            return lex.encode('utf8')
        except:
            return lex

# ============================================= # 

''' ANTECEDENT SELECTION '''
      
def get_best(ante_cands,ante_cands_csets,mable,docid):
    ''' Antecedent selection: return best of the candidates. '''
    
    ante=[]
    all_antes=ante_cands+ante_cands_csets
    if all_antes==[]:
        return ante
    all_antes.sort(reverse=True)
    ante_cands.sort(reverse=True)
    ante_cands_csets.sort(reverse=True)    

    if mode=='train':

        if classifier=='mle':
            if not mable[4] in raw_counts: 
                raw_counts[mable[4]]={} 
            pronouns_counts[mable[4]]+=1
            ante_counts[mable[4]]+=len(all_antes)
            
        if classifier=='thebeast':
            f=open(train_file,'a')  #accumualte training cases

        if classifier=='wapiti':
            wapiti_cmd=''
                
        ante=get_true_ante(mable,ante_cands,ante_cands_csets)
        if ante==[]:    #don't learn on cases where there is no true ante
            return ante

    if mode=='test':
        
        if len(all_antes)==1:
            single_ante_counts[mable[4]].append(1)
            return all_antes[0]
                
        #random baseline
        #return all_antes[random.randint(0,len(all_antes)-1)]
        
        #most recent baseline
        #return all_antes[0]
        
        #most recent subject baseline
        #try: return next(m for m in all_antes if m[8]=='SUBJ')
        #except StopIteration: return all_antes[0]
        
        #upper bound
        #true_ante=get_true_ante(mable,ante_cands,ante_cands_csets)     
        #if not true_ante==[]: return true_ante

        if classifier=='mle':    
            weighted_antes=[]          
            avg_ante_counts[mable[4]].append(len(all_antes))       
            if not mable[4] in raw_counts_twin_features:                
                raw_counts_twin_features[mable[4]]={}            
            
        if classifier=='thebeast':
            f=open('mln/test.atoms','w')  #single test case
            
        if classifier=='wapiti':
            wapiti_cmd=''   
            wapiti_ante_weights=[]
                          
    if classifier=='thebeast':  #write the features for thebeast
        f.write('>>\n')
        f.write('>docid\n'+re.search('\d+',docid).group()+'\n\n')               
        
        if mode=='train':   #add the true antecedent 
            f.write('>coref\n')
            f.write(str(mable[0])+' '+str(ante[0])+'\n\n')    

        f.write('>anaphor\n'+str(mable[0])+'\n\n')
        
        if mable[4]=='PPOSAT':
            if (mable[1],mable[2]) in pposat_heads:
                pposat_head=pposat_heads[mable[1],mable[2]]
                f.write('>pposat_head_gf\n')
                f.write(str(mable[0])+' "'+pposat_head[7].upper()+'"\n\n')
                                    
        f.write('>candidate_index\n')
        for m in all_antes: 
            f.write(str(m[0])+' '+str(all_antes.index(m))+'\n')
        f.write('\n>ante_cand_wl\n')
        for m in ante_cands: 
            f.write(str(m[0])+'\n')
        f.write('\n>ante_cand_cset\n')
        for m in ante_cands_csets: 
            f.write(str(m[0])+'\n')
        f.write('\n>ante_cand\n')
        for m in all_antes: 
            f.write(str(m[0])+'\n')
        f.write('\n>ante_type\n')
        for m in ante_cands: 
            f.write(str(m[0])+' "New"\n')
        for m in ante_cands_csets: 
            f.write(str(m[0])+' "Old"\n')
        f.write('\n>has_genus\n')
        for m in all_antes:
            f.write(str(m[0])+' "'+m[6]+'"\n')
        f.write(str(mable[0])+' "'+mable[6]+'"\n\n')
        f.write('>has_num\n')
        for m in all_antes:
            f.write(str(m[0])+' "'+m[7]+'"\n')
        f.write(str(mable[0])+' "'+mable[7]+'"\n\n')                
        f.write('>has_anim\n')
        for m in all_antes: 
            f.write(str(m[0])+' "'+m[9]+'"\n')
        f.write(str(mable[0])+' "'+mable[9]+'"\n\n')        
        f.write('>has_pos\n')
        for m in all_antes: 
            f.write(str(m[0])+' "'+m[4]+'"\n')
        f.write(str(mable[0])+' "'+mable[4]+'"\n\n')
        f.write('>has_gf\n')
        for m in all_antes: 
            f.write(str(m[0])+' "'+m[8]+'"\n')
        f.write(str(mable[0])+' "'+mable[8]+'"\n\n')
        f.write('>has_ne_type\n')
        for m in all_antes: 
            if not m[12]=='*': 
                f.write(str(m[0])+' '+m[12]+'\n') 
        f.write('\n>has_connector\n')
        for m in all_antes:
            if m[-2]=='conn': 
                f.write(str(m[0])+'\n')
        if mable[-2]=='conn': 
            f.write(str(mable[0])+'\n')        
        f.write('\n>in_sentence\n')
        for m in all_antes: 
            f.write(str(m[0])+' '+str(m[1])+'\n')
        f.write(str(mable[0])+' '+str(mable[1])+'\n\n')              
        f.write('\n>parent_id\n')
        for m in all_antes: 
            f.write(str(m[0])+' '+str(m[10])+'\n')
        f.write(str(mable[0])+' '+str(mable[10])+'\n\n')
        f.write('\n')
        f.close()     
        
        if mode=='test':                
            thebeast.stdin.write('load corpus from "mln/test.atoms";test to "mln/thebeast.res";\n')
            outs=[]
            while True:
                out = thebeast.stdout.readline()
                outs.append(out)
                if out.startswith('End:coref'): 
                    break                                   
            res_thebeast=open('mln/thebeast.res').read()
            try:
                ante=re.search('>coref\n\d+.*?(\d+)',res_thebeast).group(1)
                ante=next(a for a in all_antes if a[0]==int(ante))
            except:        
                pass                  

    if classifier=='mle':
        ante_features={}
                
        for a in all_antes:    
            features={}            

            # ================================================================ #
            ''' GET FEATURES '''

            """
            
            ''' BASELINE FEATURES '''

            # sent. distance
            features['sent_dist'] = mable[1]-a[1]

            # markable distance
            features['mable_dist_all'] = mable[0]-a[0]
            
            # gramm. function of ante
            features['gf_ante'] = a[8]
            
            # gf parallel
            features['gf_parallel']='parallel' if mable[8] == a[8] else 'not_parallel'
            
            # pos ante
            features['pos_ante'] = a[4]

            # det
            #'''
            if a[4] == 'NN':
                if (a[1],a[2]) in determiners:
                    det=determiners[a[1],a[2]]
                    if det.startswith('d'): features['def']='def'
                    elif det.startswith('e'): features['def']='indef'
                    else: features['def']='unk'
                else: features['def']='indef'
            else: features['def'] = 'na'
            #'''
            
            """      
                  
            # ================================================================ #
            ''' EXTENDED FEATURES '''
            
            #'''
            
            """ DISTANCE """

            #sentence distance
            if mable[4] not in ['PRELS','PRELAT']:
                sent_type=mable[13] #conn or noconn, also the feature name
                features[sent_type]=mable[1]-a[1]
                
            #markable distance in the same sentence
            if mable[1]==a[1]:      #same sentence
                features['mable_dist']=mable[0]-a[0]

            #candidate index
            cand_index=all_antes.index(a)
            features['cand_index']=cand_index  

            """ SYNTAX """
            
            #grammatical function of the antecedent 
            features['gf_ante']=a[8]        

            #gf_seq: sent. dist., gf ante, gf pronoun, PoS ante
            gf_seq=str(mable[1]-a[1])+'_'+a[8]+'_'+mable[8]+'_'+a[4]
            features['gf_seq']=gf_seq
         
            #PPOSAT specific features:
            if mable[4]=='PPOSAT':
                same_head='not_same_head'  # PPOSAT is governed by same verb as ante
                if a[1]==mable[1] and a[10]==mable[10]:
                    same_head='same_head'                
                if (mable[1],mable[2]) in pposat_heads:
                    pposat_head=pposat_heads[mable[1],mable[2]]
                    features['head_gf_seq']='_'.join([str(mable[1]-a[1]),a[8],pposat_head[7].upper(),a[4]])
                    if same_head=='same_head':
                        features['same_head_gf_seq']='_'.join([a[8],pposat_head[7].upper(),a[4]])

            #(sub)clause type, i.e. gf of verb governing the candidate
            if (a[1],a[10]) in verbs:
                features['subclause']=verbs[a[1],a[10]]['verb'][7]
            else:
                features['subclause']='*'
                
            #(sub)clause type seq
            sent_dist='0' if mable[1]==a[1] else '1'
            if (mable[1],mable[10]) in verbs:
                features['subclause_seq']=sent_dist+'_'+features['subclause']+'_'+verbs[mable[1],mable[10]]['verb'][7]
            else:                
                features['subclause_seq']=sent_dist+'_'+features['subclause']+'_*' 

            """ ANTE PROPERTIES """    
            #animacy of the antecedent; condition on gen+num also? mln has gen atleast             
            features['anim_num_gen']=a[9]+'_'+a[6]+'_'+a[7]         

            #ne_type fo the antecedent
            if not a[12]=='-': features['ne_type']=a[12] 
            #features['ne_type']=a[12] 

            #gender of ante
            features['gen']=a[6]            
                        
            #number of ante
            features['num']=a[7]

            #preposition
            if a[8]=='PN' and (a[1],a[2]) in prepositions: features['pp_ante']=prepositions[a[1],a[2]]

            """ DISCOURSE """

            #discourse status: old/new
            if a in ante_cands_csets: discourse_status='old'
            else: discourse_status='new'
            features['discourse_status']=discourse_status

            # cset candidate: how old is the entity, i.e. in which sentence was it introduced
            if a in ante_cands_csets:
                ante_cset=next(c for c in csets if a in c)
                features['entity_introduction_sentence']=ante_cset[0][1]-mables[0][1]

            #'''
            
            ante_features[a[0]]=features
                               
            if mode=='test':
                weight={}
            
            pos=mable[4]            
            #add feature counts to pos/neg raw_counts depending whether a is the true_ante or not                
            #for i in range(1,len(features)+1): #all feature combinations
            #for i in range(1,4):    #only unary features                  
            for i in range(1,2):    #only unary features
                for c in combinations(features,i):
                    combined_feature_name='/'.join([x for x in c])               
                    combined_feature_values='/'.join([str(features[x]) for x in c])       
                    if mode=='train':            
                        #dict housekeeping
                        if not combined_feature_name in raw_counts[pos]:
                            raw_counts[pos][combined_feature_name]={}
                        if not combined_feature_values in raw_counts[pos][combined_feature_name]:
                            raw_counts[pos][combined_feature_name][combined_feature_values]={'pos':0,'neg':0}
                        #count incrementation
                        if a!=ante:
                            raw_counts[pos][combined_feature_name][combined_feature_values]['neg']+=1
                        else:
                            raw_counts[pos][combined_feature_name][combined_feature_values]['pos']+=1
                    if mode=='test':
                        if combined_feature_name in weights_global[pos]:
                            if combined_feature_values in weights_global[pos][combined_feature_name]:
                                weight[combined_feature_name]=weights_global[pos][combined_feature_name][combined_feature_values]
                                #weight.append(weights_global[pos][combined_feature_name][combined_feature_values])    
            if mode=='test':
                weighted_antes.append([reduce(operator.mul,weight.values()),a,weight])  #product of the weights
                                         
        if mode=='test':
            
            if not weighted_antes==[]:
                weighted_antes.sort(reverse=True)
                ante=weighted_antes[0][1]
                
                if verb_postfilter_context_graph:
                
                    if mable[4] in ['PPER','PRELS','PDS'] and mable[8] in ['SUBJ','OBJA','OBJD','OBJP'] and not mable[11]=='*' and len(weighted_antes)>1:
                        global verb_postfilter_cnt
                        verb_postfilter_cnt+=1
                        pper_verb_utf8=mable[11].replace('#','')
                        
                        if pper_verb_utf8 in G:
                            gf=mable[8].lower()
                            add_args={}
                            
                            # Check for additional verb arguments, i.e. construct context
                            if (mable[1],mable[10]) in verbs:  
                                for v_gf,varg in verbs[mable[1],mable[10]].items():
                                    if v_gf in ['verb','subcat']: continue
                                    if v_gf in ['subj','obja','objd','objp']:
                                        if varg[4]=='NN':        
                                            #if v_gf=='objp': v_gf='pn' # We don't have PNs in the graph, useless
                                            # Only selected args; well we don't have anything else, really, so the filter doesn't do anything here
                                            #if v_gf in ['pn', 'subj', 'obja', 'objd']:
                                            if '-' in varg[2]: varg[2]=re.sub('.*-','',varg[2])
                                            if varg[2] in G: add_args[v_gf]=varg[2]
                                            else:
                                                lex_varg=get_lex_token(varg[2],'graph')
                                                if '-' in lex_varg: lex_varg=re.sub('.*-','',lex_varg)
                                                if lex_varg in G: add_args[v_gf]=lex_varg
                                        elif varg[4] in ['PPER','PRELS'] and int(varg[0])<mable[2]: # Get nominal ante of pronoun args
                                            try:
                                                closest_mable=next(m for m in mables if m[1]==mable[1] and m[2]==int(varg[0]))
                                                if closest_mable!=mable: 
                                                    lex_varg=get_lex(closest_mable,'graph')
                                                    if '-' in lex_varg: lex_varg=re.sub('.*-','',lex_varg)
                                                    if not lex_varg.startswith(varg[2].title()) and lex_varg in G: add_args[v_gf]=lex_varg
                                            except StopIteration: pass

                            # Determine nbest args of pper verb
                            # TODO: use count to sort instead of npmi?
                            pper_verb_cooc1=graph.first_order_coocurrences(G,pper_verb_utf8,gf) 
                            pper_verb_cooc1_lemmas=[n[1] for n in pper_verb_cooc1]
                            if len(pper_verb_cooc1)>=100: nbest=10
                            else: nbest=len(pper_verb_cooc1)/10
                            if nbest<3: nbest=3    # at least three
                            
                            ante_features={}
                            reranked_antes=[]
                            
                            for ax in weighted_antes[:2]: 
                                a=ax[1]   
                                features={}
                                lex_utf8=a[-1] if a[-1] in G else get_lex(a,'graph')
                                if '-' in lex_utf8: lex_utf8=re.sub('.*-','',lex_utf8)
                                
                                if lex_utf8 in G:
                                    verb_features={}
                                    # 1st order co-occurrence
                                    verb_features['cooc1']=graph.preference(G,lex_utf8,pper_verb_utf8,gf)

                                    # Similar arg cooc1
                                    """
                                    di=graph.preference(G,lex_utf8,pper_verb_utf8,gf)
                                    if di>0: verb_features['cooc1']=di
                                    else:
                                        sibling=graph.similar_arg(G,lex_utf8,pper_verb_utf8,gf)
                                        cooc1_sibling=graph.preference(G,sibling[1],pper_verb_utf8,gf)   
                                        verb_features['cooc1']=cooc1_sibling*sibling[0]                                     
                                    """
                                    #"""
                                    # Cooc1 of pper_verb of pper_verb arg most similar to ante_cand; unfortunately a speed bottleneck
                                    sibling=graph.similar_arg(G,lex_utf8,pper_verb_utf8,gf,weight='scount')
                                    cooc1_sibling=graph.preference(G,sibling[1],pper_verb_utf8,gf)
                                    # Degrade preference of similar arg by similarity to the similar arg
                                    verb_features['cooc1_similar']=cooc1_sibling*sibling[0]
                                    #"""
                                    
                                    """
                                    #nmf scaled
                                    #maxx=max(subj_nmf[n_index_subj['Hund']].toarray().flatten())
                                    #nmf_pref=subj_nmf[n_index_subj['Hund'],v_index_subj['bellen']]
                                    #nmf_pref=nmf_pref/maxx
                                    """
                                    
                                    # Similarity of ante_cand to topn cooc1 args of pper_verb_utf8
                                    #"""
                                    #cooc1_sim=[ graph.similarity(G,lex_utf8,n,gf,weight='scount') for n in pper_verb_cooc1_lemmas[:nbest] ]
                                    #cooc1_sim=np.mean(cooc1_sim) if not cooc1_sim==[] else 0
                                    #verb_features['cooc1_sim']=cooc1_sim
                                    verb_features['cooc1_sim']=graph.similarity_one_vs_all(G,lex_utf8,pper_verb_cooc1_lemmas[:nbest],gf,gf,weight='scount')
                                    #"""
                                    
                                    """            
                                    # Similarity of ante_verb and pper_verb
                                    # Only in cases where both candidates have relevant gf, i.e. not GMOD or PN
                                    try:
                                        next(m for m in weighted_antes[:2] if not m[1][8] in ['SUBJ','OBJA','OBJD','OBJP']) # Relevant GFs
                                    except:
                                        verb_sim=graph.weighted_path(G,pper_verb_utf8,a[11],gf,a[8].lower(),weight='scount') if a[8] in ['SUBJ','OBJA','OBJD','OBJP'] and a[11] in G else 0
                                        verb_features['verb_sim']=verb_sim
                                    """
                                    verb_features['verb_sim']=graph.weighted_path(G,pper_verb_utf8,a[11],gf,a[8].lower(),weight='scount') if a[8] in ['SUBJ','OBJA','OBJD','OBJP'] and a[11] in G else 0

                                    #"""
                                    # Similarity of ante to additional pper_verb args, i.e. context
                                    add_args_sim=[ graph.weighted_path(G,lex_utf8,v_arg,gf,v_gf,weight='scount') for v_gf,v_arg in add_args.items() ]        
                                    # Take the mean instead of sum to not overweigh add_args_sim feature in the sum below
                                    add_args_sim=np.mean(add_args_sim) if not add_args_sim==[] else 0
                                    verb_features['add_args_sim']=add_args_sim
                                    #"""

                                    """                   
                                    # Similarity of nbest args of ante_verb to nbest args of pper_verb
                                    # cooc1 of nbest ante_verb args with pper_verb; only where cooc1>0 
                                    verb_arg_sim,sim_verb_args_pper_verb=0,0
                                    #TODO Problem here: only similarity of subj, obja, objd antes is considered, and it's always positive.
                                    #if we sum up, these candidates will always benefit from that. 
                                    #That is, e.g. gmod and pn candidates are always discouraged.. Then again: how often are they among the top 2 candidates anyway?
                                    if a[8] in ['SUBJ','OBJA','OBJD','OBJP'] and a[11] in G and not pper_verb_cooc1_lemmas==[]:
                                        ante_verb_cooc1=graph.first_order_coocurrences(G,a[11],a[8].lower())
                                        if not ante_verb_cooc1==[]:
                                            if len(ante_verb_cooc1)>=100: nbest=10
                                            else: nbest=len(ante_verb_cooc1)/10
                                            if nbest<3: nbest=3    # at least three   
                                            ante_verb_cooc1_lemmas=[n[1] for n in ante_verb_cooc1[:nbest]]
                                            verb_arg_sim=graph.n_similarity(G,ante_verb_cooc1_lemmas, pper_verb_cooc1_lemmas[:nbest],gf)
                                            
                                            # cooc1 of nbest ante_verb args with pper_verb; only where cooc1>0 
                                            sim_verb_args_pper_verb=[]
                                            for n in ante_verb_cooc1_lemmas:
                                                di=graph.scaled_preference(G,n,pper_verb_utf8,gf)
                                                if not di==0: sim_verb_args_pper_verb.append(di)
                                            sim_verb_args_pper_verb=np.mean(sim_verb_args_pper_verb) if not sim_verb_args_pper_verb==[] else 0
                                            
                                    verb_features['sim_verb_args_pper_verb']=sim_verb_args_pper_verb                                              
                                    verb_features['verb_arg_sim']=verb_arg_sim
                                    """
                                    # Mean of all weights
                                    sim=np.mean(verb_features.values())
                                    ante_features[a[0]]=verb_features
                                    reranked_antes.append([sim,a])
                                    
                            if len(reranked_antes)==2:
                                reranked_antes.sort(reverse=True)
                                true_ante=get_true_ante(mable,ante_cands,ante_cands_csets) 
                                
                                if not true_ante==[] and reranked_antes[0][0]>reranked_antes[1][0]:
                                    verb_ante=reranked_antes[0][1]
                                    
                                    """
                                    # Feature vector for Weka and the lot
                                    runner_up=reranked_antes[1][1]
                                    inst=''
                                    feats=['cooc1', 'cooc1_sim', 'add_args_sim', 'sim_verb_args_pper_verb', 'verb_arg_sim', 'verb_sim']
                                    for f in feats:
                                        inst+="%.2f" % ante_features[verb_ante[0]][f]   #round to 2 decimals
                                        inst+=' '
                                    for f in feats:
                                        inst+="%.2f" % ante_features[runner_up[0]][f]
                                        inst+=' '
                                    for f in feats:
                                        inst+="%.2f" % (ante_features[verb_ante[0]][f]-ante_features[runner_up[0]][f])
                                        inst+=' '
                                    inst+='pos' if verb_ante==true_ante else 'neg'
                                    """

                                    # Trace
                                    #"""
                                    if mable[4]=='PPER':
                                        print '\n\n',mable
                                        print 'true_ante:\t',true_ante
                                        print 'sel_ante:\t',ante     
                                        print 'verb_ante:\t',verb_ante                                
                                        print add_args
                                        print pper_verb_cooc1_lemmas[:nbest]
                                        for x in reranked_antes:print x
                                        for x,y in ante_features.items(): print x,y                                        
                                        print ''
                                        pdb.set_trace()
                                    #"""
                                                                        
                                    # Evaluate
                                    if ante==true_ante==verb_ante:
                                        accuracy[pos][mable[-1]]['both correct']+=1                                           
                                    elif ante==true_ante:
                                        accuracy[pos][mable[-1]]['baseline correct']+=1                                          
                                    elif verb_ante==true_ante:
                                        accuracy[pos][mable[-1]]['verb_sel correct']+=1                                                                             
                                    else:
                                        accuracy[pos][mable[-1]]['both wrong']+=1



                if verb_postfilter_context_w2v:
                    if mable[4] in ['PPER','PRELS','PDS'] and mable[8] in ['SUBJ','OBJA','OBJD','OBJP'] and not mable[11]=='*' and len(weighted_antes)>1:  
                        #global verb_postfilter_cnt
                        #verb_postfilter_cnt+=1
                        pper_verb_unicode=mable[11].replace('#','').decode('utf8')+'^V'
                        if pper_verb_unicode in w2v_model_gf:
                            gf=mable[8].lower()
                            if gf=='objp': gf='pn'
                            context=[]  #build the pronoun's context              
                            if (mable[1],mable[10]) in verbs:  #check for additional verb arguments
                                for v_gf,varg in verbs[mable[1],mable[10]].items():
                                    if v_gf in ['verb','subcat']: continue
                                    #if v_gf in ['pn', 'subj', 'obja', 'objd']: #only selected args                                    
                                    if varg[4]=='NN':        
                                        if v_gf=='objp': v_gf='pn'
                                        if varg[2]+'^'+v_gf in w2v_model_gf: context.append(varg[2]+'^'+v_gf)
                                        else:
                                            lex_varg=get_lex_token(varg[2],'w2v_model_gf',nominal_mods,csets,gf=v_gf)
                                            if lex_varg in w2v_model_gf: context.append(lex_varg)
                                    elif varg[4] in ['PPER','PRELS'] and int(varg[0])<mable[2]: #get nominal ante of pronoun args
                                        try:
                                            closest_mable=next(m for m in mables if m[1]==mable[1] and m[2]==int(varg[0]))
                                            if closest_mable!=mable: 
                                                lex_varg=get_lex(closest_mable,'w2v_model_gf',nominal_mods,csets,varg[7])
                                                if not lex_varg.startswith(varg[2].title().decode('utf8')) and lex_varg in w2v_model_gf: context.append(lex_varg)
                                        except StopIteration: pass

                            context.append(pper_verb_unicode)                        
                            if not context==[]:
                            #if len(context)>1:
                                ante_features={}
                                reranked_antes=[]
                                for ax in weighted_antes[:2]: 
                                    a=ax[1]   
                                    features={}
                                    if a[-1].decode('utf8')+'^'+gf in w2v_model_gf: lex_unicode=a[-1].decode('utf8')+'^'+gf
                                    else: lex_unicode=get_lex(a,'w2v_model_gf',gf.lower())
                                    if lex_unicode in w2v_model_gf:
                                        sim=w2v_model_gf.n_similarity(context,[lex_unicode])    #average similarity to all args
                                        #sim=w2v_model_gf.similarity(pper_verb_unicode,lex_unicode)    #similarity to verb
                                        #multiply vectors of context args, cosine to product af context: much worse
                                        #context_prod=reduce(operator.mul,[w2v_model_gf[carg] for carg in context])
                                        #sim=1 - cosine(w2v_model_gf[lex_unicode],context_prod)
                                        #sim=w2v_model_gf.similarity(lex_unicode,pper_verb_unicode) # only similarity to the verb
                                        reranked_antes.append([sim,a])
                                if len(reranked_antes)==2:
                                    reranked_antes.sort(reverse=True)
                                    true_ante=get_true_ante(mable,ante_cands,ante_cands_csets) 
                                    if not true_ante==[] and reranked_antes[0][0]>reranked_antes[1][0]:
                                        verb_ante=reranked_antes[0][1]
                                        if ante==true_ante==verb_ante:
                                            accuracy[pos][mable[-1]]['both correct']+=1
                                        elif ante==true_ante:
                                            accuracy[pos][mable[-1]]['baseline correct']+=1
                                        elif verb_ante==true_ante:
                                            accuracy[pos][mable[-1]]['verb_sel correct']+=1
                                            """
                                            print '\n\n',mable
                                            print 'true_ante:\t',true_ante
                                            print 'sel_ante:\t',ante                                    
                                            print context
                                            for x in reranked_antes:print x
                                            print ''
                                            pdb.set_trace()
                                            """                                            
                                        else:
                                            accuracy[pos][mable[-1]]['both wrong']+=1
                            

                if not verb_postfilter=='off':
                    #TODO don't restrict gf -> w2v and new graph with pn
                    if mable[4]=='PPER' and mable[8] in ['SUBJ','OBJA','OBJD'] and not mable[11]=='*' and len(weighted_antes)>1:  
                        verb_postfilter_cnt+=1
                        pper_verb_utf8=mable[11].replace('#','')
                        pper_verb_unicode=pper_verb_utf8.decode('utf8')+'^V'
                        gf=mable[8].lower()
                        context=[]
                        add_args_pper_verb={}            
                        if (mable[1],mable[10]) in verbs:  #check for additional verb arguments
                            for v_gf,varg in verbs[mable[1],mable[10]].items():
                                if v_gf in ['verb','subcat']: continue
                                if varg[4]=='NN':
                                    if varg[2] in G: add_args_pper_verb[v_gf]=varg[2]
                                    else:
                                        lex_varg=get_lex_token(varg[2],res='graph')
                                        if lex_varg in G:
                                            add_args_pper_verb[v_gf]=lex_varg                
                                    if v_gf=='objp': v_gf='pn'
                                    if varg[2]+'^'+v_gf in w2v_model_gf: context.append(varg[2]+'^'+v_gf)
                                    else:
                                        lex_varg=get_lex_token(varg[2],res='w2v_model_gf',gf=v_gf)
                                        if lex_varg in w2v_model_gf: context.append(lex_varg)
                                        
                        context_no_verb=list(context)
                        context.append(pper_verb_unicode)                        
                        ante_features={}
                        for ax in weighted_antes[:2]: 
                            a=ax[1]   
                            features={}
                            lex_unicode=get_lex(a,'w2v_model_gf',gf.lower())  #for word2vec models #TODO check if it isn't a pronoun or an NE!
                            lex_utf8=get_lex(a,'graph') #for graph
                            try: lex_utf8=lex_utf8.encode('utf8') #for networkx graph
                            except: pass

                            #first order co-occurrence features
                            di=graph.preference(G,lex_utf8,pper_verb_utf8,gf,weight='count')    #direct insert into graph: raw counts, log2
                            features['direct_insert_counts']=di
                            features['direct_insert_binary']=1 if not di==0 else 0  #whether we have seen the ante pper_verb combination, yes or no
                            di=graph.preference(G,lex_utf8,pper_verb_utf8,gf,weight='tf_idf')   #direct insert into graph: tf-idf, log2
                            features['direct_insert_tf_idf']=di
                            
                            #nmf                    
                            if gf=='subj' and pper_verb_utf8 in v_index_subj and lex_utf8 in n_index_subj:
                                nmf_pref=subj_nmf[n_index_subj[lex_utf8],v_index_subj[pper_verb_utf8]]
                                features['nmf_pref']=nmf_pref
                            elif gf=='obja' and pper_verb_utf8 in v_index_obja and lex_utf8 in n_index_obja:
                                nmf_pref=obja_nmf[n_index_obja[lex_utf8],v_index_obja[pper_verb_utf8]]
                                features['nmf_pref']=nmf_pref
                            elif gf=='objd' and pper_verb_utf8 in v_index_objd and lex_utf8 in n_index_objd:
                                nmf_pref=objd_nmf[n_index_objd[lex_utf8],v_index_objd[pper_verb_utf8]]
                                features['nmf_pref']=nmf_pref
                            else: features['nmf_pref']=0

                            #similarity of ante_verb and pper_verb; TODO this leads to a serious slow down, optimize
                            if not a[11]=='*' and a[8].lower() in ['subj','obja','objd','objp']:
                                ante_verb=a[11].replace('#','')
                                try: ante_verb=ante_verb.encode('utf8')
                                except: pass
                                sim_verbs_tf_idf=[]                
                                paths=graph.paths(G, ante_verb, pper_verb_utf8, a[8].lower(), gf)
                                sim_verbs=len(paths)
                                for path in paths: sim_verbs_tf_idf.append( np.mean( [path[1]['tf_idf'],path[2]['tf_idf']] ) )
                                features['ante_verb_pper_verb_sim_avg']=len(paths)
                                features['ante_verb_pper_verb_sim_avg_tf_idf']=np.mean(sim_verbs_tf_idf) if not sim_verbs_tf_idf==[] else 0
                            else: features['ante_verb_pper_verb_sim_avg'],features['ante_verb_pper_verb_sim_avg_tf_idf']=0,0
                                                  
                            #similarity to additional pper_verb args  
                            if a[8].lower() in ['subj','obja','objd','objp'] and not add_args_pper_verb=={}:
                                sim_add_args=[]
                                sim_add_args_tf_idf=[]                
                                for add_arg_gf,add_arg in add_args_pper_verb.items():
                                    if add_arg_gf in ['subj','obja','objd','objp']:
                                        paths=graph.paths(G, lex_utf8, add_arg, a[8].lower(), add_arg_gf)
                                        sim_add_args.append(len(paths))
                                        for path in paths: sim_add_args_tf_idf.append( np.mean( [path[1]['tf_idf'],path[2]['tf_idf']] ) ) 
                                features['ante_pper_add_args_sim_avg']=np.mean(sim_add_args) if not sum(sim_add_args)==0 else 0
                                features['ante_pper_add_args_sim_avg_tf_idf']=np.mean(sim_add_args_tf_idf) if not sim_add_args_tf_idf==[] else 0
                            else: features['ante_pper_add_args_sim_avg'], features['ante_pper_add_args_sim_avg_tf_idf']=0,0

                            #w2v_model_gf similarity of ante to pper_verb and pper context
                            if lex_unicode in w2v_model_gf and pper_verb_unicode in w2v_model_gf:
                                w2v_sim_pper_verb=w2v_model_gf.similarity(lex_unicode,pper_verb_unicode)    #sim ante pper_verb
                                features['w2v_sim_pper_verb']=w2v_sim_pper_verb
                                if len(context)>1:
                                    w2v_sim_context=w2v_model_gf.n_similarity([lex_unicode],context)
                                    features['w2v_sim_context']=w2v_sim_context #sim ante pper_context
                                    w2v_sim_context_args=w2v_model_gf.n_similarity([lex_unicode],context_no_verb)
                                    features['w2v_sim_context_args']=w2v_sim_context_args 
                                else: features['w2v_sim_context'],features['w2v_sim_context_args']=0,0                     
                            else: features['w2v_sim_pper_verb'],features['w2v_sim_context'],features['w2v_sim_context_args']=0,0,0
                            
                            #w2v_model_gf similarity of ante_verb to pper_verb
                            if pper_verb_unicode in w2v_model_gf:
                                ante_verb_unicode=a[11].replace('#','').decode('utf8')+'^V'
                                if ante_verb_unicode in w2v_model_gf:
                                    features['w2v_sim_ante_verb_pper_verb']=w2v_model_gf.similarity(ante_verb_unicode,pper_verb_unicode)
                                else: features['w2v_sim_ante_verb_pper_verb']=0
                            else: features['w2v_sim_ante_verb_pper_verb']=0                                
                            
                            ante_features[a[0]]=features
                        
                        ante_features['both']={}
                        
                        lex1_utf8=get_lex(weighted_antes[0][1],'graph')
                        lex2_utf8=get_lex(weighted_antes[1][1],'graph')
                        if lex1_utf8 in G and lex2_utf8 in G:
                            graph_sim_cands=graph.similarity(G,lex1_utf8,lex2_utf8,gf)
                            ante_features['both']['graph_sim_cands']= "%.1f" % graph_sim_cands
                        else: ante_features['both']['graph_sim_cands']=0
                        
                        lex1_unicode=get_lex(weighted_antes[0][1],'w2v_model_gf',gf)
                        lex2_unicode=get_lex(weighted_antes[1][1],'w2v_model_gf',gf)                                          
                        if lex1_unicode in w2v_model_gf and lex2_unicode in w2v_model_gf:
                            w2v_sim_cands=w2v_model_gf.similarity(lex1_unicode,lex2_unicode)
                            ante_features['both']['w2v_sim_cands']="%.1f" %  w2v_sim_cands
                        else: ante_features['both']['w2v_sim_cands']=0
                        
                        #ante_features['both']['specificity_tf_idf']=G.node[pper_verb_utf8]['tf_idf'][gf] if pper_verb_utf8 in G else 0
                        
                        for feat in ante_features[weighted_antes[0][1][0]]:
                            if feat in ante_features[weighted_antes[1][1][0]]:
                                if not ante_features[weighted_antes[0][1][0]][feat]+ante_features[weighted_antes[1][1][0]][feat]==0:
                                    if ante_features[weighted_antes[1][1][0]][feat]>ante_features[weighted_antes[0][1][0]][feat]:
                                        diff='>'
                                    elif ante_features[weighted_antes[1][1][0]][feat]<ante_features[weighted_antes[0][1][0]][feat]:
                                        diff='<'
                                    else: diff='='
                                    """
                                    diff=1.*ante_features[weighted_antes[1][1][0]][feat]/(ante_features[weighted_antes[0][1][0]][feat]+ante_features[weighted_antes[1][1][0]][feat])
                                    if diff <0: diff=0
                                    if diff >1: diff=1
                                    """                                         
                                else: diff='='
                           
                                #ante_features['both'][feat]= "%.1f" % diff
                                ante_features['both'][feat]=diff 
                        
                        #features: from ante, competitor, and both #TODO test ony both?
                        #instance_str=' '.join([str(x) for x in ante_features[weighted_antes[0][1][0]].values()])+' '
                        #instance_str+=' '.join([str(x) for x in ante_features[weighted_antes[1][1][0]].values()])+' '
                        instance_str=' '.join([str(x) for x in ante_features['both'].values()])+' '
                        if verb_postfilter_feature_names==[]:
                            verb_postfilter_feature_names.extend([str(x)+'_both' for x in ante_features['both']])
                        if verb_postfilter=='train':
                            true_ante=get_true_ante(mable,ante_cands,ante_cands_csets) 
                            if not true_ante==[]:
                                #TODO change this! pos if the verb selection is correct! But how do we determine the verb selection???
                                if true_ante==ante:
                                    instance_str+='neg' #don't change the antecedent
                                else:
                                    instance_str+='pos' #take verb based selected antecedent
                                verb_postfilter_str.append(instance_str)                                                

                if twin_mode!='off':
                    # binary classification task: take ante from pre-ranking? (i.e. or take 2nd best ranked candidate)
                    # feature structure: ante, 2nd ranked, pronoun                
                    twin_features=defaultdict(lambda: defaultdict(tuple))
                    best_id=ante[0]
                    sec_best_id=weighted_antes[1][1][0]
                    
                    # these features don't change the output
                    """
                    twin_features['gfs']=(ante[8],weighted_antes[1][1][8],mable[8])
                    twin_features['sent_dist']=(mable[1]-ante[1],mable[1]-weighted_antes[1][1][1])
                    twin_features['anim']=(ante_features[best_id]['anim'],ante_features[sec_best_id]['anim'])
                    twin_features['gf_seqs']=(ante_features[best_id]['gf_seq'],ante_features[sec_best_id]['gf_seq'])                        
                    twin_features['cand_index']=(ante_features[best_id]['cand_index'],ante_features[sec_best_id]['cand_index'])                        
                    twin_features['num']=(ante[7],weighted_antes[1][1][7])                        
                    twin_features['gen']=(ante[6],weighted_antes[1][1][6])
                    twin_features['discourse_status']=(ante_features[best_id]['discourse_status'],ante_features[sec_best_id]['discourse_status'])
                    # this one makes it worse                    
                    twin_features['anim_cand_index']=(ante_features[best_id]['cand_index'],ante_features[best_id]['anim'],ante_features[sec_best_id]['cand_index'],ante_features[sec_best_id]['anim'])
                    """                    
                    # TODO: try anim vs. discourse_status, cand_index vs. discourse_status
                    twin_features['discourse_status_cand_index']=(ante_features[best_id]['cand_index'],ante_features[best_id]['discourse_status'],ante_features[sec_best_id]['cand_index'],ante_features[sec_best_id]['discourse_status'])
                    
                    pos=mable[4]
                    if twin_mode=='train':  #collect the counts, #TODO count cases for bias odd ratio
                        true_ante=get_true_ante(mable,ante_cands,ante_cands_csets)            
                        for feat in twin_features:
                            if not feat in raw_counts_twin_features[pos]:
                                raw_counts_twin_features[pos][feat]={}
                            if not twin_features[feat] in raw_counts_twin_features[pos][feat]:
                                raw_counts_twin_features[pos][feat][twin_features[feat]]={'pos':0,'neg':0}
                            if ante==true_ante:
                                raw_counts_twin_features[pos][feat][twin_features[feat]]['pos']+=1
                            else:
                                raw_counts_twin_features[pos][feat][twin_features[feat]]['neg']+=1
                    if twin_mode=='test':   #get the feature weights          
                        for feat in twin_features:
                            weights_1st,weights_2nd=[],[]
                            for feat in twin_features:
                                if twin_features[feat] in twin_weights_1st[pos][feat] and twin_features[feat] in twin_weights_2nd[pos][feat]:
                                    weights_1st.append(twin_weights_1st[pos][feat][twin_features[feat]])
                                    weights_2nd.append(twin_weights_2nd[pos][feat][twin_features[feat]])                                    
                        if reduce(operator.mul,weights_1st)< reduce(operator.mul,weights_2nd):
                        #if sum(weights_2nd)>sum(weights_1st):
                            ante=weighted_antes[1][1]   

        
        #index of the true ante
        """
        true_ante=get_true_ante(mable,ante_cands,ante_cands_csets)
        if true_ante!=[]:
            rank=next(weighted_antes.index(x) for x in weighted_antes if x[1]==true_ante)
            lemma=re.search('[^|]+',mable[-1]).group()
            eval_ante_index[mable[4]][lemma][rank]+=1
            #eval_ante_index[mable[4]][mable[-1]].append(next(weighted_antes.index(x) for x in weighted_antes if x[1]==true_ante))        
        """                                                      
        if trace_errors:
            #if float(re.search('\d+\.\d+',docid).group()) >= 990505.49 and mable[4]=='PPER':
            if not mable[4] in ['NE','NN']:
                true_ante=get_true_ante(mable,ante_cands,ante_cands_csets)  
                if not ante==true_ante:
                    print ''
                    print mable
                    print 'true:\t',true_ante            
                    print 'sel:\t',ante
                    for x in all_antes:
                        print x
                    for x in weighted_antes[:2]:
                        print x                        
                    print ''
                    if true_ante!=[]:
                        rank=next(weighted_antes.index(x) for x in weighted_antes if x[1]==true_ante)
                        lemma=re.search('[^|]+',mable[-1]).group()
                        eval_ante_index[mable[4]][lemma][rank]+=1
                        pdb.set_trace()     

    if classifier=='wapiti':
        if wapiti_mode=='sequence': # model sequence of candidates and pronoun
            wapiti_cmd=''
            for a in reversed(all_antes):
                wapiti_cmd+=str(a[0])+'\t'                      #mable id
                wapiti_cmd+=mable[13]+'/'+str(mable[1]-a[1])+'\t'             #connector+sent_dist
                wapiti_cmd+=str(mable[1]-a[1])+'/'+str(mable[0]-a[0])+'\t'             #mable_dist with sentence distance  
                wapiti_cmd+=str(all_antes.index(a))+'\t'         #candidate index    
                wapiti_cmd+=a[8].lower()+'\t'                          #gf ante           
                if a[8]=='PN': 
                    try:
                        wapiticmd+=prepositions[a[1],a[2]]+'\t'
                    except:     #coordinations and PPOSATs sometimes
                        wapiti_cmd+='-\t'       
                else: wapiti_cmd+='-\t'            
                wapiti_cmd+=str(mable[1]-a[1])+'_'+a[8]+'_'+mable[8]+'_'+a[4]+'\t'             #gf seq
                if a in ante_cands_csets: wapiti_cmd+='old\t'      #discourse status ante
                else: wapiti_cmd+='new\t'         
                wapiti_cmd+=a[9]+'_'+a[6]+'_'+a[7]+'\t'                           #ante animacy
                wapiti_cmd+=a[4]+'/'+a[12]+'\t'                      #ne class            
                wapiti_cmd+=a[6]+'\t'                      #gender            
                wapiti_cmd+=a[7]+'\t'                      #number   
                if a in ante_cands_csets:                   #entity introduction sentence
                    ante_cset=next(c for c in csets if a in c)
                    wapiti_cmd+=str(ante_cset[0][1]-mables[0][1])+'\t'
                else: wapiti_cmd+=str(a[1])+'\t'
                
                if (a[1],a[10]) in verbs:   #subclause type
                    ante_subclause=verbs[a[1],a[10]]['verb'][7]
                    wapiti_cmd+=verbs[a[1],a[10]]['verb'][7]+'\t'
                else:
                    ante_subclause='-'
                    wapiti_cmd+='-\t'    
                    
                sent_dist='0' if mable[1]==a[1] else '1'
                if (mable[1],mable[10]) in verbs:   #subclause type sequence
                    wapiti_cmd+=sent_dist+'_'+ante_subclause+'_'+verbs[mable[1],mable[10]]['verb'][7]+'\t'
                else:                
                    wapiti_cmd+=sent_dist+'_'+ante_subclause+'_*\t'
                                                    
                if mable[4]=='PPOSAT':
                    same_head='same_head' if a[1]==mable[1] and a[10]==mable[10] else 'not_same_head'
                    if (mable[1],mable[2]) in pposat_heads:
                        pposat_head=pposat_heads[mable[1],mable[2]]
                        wapiti_cmd+='_'.join([str(mable[1]-a[1]),a[8],pposat_head[7].upper(),a[4]])+'\t'    #possessed gf sequence
                        if same_head=='same_head':
                            wapiti_cmd+='_'.join([a[8],pposat_head[7].upper(),a[4]])+'\t'   #same verb gf sequence
                        else: wapiti_cmd+='-\t'  #dummy features
                    else: wapiti_cmd+='-\t-\t'                            
                else: wapiti_cmd+='-\t-\t'
                                     
                if mode=='train':
                    if a!=ante: 
                        wapiti_cmd+='no\n'            #coref
                    else: 
                        wapiti_cmd+='ante\n' 
                else:
                    wapiti_cmd+='?\n'      
            wapiti_cmd+='\n'
            


            if mode=='train':                
                if mable[4]=='PPER': # add to training instances
                    pper_train.append(wapiti_cmd)     
                elif mable[4]=='PPOSAT': 
                    pposat_train.append(wapiti_cmd)        
                elif mable[4]=='PRELS': 
                    prels_train.append(wapiti_cmd)        
                elif mable[4]=='PRELAT': 
                    prelat_train.append(wapiti_cmd)                            
                elif mable[4]=='PDS': 
                    pds_train.append(wapiti_cmd)                            
                else: 
                    print 'mable type problem',mable
                    pdb.set_trace()                                 
    
            if mode=='test':
                wapiti_res=[]
                if mable[4]=='PPER':
                    wapiti_pper.stdin.write(wapiti_cmd)
                    #wapiti_pper.stdin.write('\n')
                    while True:
                        out = wapiti_pper.stdout.readline()
                        wapiti_res.append(out.strip())
                        if out=='\n':
                            break               
                elif mable[4]=='PPOSAT':
                    wapiti_pposat.stdin.write(wapiti_cmd)
                    #wapiti_pper.stdin.write('\n')                    
                    while True:
                        out = wapiti_pposat.stdout.readline()
                        wapiti_res.append(out.strip())
                        if out=='\n':
                            break         
                elif mable[4]=='PRELS':
                    wapiti_prels.stdin.write(wapiti_cmd)
                    #wapiti_pper.stdin.write('\n')                    
                    while True:
                        out = wapiti_prels.stdout.readline()
                        wapiti_res.append(out.strip())
                        if out=='\n':
                            break   
                elif mable[4]=='PRELAT':
                    wapiti_prelat.stdin.write(wapiti_cmd)
                    while True:
                        out = wapiti_prelat.stdout.readline()
                        wapiti_res.append(out.strip())
                        if out=='\n':
                            break                               
                elif mable[4]=='PDS':
                    wapiti_pds.stdin.write(wapiti_cmd)
                    #wapiti_pper.stdin.write('\n')                    
                    while True:
                        out = wapiti_pds.stdout.readline()
                        wapiti_res.append(out.strip())
                        if out=='\n':
                            break                                         
                else:
                    print 'no wapiti classifier for ',mable[4] 

                wapiti_ante_weights=[]         
                for line in wapiti_res:    #last line is newline, second last is pronoun
                    if re.match('^\d',line):
                        line=line.split('\t')
                        if line[-1].startswith('no'):
                            w=-float(line[-1].split('/')[-1])   #negative weight
                            wapiti_ante_weights.append([w,int(line[0])])
                        elif line[-1].startswith('ante'):
                            w=float(line[-1].split('/')[-1])
                            wapiti_ante_weights.append([w,int(line[0])])
                wapiti_ante_weights.sort(reverse=True)
                ante=next(a for a in all_antes if a[0]==wapiti_ante_weights[0][1])
    
        if wapiti_mode=='single':   # model pair of ante cand an pronoun
            for a in all_antes:         
                wapiti_cmd=str(a[0])+'\t'                      #mable id
                wapiti_cmd+=mable[13]+'/'+str(mable[1]-a[1])+'\t'             #connector+sent_dist
                wapiti_cmd+=str(mable[1]-a[1])+'/'+str(mable[0]-a[0])+'\t'             #mable_dist with sentence distance  
                wapiti_cmd+=str(all_antes.index(a))+'\t'         #candidate index    
                wapiti_cmd+=a[8].lower()+'\t'                          #gf ante           
                if a[8]=='PN': 
                    try:
                        wapiticmd+=prepositions[a[1],a[2]]+'\t'
                    except:     #coordinations and PPOSATs sometimes
                        wapiti_cmd+='-\t'       
                else: wapiti_cmd+='-\t'            
                wapiti_cmd+=str(mable[1]-a[1])+'_'+a[8]+'_'+mable[8]+'_'+a[4]+'\t'             #gf seq
                if a in ante_cands_csets: wapiti_cmd+='old\t'      #discourse status ante
                else: wapiti_cmd+='new\t'         
                wapiti_cmd+=a[9]+'_'+a[6]+'_'+a[7]+'\t'                           #ante animacy
                wapiti_cmd+=a[4]+'/'+a[12]+'\t'                      #ne class            
                wapiti_cmd+=a[6]+'\t'                      #gender            
                wapiti_cmd+=a[7]+'\t'                      #number   
                if a in ante_cands_csets:                   #entity introduction sentence
                    ante_cset=next(c for c in csets if a in c)
                    wapiti_cmd+=str(ante_cset[0][1]-mables[0][1])+'\t'
                else: wapiti_cmd+=str(a[1])+'\t'
                
                if (a[1],a[10]) in verbs:   #subclause type
                    ante_subclause=verbs[a[1],a[10]]['verb'][7]
                    wapiti_cmd+=verbs[a[1],a[10]]['verb'][7]+'\t'
                else:
                    ante_subclause='-'
                    wapiti_cmd+='-\t'    
                    
                sent_dist='0' if mable[1]==a[1] else '1'
                if (mable[1],mable[10]) in verbs:   #subclause type sequence
                    wapiti_cmd+=sent_dist+'_'+ante_subclause+'_'+verbs[mable[1],mable[10]]['verb'][7]+'\t'
                else:                
                    wapiti_cmd+=sent_dist+'_'+ante_subclause+'_*\t'
                                                    
                if mable[4]=='PPOSAT':
                    same_head='same_head' if a[1]==mable[1] and a[10]==mable[10] else 'not_same_head'
                    if (mable[1],mable[2]) in pposat_heads:
                        pposat_head=pposat_heads[mable[1],mable[2]]
                        wapiti_cmd+='_'.join([str(mable[1]-a[1]),a[8],pposat_head[7].upper(),a[4]])+'\t'    #possessed gf sequence
                        if same_head=='same_head':
                            wapiti_cmd+='_'.join([a[8],pposat_head[7].upper(),a[4]])+'\t'   #same verb gf sequence
                        else: wapiti_cmd+='-\t'  #dummy features
                    else: wapiti_cmd+='-\t-\t'                            
                else: wapiti_cmd+='-\t-\t'
                
                if mode=='train':
                    if a!=ante: 
                        wapiti_cmd+='no\n'            #coref
                    else: 
                        wapiti_cmd+='ante\n'       
                            
                    if mable[4]=='PPER': # add to training instances
                        pper_train.append(wapiti_cmd)     
                    elif mable[4]=='PPOSAT': 
                        pposat_train.append(wapiti_cmd)        
                    elif mable[4]=='PRELS': 
                        prels_train.append(wapiti_cmd)  
                    elif mable[4]=='PRELAT': 
                        prelat_train.append(wapiti_cmd)                                 
                    elif mable[4]=='PDS': 
                        pds_train.append(wapiti_cmd)                            
                    else: 
                        print 'mable type problem',mable
                        pdb.set_trace()     
                        
                if mode=='test':
                    wapiti_cmd+='?\n'
                    wapiti_res=[]
                    if mable[4]=='PPER':
                        wapiti_pper.stdin.write(wapiti_cmd)
                        while True:
                            out = wapiti_pper.stdout.readline()
                            wapiti_res.append(out.strip())
                            if out=='\n':
                                break               
                    elif mable[4]=='PPOSAT':
                        wapiti_pposat.stdin.write(wapiti_cmd)               
                        while True:
                            out = wapiti_pposat.stdout.readline()
                            wapiti_res.append(out.strip())
                            if out=='\n':
                                break         
                    elif mable[4]=='PRELS':
                        wapiti_prels.stdin.write(wapiti_cmd)
                        while True:
                            out = wapiti_prels.stdout.readline()
                            wapiti_res.append(out.strip())
                            if out=='\n':
                                break   
                    elif mable[4]=='PRELAT':
                        wapiti_prelat.stdin.write(wapiti_cmd)
                        while True:
                            out = wapiti_prelat.stdout.readline()
                            wapiti_res.append(out.strip())
                            if out=='\n':
                                break                                   
                    elif mable[4]=='PDS':
                        wapiti_pds.stdin.write(wapiti_cmd)
                        while True:
                            out = wapiti_pds.stdout.readline()
                            wapiti_res.append(out.strip())
                            if out=='\n':
                                break                                         
                    else:
                        print 'no wapiti classifier for ',mable[4] 

                    # format the weight
                    ante_weight=float(re.search('.*(\d\.\d+)$',wapiti_res[1]).group(1))
                    if re.match('.*\tante/\d\.\d+$',wapiti_res[1]):  #classified as ante
                        wapiti_ante_weights.append([ante_weight,a])
                    else:
                        wapiti_ante_weights.append([1-ante_weight,a])

            if mode=='test':
                if not wapiti_ante_weights==[]:
                    wapiti_ante_weights.sort(reverse=True)
                    ante=wapiti_ante_weights[0][1] 
            
    return ante
    
# ============================================= # 

''' MAIN LOOP '''
      
def main(file1,file2='',file3=''): 
    """
    Main loop
    train:
    main('../tueba_files/train_ner.mables.parzu')
    test:
    main('../tueba_files/dev_ner.mables.parzu','../tueba_files/dev_9.1.conll','mle_dev_tmp.res')   #real preprocessing
    main('../tueba_files/dev_gold_prep.mables','../tueba_files/dev_9.1.conll','mle_dev_tmp.res')   #gold preprocessing
    """
    
    if mode=='train':
        if not os.path.isfile(file1):
            print >>sys.stderr,file1,'does not exist'
            return
    else:
        if not os.path.isfile(file1):
            print >>sys.stderr,file1,'does not exist'
            return
        
        if not os.path.isfile(file2):
            print >>sys.stderr,file2,'does not exist'
            return               
    
    with open(file1,'r') as f: 
        docs=f.read()
    docs=docs.split('####')
    del docs[-1]

    if mode=='train':
        if classifier=='mle':
            global raw_counts,weights
            raw_counts={}        
            weights={}
        if classifier=='wapiti':
            global prels_train,prelat_train,pper_train,pposat_train,pds_train
            prels_train,prelat_train,pper_train,pposat_train,pds_train=[],[],[],[],[]

    if mode=='test':
        global res
        res={}
        
        if classifier=='mle':
            global weights_global
            if preprocessing=='gold': weights_global=eval(open('mle_weights_tmp','r').read())
            if preprocessing=='real': weights_global=eval(open('mle_weights_real','r').read())
            global raw_counts_twin_features            
            raw_counts_twin_features={}
            
            if twin_mode=='test':
                global twin_weights_1st,twin_weights_2nd
                twin_weights_1st=eval(open('mle_weights_twin_candidates_1st','r').read())
                twin_weights_2nd=eval(open('mle_weights_twin_candidates_2nd','r').read())
            
        if classifier=='thebeast':            
            global thebeast
            thebeast=subprocess.Popen(['/home/user/tuggener/hex_storage/TheBeast/bin/thebeast'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            #thebeast.stdin.write('set solver.model.solver = "gurobi";include "mln/types.pml";include "mln/coref.pml";set solver.model.initIntegers = true;load weights from "mln/manual_weights";\n')    #use gurobi as ILP solver
            thebeast.stdin.write('include "mln/types.pml";include "mln/coref.pml";set solver.model.initIntegers = true;load weights from "mln/manual_weights";\n')    #use LPsolve as ILP solver
            while True:
                out = thebeast.stdout.readline().strip()
                if out.endswith('weights loaded.'):
                    break 
                              
        if classifier=='wapiti':
            global wapiti_pper,wapiti_pposat,wapiti_prels,wapiti_prelat,wapiti_pds
            #TODO try with and withou -p (posterior) paramater
            wapiti_pper=subprocess.Popen(['wapiti', 'label' ,'-p', '-s','-m',wapiti_path+'/pper_model'],stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            while True:
                s=wapiti_pper.stderr.readline()
                if s.endswith('Label sequences\n'):
                    break
            wapiti_pposat=subprocess.Popen(['wapiti', 'label' ,'-p', '-s','-m',wapiti_path+'/pposat_model'],stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            while True:
                s=wapiti_pposat.stderr.readline()
                if s.endswith('Label sequences\n'):
                    break    
            wapiti_prels=subprocess.Popen(['wapiti', 'label' ,'-p', '-s','-m',wapiti_path+'/prels_model'],stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)        
            while True:
                s=wapiti_prels.stderr.readline()
                if s.endswith('Label sequences\n'):
                    break 
            wapiti_prelat=subprocess.Popen(['wapiti', 'label' ,'-p', '-s','-m',wapiti_path+'/prelat_model'],stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)        
            while True:
                s=wapiti_prelat.stderr.readline()
                if s.endswith('Label sequences\n'):
                    break                     
            wapiti_pds=subprocess.Popen(['wapiti', 'label' ,'-p', '-s','-m',wapiti_path+'/pds_model'],stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)        
            while True:
                s=wapiti_pds.stderr.readline()
                if s.endswith('Label sequences\n'):
                    break 
                    
        if verb_postfilter=='test':
            global wapiti_verb_postfilter
            wapiti_verb_postfilter=subprocess.Popen(['wapiti', 'label' ,'-p', '-s','-m','wapiti_verbs/model_all'],stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)        
            while True:
                s=wapiti_verb_postfilter.stderr.readline()
                if s.endswith('Label sequences\n'):
                    break                  
    
    #nested defaultdict; store per POS and lemma tp and number of cases where true ante is present to evaluate accuracy only when the true ante is present
    #eval_all=defaultdict(lambda: defaultdict(lambda:defaultdict(lambda:defaultdict(int))))
    eval_all=defaultdict(lambda: defaultdict(lambda:defaultdict(int)))      
    subj_sel=defaultdict(list)
    global eval_ante_index
    eval_ante_index=defaultdict(lambda: defaultdict(lambda:defaultdict(int)))
    
    if mode=='train':
        print >> sys.stderr,'Training on file',file1
    if mode=='test':
        print >> sys.stderr,'Testing on file',file1
    
    doc_counter=1
                
    for doc in docs:

        #load information from preprocessing
        docid=re.search('docid= ?(.*)',doc).group(1)
        global mables            
        mables=eval(re.search('mables=(.*)',doc).group(1))
        str_match=eval(re.search('str_matches=(.*)',doc).group(1))
        global coref
        coref=eval(re.search('coref=(.*)',doc).group(1))
        global pposat_heads,prepositions
        pposat_heads=eval(re.search('pposat_heads=(.*)',doc).group(1))
        prepositions=eval(re.search('prepositions=(.*)',doc).group(1))
        global nominal_mods
        nominal_mods=eval(re.search('nominal_mods=(.*)',doc).group(1))
        global determiners
        determiners=eval(re.search('determiners=(.*)',doc).group(1))             
        global verbs
        verbs=eval(re.search('verbs=(.*)',doc).group(1))
        global wl,csets
        wl,csets=[],[]

        sys.stderr.write('\r'+'doc name: '+str(docid)+'\tdoc counter: '+str(doc_counter))
        sys.stderr.flush()
        doc_counter+=1
            
        for mable in mables:
        
            matched=0
            global ante_cands_csets,ante_cands,ante
            ante_cands_csets,ante_cands,ante=[],[],[]
            
            if mable[4] in ['NN','NE']:
                # use str_match from preprocessing
                for match in str_match:
                    if mable[0] in match and not match.index(mable[0])==0:
                        ante_id=match[match.index(mable[0])-1]
                        ante=next(m for m in mables if m[0]==ante_id)
                        update_csets(ante,mable)
                        matched=1
                        break    
                
                if trace_nouns:# and doc_counter>32:
                    #trace resolution
                    if matched==1:
                        try:
                            mable_cset=next(ID for ID,c in coref.items() if mable[1:4] in c)    #check if it is a true mention
                            ante_cset=next(ID for ID,c in coref.items() if ante[1:4] in c)
                            if mable_cset!=ante_cset:
                                print >>sys.stderr,'\n',docid
                                print >>sys.stderr,'\nWrong resolution'
                                print >>sys.stderr,mable,ante
                                #pdb.set_trace()
                        except StopIteration:
                            pass
                            print >>sys.stderr,'\n',docid                            
                            print >>sys.stderr,'\nResolved a non gold mention'
                            print >>sys.stderr,'ante:',ante,'\nmable:',mable
                            #pdb.set_trace()
                    else:
                        try:
                            mable_cset=next(ID for ID,c in coref.items() if mable[1:4] in c)
                            if not coref[mable_cset].index(mable[1:4])==0:
                                print >>sys.stderr,docid
                                print >> sys.stderr,'\n','\nUnresolved gold mention'
                                print >>sys.stderr,mable
                                print >>sys.stderr,'cset'
                                for tm in coref[mable_cset]:
                                    print >>sys.stderr,get_mable(tm) 
                                if doc_counter>50:
                                    pdb.set_trace()
                        except StopIteration:
                            pass

            if mable[4]=='PRELS':
                ante_cands=[m for m in wl if m[1]==mable[1] and m[4]!='PPOSAT' and pper_comp(m,mable) and mable[0]-m[0]<5]               
                for cset in csets: 
                    if cset[-1][1]==mable[1] and mable[0]-cset[-1][0]<5 and cset[-1][4]!='PPOSAT' and pper_comp(cset[-1],mable):
                    #if cset[-1][1]==mable[1] and mable[0]-cset[-1][0]<5 and pper_comp(cset[-1],mable):  
                        ante_cands_csets.append(cset[-1]) #most recent from cset
                ante=get_best(ante_cands,ante_cands_csets,mable,docid)
                if not ante==[]:
                    update_csets(ante,mable)
                    matched=1

            elif mable[4]=='PRELAT':
                ante_cands=[m for m in wl if m[1]==mable[1] and m[4]!='PPOSAT' and pper_comp(m,mable) and mable[0]-m[0]<5]
                for cset in csets: 
                    if cset[-1][1]==mable[1] and mable[0]-cset[-1][0]<5 and cset[-1][4]!='PPOSAT' and pper_comp(cset[-1],mable): 
                        ante_cands_csets.append(cset[-1]) #most recent from cset
                ante=get_best(ante_cands,ante_cands_csets,mable,docid)
                if not ante==[]:
                    update_csets(ante,mable)
                    matched=1
                        
            elif mable[4]=='PPER':
                resolve=1 if not mable[-1].lower()=='es' else 0 #: and mable[5]==3:     #only resolve 3rd person pronouns, exclude 'es'
                if resolve==1:
                    ante_cands=[m for m in wl if pper_comp(m,mable)]
                    for cset in csets: 
                        if pper_comp(cset[-1],mable): 
                            ante_cands_csets.append(cset[-1]) #most recent from cset
                    ante=get_best(ante_cands,ante_cands_csets,mable,docid)              
                    if not ante==[]:                
                        update_csets(ante,mable)
                        matched=1

            elif mable[4]=='PDS':
                ante_cands=[m for m in wl if mable[1]-m[1]<2 and pper_comp(m,mable)]     
                for cset in csets: 
                    if mable[1]-cset[-1][1]<2 and pper_comp(cset[-1],mable):    #restrict ante cands to previous sentence
                        ante_cands_csets.append(cset[-1]) #most recent from cset                                        
                ante=get_best(ante_cands,ante_cands_csets,mable,docid)              
                if not ante==[]:                
                    update_csets(ante,mable)
                    matched=1
            
            elif mable[4]=='PPOSAT':
                ante_cands=[m for m in wl if pposat_comp(m,mable)]   
                #insent=[m for m in mables if m[1]==mable[1] and m[2]-mable[2]>1 and pposat_comp(m,mable)]         
                #ante_cands.extend(insent)
                for cset in csets: 
                    if pposat_comp(cset[-1],mable): 
                        ante_cands_csets.append(cset[-1]) #most recent from cset
                ante=get_best(ante_cands,ante_cands_csets,mable,docid)              
                if not ante==[]:
                    update_csets(ante,mable)
                    matched=1
                        
            if matched==0: 
                wl.append(mable)        

            all_antes=ante_cands+ante_cands_csets
            if not mable[4] in ['NN','NE']:# and len(all_antes)>1:
                lexem=re.search('[^|]+',mable[-1].lower()).group()
                eval_all[mable[4]][lexem]['instances']+=1
                try:    # count true mentions, i.e. anaphoric pronouns
                    next(c for c in coref if mable[1:4] in coref[c] and not coref[c].index(mable[1:4])==0) #filter out cataphors
                    eval_all[mable[4]][lexem]['true_mention']+=1
                except StopIteration:
                    pass
                true_ante=get_true_ante(mable,ante_cands,ante_cands_csets)
                if not true_ante==[]:
                    if ante[8]=='SUBJ':subj_sel[mable[4]].append('subj')
                    else: subj_sel[mable[4]].append('other')
                    eval_all[mable[4]][lexem]['true_ante_present']+=1
                    if ante[:4]==true_ante[:4]:
                        eval_all[mable[4]][lexem]['tp']+=1
                    else:
                        eval_all[mable[4]][lexem]['wl']+=1
                """                        
                else:   #trace missing true antecedent
                    try:
                        next(c for c in coref if mable[1:4] in coref[c] and not coref[c].index(mable[1:4])==0) #it's a gold mention
                        print >>sys.stderr,'\ntrue ante not present\n',mable
                        print >>sys.stderr,ante_cands
                        print >>sys.stderr,ante_cands_csets                        
                        if mable[-1]=='sie': pdb.set_trace()
                    except StopIteration:
                        pass
                """
            elif mable[4] in ['NN','NE']:
                try:
                    cs_gold=next(c for c in coref if mable[1:4] in coref[c] and not coref[c].index(mable[1:4])==0 )
                    eval_all[mable[4]]['ALL']['true_mention']+=1
                    eval_all[mable[4]]['ALL']['instances']+=1                        
                    if matched==1:
                        if ante[1:4] in coref[cs_gold]:
                            eval_all[mable[4]]['ALL']['true_ante_present']+=1
                            eval_all[mable[4]]['ALL']['tp']+=1
                        else:
                            eval_all[mable[4]]['ALL']['wl']+=1
                except StopIteration:
                    if matched==1:
                        eval_all[mable[4]]['ALL']['instances']+=1
                                                              
        #Try to append first person pronoun csets to 3rd person entities
        for cset in csets:
            if cset[0][5]==1 and cset[0][7]=='SG': #first person coreference set #TODO also allow plural pronouns
                try:
                    #find a markable that is max. 1 sentence away, has number singular, has not neutral gender, is a subject, 3rd person, before the first mention of the 1st person cset and its verb is a communication verb
                    ante=next(m for m in mables if cset[0][1]-m[1]<=1 and m[7]=='SG' and m[6] in ['MASC','FEM','*'] and m[8]=='SUBJ' and m[5]==3 and m[0]<cset[0][0] and m[11] in vdic)
                    if ante in wl:                                          #the antecedent is from the waiting list
                        cset.insert(0,ante)                                 #insert the antecedent at the begining of the coreference set
                    else:
                        ante_set=next(c for c in csets if ante in c)        #the antecedent is from the coreference partition
                        ante_set+=cset                                      #merge the sets
                        csets.remove(cset)                                  #and remove the 1st person pronoun set
                except StopIteration: True

        #try to resolve remaining first person pronouns
        for p in wl:
            if p[4]=='PPER' and p[5]==1 and p[7]=='SG': #first person pronouns
                try:
                    #find a markable that is max. 1 sentence away, has number singular, has not neutral gender, is a subject, 3rd person, before the 1st person pronoun and its verb is a communication verb
                    ante=next(m for m in mables if p[1]-m[1]<=1 and m[7]=='SG' and m[6] in ['MASC','FEM','*'] and m[8]=='SUBJ' and m[5]==3 and m[0]<p[0] and m[11] in vdic)
                    if ante in wl:
                        csets.append([ante,p])
                        wl.remove(p)
                        wl.remove(ante)                
                    else:
                        ante_set=next(c for c in csets if ante in c)
                        ante_set+=[p]
                        wl.remove(p)
                except StopIteration: True

        #TODO check mables for demostrative nouns that have not been resolved, try to insert to a cset.

        #join ne-nn sets if both animate and certain criteria are met
        if link_ne_nn_sets:
            new_csets=[]
            while not csets==[]:
                cset=csets[0]
                matches=[]
                if cset[0][4]=='NE' and cset[0][9]=='ANIM' and cset[0][7]=='SG':
                    for c in csets:
                        if c[0][4]=='NN' and c[0][9]=='ANIM' and c[0][7]=='SG' and cset[0][6]==c[0][6] and len(c)<=len(cset) and (c[0][1],c[0][2]) in determiners and determiners[c[0][1],c[0][2]] not in ['ein','eine'] and c[0][0]<=cset[-1][0] and cset[0][0]<c[0][0]:
      
                            try: #first nn mention is at most 5 sentences away from closest previous ne mention
                                next(m for m in cset if m[1]<c[0][1] and c[0][1]-m[1]<6)
                            except StopIteration:
                                continue
                                
                            try: #c binding constraint is maintained
                                next(x for x in c if not x[4]=='PPOSAT' and not x[10]=='*' and [y for y in cset if y[1]==x[1] and y[10]==x[10] and not y[4]=='PPOSAT'])
                                continue
                            except StopIteration:
                                pass                                
                            """                  
                            try:    #nn set has pronoun
                                next(p for p in c if p[4] in ['PPOSAT','PPER']) 
                            except StopIteration:
                                continue
                                
                            try:    #nn set has pronoun
                                next(p for p in c if p[4] in ['PPOSAT','PPER']) 
                            except StopIteration:
                                new_csets.append(cset)
                                continue

                            """                                
                            try:
                                cset_cid=next(i for i,j in coref.items() if cset[0][1:4] in j)
                                nn_cid=next(i for i,j in coref.items() if c[0][1:4] in j)                    
                                if not cset_cid==nn_cid:
                                    eval_link_ne_nn_sets['wrong']+=1
                                    print '\nwrong'
                                    print cset
                                    print c                                
                                    #pdb.set_trace()                                                        
                                else:
                                    eval_link_ne_nn_sets['correct']+=1
                                    print '\ncorrect'
                                    print cset
                                    print c    
                                    #pdb.set_trace()                                                        
                            except StopIteration:
                                eval_link_ne_nn_sets['false_positive']+=1   
                                print '\nfalse positive'
                                print cset
                                print c                                
                                #pdb.set_trace()     
                            #pdb.set_trace()                                             
                            cset.extend(c)
                            cset.sort()
                            matches.append(c)
                    new_csets.append(cset)
                    for c in matches:
                        csets.remove(c)
                else:
                    new_csets.append(cset)            
                csets.remove(cset)
            csets=new_csets

        if mode=='test': 
            res[docid]=[mables,csets]              

            
    # ============================================= # 

    ''' TRAINING '''            

    if mode=='train':
        
        # calculate the biased MLE weights
        if classifier=='mle':
            weights=copy.deepcopy(raw_counts)
            for pos in raw_counts:
                ratio=ante_counts[pos]/float(pronouns_counts[pos])  #class bias: antecedent candidates / pronoun ratio (how many candidates on avg?)   
                for feat in raw_counts[pos]:
                    card_feat=len(raw_counts[pos][feat])    #number of different feature values, needed for smoothing
                    for val in raw_counts[pos][feat]:
                        weight=ratio * (raw_counts[pos][feat][val]['pos']+.1) / (raw_counts[pos][feat][val]['pos']+raw_counts[pos][feat][val]['neg']+.2)
                        weights[pos][feat][val]=weight

            if preprocessing=='gold':                        
                with open('mle_weights_tmp','w') as f: 
                    f.write(str(weights)+'\n')        
                with open('mle_weights_tmp_raw_counts','w') as f: 
                    f.write(str(raw_counts)+'\n')                    
            if preprocessing=='real':                        
                with open('mle_weights_real','w') as f: 
                    f.write(str(weights)+'\n')        

        if classifier=='wapiti':                       
            with open(wapiti_path+'/pper_train','w') as f:
                for i in pper_train: f.write(i)
            with open(wapiti_path+'/pposat_train','w') as f:
                for i in pposat_train: f.write(i)
            with open(wapiti_path+'/prels_train','w') as f:
                for i in prels_train: f.write(i)    
            with open(wapiti_path+'/prelat_train','w') as f:
                for i in prelat_train: f.write(i)                    
            with open(wapiti_path+'/pds_train','w') as f:
                for i in pds_train: f.write(i)                      
                                           
    if mode=='test':
    
        if twin_mode=='train':
            if classifier=='mle':
                twin_weights_1st=copy.deepcopy(raw_counts_twin_features) 
                twin_weights_2nd=copy.deepcopy(raw_counts_twin_features)                
                for pos in raw_counts_twin_features:
                    for feat in raw_counts_twin_features[pos]:
                        for val in raw_counts_twin_features[pos][feat]:
                            weight_1st=float(raw_counts_twin_features[pos][feat][val]['pos'])/(raw_counts_twin_features[pos][feat][val]['pos']+raw_counts_twin_features[pos][feat][val]['neg'])
                            twin_weights_1st[pos][feat][val]=weight_1st
                            weight_2nd=float(raw_counts_twin_features[pos][feat][val]['neg'])/(raw_counts_twin_features[pos][feat][val]['pos']+raw_counts_twin_features[pos][feat][val]['neg'])
                            twin_weights_2nd[pos][feat][val]=weight_2nd                        
                with open('mle_weights_twin_candidates_1st','w') as f: 
                    f.write(str(twin_weights_1st)+'\n')    
                with open('mle_weights_twin_candidates_2nd','w') as f: 
                    f.write(str(twin_weights_2nd)+'\n')    
    
        if classifier=='thebeast':
            thebeast.stdin.close()
            thebeast.wait()    

        # ============================================= # 

        ''' OUTPUT '''
    
        #Output            
        f=open(file3,'w')
        docnr=0
        sent=1        

        docid='1'
        csets_orig=res[docid][1]
        docnr+=1
        csets={}
        for cset in csets_orig: #convert system response for faster output
            cset_id=csets_orig.index(cset)
            for m in cset:
                if m[2]==m[3]:
                    csets[(str(m[1]),str(m[2]),'oc')]=cset_id  #'oc' for open-close cset, swt mentions
                else:
                    if not (str(m[1]),str(m[2]),'o') in csets:
                        csets[(str(m[1]),str(m[2]),'o')]=[cset_id]  #'o' for open cset, multiple mwt mentions can start at a given token, store list of cset ids
                    else:
                        csets[(str(m[1]),str(m[2]),'o')].append(cset_id)
                    if not (str(m[1]),str(m[3]),'c') in csets: #'c' for close cset; multiple mwt mentions can close at a given token, store list of cset ids
                        csets[(str(m[1]),str(m[3]),'c')]=[cset_id]  
                    else:
                        csets[(str(m[1]),str(m[3]),'c')].append(cset_id)  #'c' for close cset                                                     
        mables=res[docid][0]    #TODO make same conversion as above, if singletons are wanted in the output, otherwise it's painfully slow

        for line in open(file2,'r').readlines():
            if line=='\n':
                sent+=1
                f.write(line)
            else:
                csets_out=[]
                line=line.strip().split('\t')                                            
                """                
                if (line[0],line[1],'oc') in csets:
                    csets_out.append('('+str(csets[(line[0],line[1],'oc')])+')')  
                if (line[0],line[1],'o') in csets:
                    for cid in csets[(line[0],line[1],'o')]:
                        csets_out.append('('+str(cid))                
                if (line[0],line[1],'c') in csets:
                    for cid in csets[(line[0],line[1],'c')]:
                        csets_out.append(str(cid)+')')                  
                """

                if output_format=='tueba':
                    if (str(sent),line[2],'oc') in csets:
                        csets_out.append('('+str(csets[(str(sent),line[2],'oc')])+')')  
                    if (str(sent),line[2],'o') in csets:
                        for cid in csets[(str(sent),line[2],'o')]:
                            csets_out.append('('+str(cid))                
                    if (str(sent),line[2],'c') in csets:
                        for cid in csets[(str(sent),line[2],'c')]:
                            csets_out.append(str(cid)+')')                 
                if output_format=='semeval':
                    if (str(sent),line[0],'oc') in csets:
                        csets_out.append('('+str(csets[(str(sent),line[0],'oc')])+')')  
                    if (str(sent),line[0],'o') in csets:
                        for cid in csets[(str(sent),line[0],'o')]:
                            csets_out.append('('+str(cid))                
                    if (str(sent),line[0],'c') in csets:
                        for cid in csets[(str(sent),line[0],'c')]:
                            csets_out.append(str(cid)+')')                                             
                """                                        
                if (line[1],line[2],'oc') in csets:
                    csets_out.append('('+str(csets[(line[1],line[2],'oc')])+')')  
                if (line[1],line[2],'o') in csets:
                    for cid in csets[(line[1],line[2],'o')]:
                        csets_out.append('('+str(cid))                
                if (line[1],line[2],'c') in csets:
                    for cid in csets[(line[1],line[2],'c')]:
                        csets_out.append(str(cid)+')')                                    
                """                        
                mables_out=[]
                
                #this is for singleton output, i.e. for SemEval
                """        
                for m in mables:
                    if len([x for x in csets if m in x])==0:    #das dauert... besser loesen
                        if m[1]==int(line[0]) and m[2]==int(line[0]) and m[3]==int(line[1]): 
                            mables_out.append('('+str(m[0]+10000)+')')                    
                        elif m[1]==int(line[0]) and m[2]==line[0]: 
                            mables_out.append('('+str(m[0]+10000))
                        elif m[1]==int(line[0]) and m[3]==line[0]: 
                            mables_out.append(str(m[0]+10000)+')')
                """
                if csets_out==[] and mables_out==[]:
                    f.write('\t'.join(line[0:-1])+'\t-\n')
                elif csets_out==[] and mables_out!=[]:
                    f.write('\t'.join(line[0:-1])+'\t'+'|'.join(mables_out))
                    f.write('\n')
                elif csets_out!=[] and mables_out==[]:
                    f.write('\t'.join(line[0:-1])+'\t'+'|'.join(csets_out))
                    f.write('\n')
                else:
                    f.write('\t'.join(line[0:-1])+'\t'+'|'.join(csets_out)+'|'+'|'.join(mables_out))
                    f.write('\n')

    sys.stderr.write('\n')

    """
    for k,v in subj_sel.items():
        print k
        c=Counter(v)
        print c
    """
    
    if mode=='test' and output_pronoun_eval:
    
        print >> sys.stderr,'\nPronoun resolution accuracy when true ante is among the candidates\n'
        
        all_tp,all_true_ante_present,all_cases,all_true_mentions=0,0,0,0            
        
        # lemma normalization
        if 'seine' in eval_all['PPOSAT']:
            for k in eval_all['PPOSAT']['seine']:
                eval_all['PPOSAT']['sein'][k]+=eval_all['PPOSAT']['seine'][k]
        del eval_all['PPOSAT']['seine']
        
        if 'ihre' in eval_all['PPOSAT']:
            for k in eval_all['PPOSAT']['ihre']:
                eval_all['PPOSAT']['ihr'][k]+=eval_all['PPOSAT']['ihre'][k]
        del eval_all['PPOSAT']['ihre']        

        accuracies=defaultdict(float)
        
        for pos in eval_all:
            if not pos in ['NN','NE']:  
                print >> sys.stderr,pos
                pos_tp,pos_true_ante_present,pos_cases,pos_true_mentions=0,0,0,0    # PoS-wide counts
                for lemma,c in eval_all[pos].items():
                    pos_tp+=c['tp']
                    pos_true_ante_present+=c['true_ante_present']
                    pos_cases+=c['instances']
                    pos_true_mentions+=c['true_mention']                        
                    
                    # print eval for selected PoS and lemmas
                    if pos in ['PPER','PPOSAT'] and lemma in ['er','sie','sein','ihr']:    
                        print >> sys.stderr,lemma+'\t', 
                        if not c['true_ante_present']==0:
                            print >>sys.stderr,"%.2f" % (100*float(c['tp'])/c['true_ante_present']),'% ('+str(c['tp'])+')\t\t',
                            print >>sys.stderr,'true ante present in '+str(c['true_ante_present'])+' of '+str(c['instances'])+' cases ('+"%.2f" % (100*float(c['true_ante_present'])/c['instances'])+'%),',
                            if not pos_true_mentions==0:
                                print >>sys.stderr,c['true_mention'],'true mentions ('+"%.2f" % (100*float(pos_true_ante_present)/pos_true_mentions)+'%)'
                            else:
                                print >>sys.stderr,c['true_mention'],'true mentions'
                        else:
                            print >>sys.stderr,'\t\tno true ante in '+str(c['instances'])+' cases,',c['true_mention'],'true mentions'
                acc=(100*float(pos_tp)/pos_true_ante_present) if not pos_true_ante_present==0 else 0.
                accuracies[pos]=acc                            
                print >>sys.stderr,'ALL:\t',"%.2f" % (acc),'% ('+str(pos_tp)+')\t\t',
                #print >>sys.stderr,'ALL:\t',"%.2f" % (100*float(pos_tp)/pos_true_ante_present),'% ('+str(pos_tp)+')\t\t',
                print >>sys.stderr,'true ante present in '+str(pos_true_ante_present)+' of '+str(pos_cases)+' cases ('+"%.2f" % (100*float(pos_true_ante_present)/pos_cases)+'%),',
                acc2=100*float(pos_true_ante_present)/pos_true_mentions if not pos_true_mentions==0 else 0
                print >>sys.stderr,pos_true_mentions,'true mentions ('+"%.2f" % (acc2)+'%)'             
                all_tp+=pos_tp
                all_true_ante_present+=pos_true_ante_present
                all_cases+=pos_cases
                all_true_mentions+=pos_true_mentions
                print >>sys.stderr,''                
                
        accuracies['ALL']=100*float(all_tp)/all_true_ante_present                
        print >>sys.stderr,'OVERALL:\t',"%.2f" % (100*float(all_tp)/all_true_ante_present),'% ('+str(all_tp)+')\t\t',
        print >>sys.stderr,'true ante present in '+str(all_true_ante_present)+' of '+str(all_cases)+' cases ('+"%.2f" % (100*float(all_true_ante_present)/all_cases)+'%),',
        print >>sys.stderr,all_true_mentions,'true mentions ('+"%.2f" % (100*float(all_true_ante_present)/all_true_mentions)+'%)'

        print '\nLaTeX output:'
        for pos in ['PPER','PPOSAT','PRELS','PDS','PRELAT','ALL']: print '&', "%.2f" % accuracies[pos],
        print '\\\\' 
        
        #cPickle.dump(single_ante_counts,open('single_ante_counts.cpkl','w'))
        #cPickle.dump(avg_ante_counts,open('avg_ante_counts.cpkl','w'))  

        # Print rank index frequency of the true antecedent
        """
        # Sum rank counts for 3rd person pronouns
        eval_ante_index['PPER']['3rd']=copy.deepcopy(eval_ante_index['PPER']['er'])
        for k,v in eval_ante_index['PPER']['sie'].items(): eval_ante_index['PPER']['3rd'][k]+=v
        eval_ante_index['PPOSAT']['3rd']=copy.deepcopy(eval_ante_index['PPOSAT']['sein'])
        for k,v in eval_ante_index['PPOSAT']['ihr'].items(): eval_ante_index['PPOSAT']['3rd'][k]+=v        
        for pos in eval_ante_index: 
            print >> sys.stderr,pos
            for lexem in eval_ante_index[pos]:
                print >> sys.stderr,lexem        
                for r,c in eval_ante_index[pos][lexem].items():
                    print >> sys.stderr,str(r+1)+':',c,
                print >> sys.stderr,''                
            print >> sys.stderr,''            
        """ 
           
    if verb_postfilter=='train':   
        f=open('verb_postfilter.arff','w')
        f.write('@RELATION verb_sel_pref\n\n')
        for n in verb_postfilter_feature_names: 
            f.write('@ATTRIBUTE '+n+' NUMERIC\n')
        f.write('@ATTRIBUTE class {pos,neg}\n\n')
        f.write('@DATA\n')
        f.write('\n'.join(verb_postfilter_str))
        f.close()        
        print >>sys.stderr,'verb_postfilter_cnt:',verb_postfilter_cnt
    if verb_postfilter=='test':
        tmp=wapiti_verb_postfilter.communicate()    #close it
        print >>sys.stderr,'verb_postfilter_cnt:',verb_postfilter_cnt

    if link_ne_nn_sets:
        print >>sys.stderr,'eval_link_ne_nn_sets',dict(eval_link_ne_nn_sets)

    if verb_postfilter_context_w2v or verb_postfilter_context_graph:
        print >>sys.stderr,'\n'
        for pos,lemmas in accuracy.items():
            print >>sys.stderr,pos
            for lemma,counts in lemmas.items():
                print >>sys.stderr,'\t',lemma,'\t',
                for k in sorted(counts): print >>sys.stderr,k+':',counts[k],'\t',
                print >>sys.stderr,'total:',sum(counts.values())
                print >>sys.stderr,'\t\tbaseline acc:',"%.2f" % (100.*float(counts['baseline correct']+counts['both correct']) / sum(counts.values())),'%',
                print >>sys.stderr,'verb_sel acc:',"%.2f" % (100.*float(counts['verb_sel correct']+counts['both correct']) / sum(counts.values())),'%',
                print >>sys.stderr,'upper bound:',"%.2f" % (100.*float(counts['baseline correct']+counts['both correct']+counts['verb_sel correct']) / sum(counts.values())),'%'

# ============================================= #    

if __name__ == '__main__':
    if mode=='train': main(sys.argv[1])
    else: main(sys.argv[1],sys.argv[2],sys.argv[3])
