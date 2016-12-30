# -*- coding: utf-8 -*-
"""
travers ParZu parse
get main verb for each token
get subcategorisation frame for each verb
"""

import pdb
import sys
import cPickle

#global vvpp_sein_exclussive
#vvpp_sein_exclussive=cPickle.load(open('vvpp_sein_exclussive.cpkl','r'))

def get_gov(tok, sent, mode = 'active'):
    ''' return the main verb governing a token '''
    if tok[4] in ['VVPP', 'VVFIN', 'VVINF']:    # main verb
        return tok, mode
    elif tok[4] in ['VAFIN', 'VMINF', 'VAINF', 'VMFIN', 'VAPP']: # aux / modal verb (hat geschlafen; kann / sollte schlafen)
        try:
            dep = next( t for t in sent if t[6] == tok[0] and t[7] == 'aux' and t[4].startswith('V') )      
            if dep[4] in ['VAINF', 'VAFIN']:
                if dep[2] in ['werden', 'werden%passiv']:
                    mode = 'passive'
                    return get_gov(dep, sent, mode)            
                elif tok[2] in ['sein', 'sein%aux']:
                    if dep[4] == 'VVPP':
                        return  dep, 'active' # er ist gegangen.. problematic for verbs, ok for governor token id exclussiveness
                    else:
                        if sent[int(tok[6])-1][4] in ['VMFIN', 'VMINF']: # hätte sein können
                            return tok, mode
                        if dep[2] in ['haben', 'haben%aux']:    # ist zu haben
                            return tok, mode
                        #print >> sys.stderr,'sein als verb', sent,tok
                        #pdb.set_trace() # do we ever get here?
                        return get_gov(dep, sent, mode)
                else:
                    return get_gov(dep, sent, mode)    
            else:
                if tok[2] in ['werden', 'werden%passiv']:
                    mode = 'passive'
                return get_gov(dep, sent, mode)                                      
        except:
            if tok[4] in ['VAFIN', 'VAINF']:
                if tok[2] in ['haben', 'haben%aux']:
                    try:    # X has Y, there is a direct object
                        obja = next(t for t in sent if t[6] == tok[0] and t[7] == 'obja')
                        return tok, mode
                    except StopIteration:
                        return  # trace it?
                elif tok[2] in ['sein', 'sein%aux']:
                    try:    # from debugging, could change this to return tok, mode
                        next(t for t in sent if t[6] == tok[0] and t[7] in ['pred', 'objg'])
                        return tok, mode
                    except StopIteration:
                        return tok,mode  
    elif tok[4] == 'VVIZU': # Ein Konkurs wäre nicht auszuschliessen
        return tok, 'passive'                                                   
    else:
        if tok[6] == '0':
            return
        gov = sent[int(tok[6])-1]
        return get_gov(gov, sent, mode)
        

    """    
    if tok[4].startswith('VV'): # full verb
        return tok, mode
        
    elif tok[4].startswith('VA'): # aux verb (hat geschlafen)
        if tok[2] in ['haben', 'haben%aux']:
            try:
                dep = next( t for t in sent if t[6] == tok[0] and t[7] == 'aux' and t[4].startswith('V') ) 
                return get_gov(dep, sent, mode)
            except StopIteration:
                return
                try:    # X has Y
                    obja = next(t for t in sent if t[6] == tok[0] and t[7] == 'obja')
                    return tok, mode
                except StopIteration:
                    return                
        elif tok[2] in ['sein', 'sein%aux', 'werden', 'werden%passiv']:
            try:
                dep = next( t for t in sent if t[6] == tok[0] and t[7] == 'aux' and t[4].startswith('V') )
                if dep[4] == 'VVPP':
                    if tok[2] in ['sein', 'sein%aux'] and dep[2] in vvpp_sein_exclussive:  # Er ist gekommen vs. Er ist geschlagen -> statistics
                        mode = 'active'
                    else:
                        mode = 'passive'
                else:
                    mode = 'passive'
                return get_gov(dep, sent, mode)
            except StopIteration:
                return
    
    elif tok[4].startswith('VM'): # modal verb (sollte schlafen)
        try:
            dep = next( t for t in sent if t[6] == tok[0] and t[7] == 'aux' and t[4].startswith('V') ) 
            return get_gov(dep, sent, mode)
        except StopIteration:
            return
                                                       
    elif tok[6] == '0': # reached root, no verb found
        return
        
    else:
        gov = sent[int(tok[6])-1]
        return get_gov(gov, sent, mode)    
    """    

def get_subcat(frames,sent):
    ''' return the adjusted subcategorisation frame of a verb based on its arguments '''
    subcats = {}
    for frame in frames:
        frame_tok_id = str(frame[1])
        subcat = list( set( frames[frame].keys()+[t[7] for t in sent if t[6] == frame_tok_id] ) )
        if 'subj' in subcat:
            subcat.remove('subj')   # we assume subject for every verb
        if 'objd' in subcat and 'objd' in frames[frame] and frames[frame]['objd'][4] == 'PRF':  # reflexives
            subcat.remove('objd')
            subcat.append('prf')             
        if 'obja' in subcat and 'obja' in frames[frame] and frames[frame]['obja'][4] == 'PRF':
            subcat.remove('obja')
            subcat.append('prf')        
        if 'objc' in subcat:        # treat sentence clause object as direct object
            subcat.remove('objc')
            if not 'obja' in subcat:
                subcat.append('obja')   
        if 's' in subcat:        # treat sentence object as direct object
            subcat.remove('s')
            if not 'obja' in subcat:
                subcat.append('obja')               
        if 'obji' in subcat:        # treat sentence object as direct object
            subcat.remove('obji')
            if not 'obja' in subcat:
                subcat.append('obja')   
        subcat = [gf for gf in subcat if gf in ['subj', 'obja', 'pred', 'objd', 'objp', 'prf', 'objg']]  
        lemma=sent[int(frame[1])-1][2].replace('#','')  #verb string
        verb=sent[int(frame[1])-1]  #verb token
        frames[frame]['verb']=verb
        if not subcat==[]:
            lemma+='_'+'_'.join(sorted(subcat,reverse=True))
        frames[frame]['subcat']=lemma 
        subcats[frame]=frames[frame]
    return subcats
