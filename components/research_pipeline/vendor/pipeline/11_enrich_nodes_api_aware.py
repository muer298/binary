#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Enrich selected USG nodes with API-aware fields in attrs.

Adds normalized API name, API family, semantic-template marker and call-like flags.
This is a minimal feature-layer improvement that can be consumed by the existing
06_featurize_objects.py script.
"""
import argparse, json, re
from pathlib import Path
from collections import Counter

API_FAMILY={
 'memory_alloc':['malloc','calloc','realloc','operator.new','new'],
 'memory_free':['free','operator.delete','delete'],
 'memory_copy':['memcpy','memmove','memset','bcopy','bzero'],
 'string':['strlen','strcmp','strncmp','strcpy','strncpy','strstr','strchr','atoi','strtol'],
 'format_io':['printf','fprintf','sprintf','snprintf','scanf','sscanf','puts','putchar'],
 'file_io':['open','read','write','close','fopen','fread','fwrite','fclose','lseek'],
 'math':['sqrt','sqrtf','pow','sin','cos','tan','exp','log','floor','ceil'],
 'process':['exit','abort','system','exec','fork'],
 'ctype':['isdigit','isalpha','isalnum','tolower','toupper'],
 'crypto':['md5','sha','aes','des','camellia']}
BAD_API={'_start','_init','_fini','main','__libc_start_main','__gmon_start__','__do_global_dtors_aux','register_tm_clones','deregister_tm_clones','frame_dummy','__cxa_finalize','__stack_chk_fail'}
def norm_name(x):
    if not x: return ''
    s=str(x).strip().replace('@plt','').replace('.plt','')
    s=re.sub(r'@@?GLIBC[^,\s]*','',s); s=re.sub(r'\s+','',s)
    return s
def family_of(name):
    s=norm_name(name).lower()
    if not s: return ''
    for fam,keys in API_FAMILY.items():
        for k in keys:
            if k.lower() in s: return fam
    return 'other_api'
def collect_api_like_strings(attrs):
    vals=[]
    if not isinstance(attrs,dict): return vals
    for k in ['api_name','api_target','call_target','callee','function_name','func_name','semantic_template','api_template','template','name']:
        v=attrs.get(k)
        if v:
            if isinstance(v,list): vals.extend(str(x) for x in v if x)
            else: vals.append(str(v))
    return vals
ap=argparse.ArgumentParser()
ap.add_argument('--nodes', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--stats', required=True)
args=ap.parse_args()
cnt=Counter(); fam_cnt=Counter(); bad_cnt=Counter()
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
with Path(args.nodes).open('r',encoding='utf-8') as fin, Path(args.out).open('w',encoding='utf-8') as fout:
    for line in fin:
        if not line.strip(): continue
        n=json.loads(line); attrs=n.get('attrs',{})
        if not isinstance(attrs,dict): attrs={}; n['attrs']=attrs
        cnt['nodes']+=1; node_type=n.get('node_type','')
        api_strings=collect_api_like_strings(attrs)
        text=' '.join(api_strings+[json.dumps(attrs, ensure_ascii=False)[:500]]).lower()
        is_call_like=(str(attrs.get('opcode','')).upper()=='CALL' or 'call' in text or 'callee' in text)
        is_api_node=(node_type=='APISem' or bool(api_strings) or is_call_like)
        if is_api_node: cnt['api_related_nodes']+=1; attrs['api_aware_is_api_related']=True
        if is_call_like: cnt['call_like_nodes']+=1; attrs['api_aware_is_call_like']=True
        api_name=''
        for s in api_strings:
            ns=norm_name(s)
            if ns: api_name=ns; break
        if api_name:
            cnt['api_name_present']+=1; attrs['api_aware_normalized_api_name']=api_name
            fam=family_of(api_name); attrs['api_aware_family']=fam; fam_cnt[fam]+=1
            for bad in BAD_API:
                if bad.lower() in api_name.lower(): attrs['api_aware_runtime_helper']=True; bad_cnt[bad]+=1
        sem=attrs.get('semantic_template') or attrs.get('api_template') or attrs.get('template')
        if sem: cnt['semantic_template_present']+=1; attrs['api_aware_semantic_template']=sem
        if node_type=='APISem': cnt['apisem_nodes']+=1; attrs['api_aware_node_is_apisem']=True
        fout.write(json.dumps(n, ensure_ascii=False)+'\n')
stats={'input':args.nodes,'output':args.out,'counts':dict(cnt),'api_family_counts':dict(fam_cnt),'runtime_helper_counts':dict(bad_cnt)}
Path(args.stats).parent.mkdir(parents=True, exist_ok=True)
Path(args.stats).write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps(stats, indent=2, ensure_ascii=False))
