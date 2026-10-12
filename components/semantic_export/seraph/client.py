"""Explicitly authorized compatible HTTP Chat calls. No model download/start."""
from __future__ import annotations
import os,urllib.request,urllib.parse,urllib.error,ipaddress
from pathlib import Path
from .common import *
from .ssq import validate

def config(path):
    c=load(path)
    keys={'schema','endpoint','model','api_key_env','max_calls','timeout_seconds','max_tokens','temperature','json_object_response_format'}
    if set(c)!=keys or c['schema']!='seraph_chat_endpoint_v1':raise ValueError('LLM config schema mismatch')
    u=urllib.parse.urlparse(c['endpoint']);local=u.hostname=='localhost'
    try:local=local or ipaddress.ip_address(u.hostname or '').is_loopback
    except ValueError:pass
    if u.scheme not in ('https','http') or not u.hostname or u.username or u.password or u.query or u.fragment:raise ValueError('Invalid endpoint')
    if u.scheme=='http' and not local:raise ValueError('Remote endpoint requires HTTPS; loopback HTTP only')
    if not isinstance(c['model'],str) or not c['model'].strip() or 'REPLACE_' in c['model']:raise ValueError('Configure the ACTUAL served model; no model is assumed')
    for k,lo,hi in [('max_calls',1,10000),('timeout_seconds',1,1800),('max_tokens',32,16384)]:
        if type(c[k]) is not int or not lo<=c[k]<=hi:raise ValueError('Invalid budget '+k)
    if c['temperature']!=0 or type(c['json_object_response_format']) is not bool or not isinstance(c['api_key_env'],str):raise ValueError('Invalid protocol options')
    return c

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):raise ValueError('Redirect rejected: do not forward code/key to another endpoint')

def verified(prepared):
    if load(prepared/'state.json')['status'] not in ('PREPARED','PREPARED_WITH_GAPS'):raise ValueError('Prepare this exact tag first')
    verify_outputs(load(prepared/'prepare_checkpoint.json'))

def recommendations(out,session):
    dest=out/'llm'/tag_ok(session);plan=load(dest/'plan.json')
    if plan['requests_sha256']!=sha(out/'ssq_requests.jsonl'):raise ValueError('Requests changed')
    result={}
    for r in jsonl(out/'ssq_requests.jsonl'):
        d=dest/r['request_id']
        if not (d/'result.json').exists():continue
        x=load(d/'result.json')
        if x['status']!='VALIDATED':continue
        if sha(d/'response.json')!=x['response_sha256']:raise ValueError('Response changed')
        result[r['group_id']]=validate(r,load(d/'response.json'))
    return result

def ask(out,config_path,session,allow):
    if not allow:raise ValueError('No code transmitted. Explicit --allow-send-code is required.')
    verified(out);c=config(config_path);reqs=list(jsonl(out/'ssq_requests.jsonl'))
    if not reqs:raise ValueError('No eligible SSQ requests; inspect selection report')
    dest=out/'llm'/tag_ok(session);dest.mkdir(parents=True,exist_ok=True);plan={'config':c,'requests_sha256':sha(out/'ssq_requests.jsonl')}
    if (dest/'plan.json').exists() and load(dest/'plan.json')!=plan:raise ValueError('Session config changed; choose an explicit new --session (may cost again)')
    dump(dest/'plan.json',plan);n=sum(1 for _ in dest.glob('*/attempt.json'));key=os.environ.get(c['api_key_env'],'') if c['api_key_env'] else '';opener=urllib.request.build_opener(NoRedirect())
    print('ENDPOINT='+c['endpoint']+' MODEL='+c['model']+' TOTAL_SESSION_BUDGET='+str(c['max_calls']),flush=True)
    for r in reqs:
        d=dest/r['request_id']
        if (d/'attempt.json').exists():continue
        if n>=c['max_calls']:break
        d.mkdir();dump(d/'attempt.json',{'status':'RESERVED','request_id':r['request_id'],'time':stamp(),'prompt_sha256':r['prompt_sha256']});n+=1
        body={'model':c['model'],'messages':r['messages'],'temperature':0,'max_tokens':c['max_tokens']}
        if c['json_object_response_format']:body['response_format']={'type':'json_object'}
        headers={'Content-Type':'application/json'}
        if key:headers['Authorization']='Bearer '+key
        try:
            request=urllib.request.Request(c['endpoint'],canonical(body).encode('utf-8'),headers,method='POST')
            with opener.open(request,timeout=c['timeout_seconds']) as response:
                raw=response.read((2<<20)+1)
                if len(raw)>2<<20:raise ValueError('Response too large')
            outer=loads(raw.decode());dump(d/'raw_response.json',outer);choice=outer['choices'][0]
            if choice.get('finish_reason') not in (None,'stop'):raise ValueError('Incomplete completion')
            obj=loads(choice['message']['content']);v=validate(r,obj)
            dump(d/'response.json',obj);dump(d/'validated.json',v);dump(d/'result.json',{'status':'VALIDATED','response_sha256':sha(d/'response.json'),'usage':outer.get('usage')})
            print('[LLM VALIDATED]',r['request_id'],flush=True)
        except Exception as e:
            error=str(e).replace(key,'[REDACTED]') if key else str(e)
            dump(d/'result.json',{'status':'FAILED','error_type':type(e).__name__,'error':error[:1000]})
            print('[LLM FAILED]',type(e).__name__,r['request_id'],flush=True)
            if isinstance(e,(urllib.error.URLError,TimeoutError)):break
    valid=recommendations(out,session);write_jsonl(dest/'recommendations.jsonl',list(valid.values()));attempts=list(dest.glob('*/attempt.json'));bad=len(attempts)-len(valid)
    summary={'status':'LLM_RESPONSES_VALIDATED' if not bad else 'PARTIAL_OR_FAILED','attempted':len(attempts),'validated':len(valid),'failed_or_incomplete':bad,'remaining_not_called':len(reqs)-len(attempts),'abstained':sum(r['abstain'] for r in valid.values()),'calibrated':False,'original_scores_modified':False,'closed_loop_SERAPH_completed':False}
    dump(dest/'summary.json',summary);print(canonical(summary),flush=True);return 0 if valid and not bad else 2
