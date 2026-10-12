import ast,csv,copy,json,os,sys,sqlite3,subprocess,tempfile,shutil,threading,time,unittest
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from seraph.common import *
from seraph.source import admit
from seraph.baseline import clean,unit_note
from seraph.ssq import request,validate
from seraph.client import ask,config
from seraph.evaluation import evaluate

FAKE=r'''#!PYTHON
import sys,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
if (root/'fail').exists():
 print('ERROR REPORT SCRIPT ERROR: mock error');raise SystemExit(0)
a=sys.argv;k=a.index('-postScript');out,sid,entriesfile,timeout=a[k+2:k+6];entries=[x.strip() for x in Path(entriesfile).read_text().splitlines() if x.strip()]
functions=[]
for fn in entries:
 text='return p->y;';pc='0x%x'%(int(fn,16)+4)
 functions.append({'sample_id':sid,'function_entry':fn,'status':'OK','lines':[{'line':1,'text':text,'tokens':[{'text':text,'start_char':0,'end_char':len(text),'anchors':[{'address':pc,'kind':'pcode_target'}]}]}]})
Path(out).write_text(json.dumps({'schema':'seraph_code_export_v7_3_0','sample_id':sid,'completed':True,'image_base':'0x400000','requested_entries':entries,'functions':functions}))
with (root/'calls').open('a') as f:f.write(sid+'\n')
print('SERAPH_CODE_EXPORT_DONE',sid,'functions',len(functions))
'''

def fixture(root):
 run=root/'run';prev=run/'downstream_readiness_v7_2_1/all_scopefix'
 for p in [run/'gold_zone',run/'checkpoints',prev/'checkpoints']:p.mkdir(parents=True,exist_ok=True)
 (run/'.pipeline.lock').touch();(prev/'.audit.lock').touch()
 src=root/'tiny.c';src.write_text('struct S { long x; long y; }; __attribute__((noinline)) long f(struct S*p){return p->y;} int main(void){struct S x={1,2};return f(&x);}')
 full=root/'full';binary=root/'stripped'
 subprocess.run(['gcc','-g','-O0','-fno-pie','-no-pie',str(src),'-o',str(full)],check=True,capture_output=True)
 shutil.copy2(full,binary);subprocess.run(['strip','--strip-all',str(binary)],check=True,capture_output=True)
 ids=['s%d'%i for i in range(4)]
 with (run/'gold_zone/pairs.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['sample_id','stripped_path','full_path']);w.writeheader()
  for sid in ids:w.writerow({'sample_id':sid,'stripped_path':str(binary),'full_path':str(root/'DO_NOT_OPEN_MISSING_FULL')})
 dump(run/'selection_report.json',{'selected_sample_ids':ids,'selection_uses_gold':False,'selection_uses_model_scores':False});dump(run/'snapshot.json',{'selected_samples':4});dump(prev/'state.json',{'status':'DONE'});dump(prev/'plan.json',{'samples':ids});dump(prev/'source_index_audit.json',{'addresses':{sid:{'ghidra_image_base':0x400000} for sid in ids}})
 db=sqlite3.connect(prev/'source_index.sqlite3')
 db.executescript('CREATE TABLE nodes(id TEXT PRIMARY KEY,sid TEXT,typ TEXT,fn TEXT,pc TEXT,raw TEXT,attrs TEXT,raw_source TEXT);CREATE TABLE edges(id TEXT,sid TEXT,src TEXT,dst TEXT,kind TEXT,typ TEXT,path TEXT,origin_valid INTEGER);CREATE TABLE ranks(sid TEXT,qid TEXT,module TEXT,rank INTEGER,tid TEXT,before REAL,after REAL,cstatus TEXT,dstatus TEXT,dtid TEXT);')
 for sid in ids:
  qid=sid+'::q';fn='00401100';pc='00401104';raw={'address':pc,'opcode':'LOAD','dwarf_truth':'MUST_NOT_LEAK'}
  db.execute('INSERT INTO nodes VALUES(?,?,?,?,?,?,?,?)',(qid,sid,'InstrSem',fn,pc,canonical(raw),'{}','raw_exact_id'))
  for m in MODULES:
   for n,t in enumerate(('A','B'),1):
    tid=sid+'::'+m+'::'+t
    db.execute('INSERT INTO nodes VALUES(?,?,?,?,?,?,?,?)',(tid,sid,m,fn,'',canonical({'type_candidates':['long'],'offset_set':[0,8],'gold_target_id':'MUST_NOT_LEAK'}),'{}','raw_exact_id'))
    db.execute('INSERT INTO ranks VALUES(?,?,?,?,?,?,?,?,?,?)',(sid,qid,m,n,tid,.6,.72 if n==1 else .70,'hypothesis','hypothesis',''))
    db.execute('INSERT INTO edges VALUES(?,?,?,?,?,?,?,?)',('e'+tid,sid,qid,tid,'evidence','fixed_offset','root.edges[0]',1))
 db.commit();db.close()
 (run/'candidate_rankings.csv').write_text('query_id,semantic_type,rank,target_id\n')
 dump(run/'checkpoints/07_rank_export.json',{'outputs':{'csv':{'sha256':sha(run/'candidate_rankings.csv')}}});dump(prev/'checkpoints/03_source_index.json',{'outputs':{str(prev/'source_index.sqlite3'):sha(prev/'source_index.sqlite3')}})
 gh=root/'mock_ghidra';(gh/'support').mkdir(parents=True);(gh/'Ghidra').mkdir();(gh/'Ghidra/application.properties').write_text('application.version=12.0.4\n');exe=gh/'support/analyzeHeadless';exe.write_text(FAKE.replace('PYTHON',sys.executable));exe.chmod(0o755)
 return run,prev,exe,full,binary

def qfixture():
 q={'sample_id':'s','query_id':'q','semantic_type':'LayoutSem','group_id':'g','function_entry':'0x401100','instruction_address':'0x401104','instruction':{'opcode':'LOAD'},'instruction_source':'raw','decision':{'locked':False},'evidence_edge_ids':[],'constraint_edge_ids':[]}
 q['candidates']=[]
 for t,v in [('A',.71),('B',.70)]:
  q['candidates'].append({'target_id':t,'candidate_status':'hypothesis','scope_relation':'same_function','origin_function_entry':'0x401100','attributes':{'offset_set':[0,8]},'attribute_source':'raw','score_after_223':v})
 code={'functions':[{'function_entry':'0x401100','status':'OK','lines':[{'line':1,'text':'return p->x;','tokens':[{'anchors':[{'address':'0x401104'}]}]}]}]}
 return q,code

def req():
 q,c=qfixture();return request(q,c,{})[0]

def resp(r):return {'request_id':r['request_id'],'abstain':False,'scores':[{'target_id':t,'score':.9 if t=='B' else .1,'support_refs':['CODE:L1']} for t in r['candidate_ids']],'explanation':'test'}

class Core(unittest.TestCase):
 def test_01_hex(self):self.assertEqual(address('00401100'),0x401100);self.assertEqual(address(45),45)
 def test_02_duplicate_json(self):
  with self.assertRaises(ValueError):loads('{"a":1,"a":2}')
 def test_03_nonfinite(self):
  with self.assertRaises(ValueError):loads('{"a":NaN}')
 def test_04_tags(self):
  for s in ['../x','/x','jobs','x/y']:
   with self.assertRaises(ValueError):tag_ok(s)
 def test_05_gold_pruning(self):self.assertEqual(clean({'nested':{'dwarf_truth':1,'type_size':4}}),{'nested':{'type_size':4}})
 def test_06_unit_annotation(self):self.assertEqual(unit_note({'byte_size':4})['bit_width_from_declared_byte_size'],32)
 def test_07_unit_conflict(self):self.assertTrue(unit_note({'byte_size':4,'bit_width':64})['unit_conflict'])
 def test_08_sections(self):
  r=req()
  for s in ('EVIDENCE','CONSTRAINT','CODE','QUERY'):self.assertIn('['+s+']',r['messages'][1]['content'])
 def test_09_locked(self):
  q,c=qfixture();q['decision']['locked']=True;self.assertEqual(request(q,c,{})[1],'confirmed_decision_preserved')
 def test_10_no_nearest(self):
  q,c=qfixture();q['instruction_address']='0x401105';self.assertEqual(request(q,c,{})[1],'no_direct_PC_token_anchor')
 def test_11_rejected(self):
  q,c=qfixture();q['candidates'][1]['candidate_status']='rejected';self.assertEqual(request(q,c,{})[1],'terminal_candidate_preserved')
 def test_12_gap(self):
  q,c=qfixture();q['candidates'][0]['score_after_223']=.95;self.assertEqual(request(q,c,{})[1],'gap_not_below_threshold')
 def test_13_valid_response(self):
  r=req();self.assertEqual(validate(r,resp(r))['ordered_target_ids'],['B','A'])
 def test_14_invented_candidate(self):
  r=req();x=resp(r);x['scores'][0]['target_id']='evil'
  with self.assertRaises(ValueError):validate(r,x)
 def test_15_invented_ref(self):
  r=req();x=resp(r);x['scores'][0]['support_refs']=['CODE:L999']
  with self.assertRaises(ValueError):validate(r,x)
 def test_16_abstain(self):
  r=req();self.assertTrue(validate(r,{'request_id':r['request_id'],'abstain':True,'scores':[],'explanation':'unknown'})['abstain'])
 def test_17_bool_score(self):
  r=req();x=resp(r);x['scores'][0]['score']=True
  with self.assertRaises(ValueError):validate(r,x)
 def test_18_missing_candidate(self):
  r=req();x=resp(r);x['scores'].pop()
  with self.assertRaises(ValueError):validate(r,x)
 def test_19_Jython_header(self):self.assertIn('#@runtime Jython',(ROOT/'ghidra_scripts/ExportSeraphCodeV730.py').read_text())
 def test_20_py38_syntax(self):
  for p in ROOT.rglob('*.py'):ast.parse(p.read_text(encoding='utf-8'),feature_version=8)

@unittest.skipUnless(shutil.which('gcc') and shutil.which('strip'),'gcc/strip required for real ELF test fixture')
class Integration(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.run,self.prev,self.gh,self.full,self.binary=fixture(self.root);self.out=self.run/'seraph_task1_v7_3_0/all'
 def tearDown(self):self.tmp.cleanup()
 def cmd(self,action='run',*extra):return subprocess.run([sys.executable,'-S',str(ROOT/'cli.py'),action,'--run-dir',str(self.run),'--analyze-headless',str(self.gh),'--min-free-gib','0']+list(extra),text=True,capture_output=True,timeout=35)
 def prep(self):
  r=self.cmd();self.assertEqual(r.returncode,0,r.stdout+r.stderr)
 def test_21_debug_firewall(self):
  self.assertFalse(admit(self.binary)['debug_sections_present'])
  with self.assertRaises(ValueError):admit(self.full)
 def test_22_prepare_5_steps_mock_Ghidra(self):
  before={str(p):sha(p) for p in self.run.rglob('*') if p.is_file()};self.prep();r=load(self.out/'task1_report.json');self.assertEqual(r['samples'],4);self.assertEqual(r['SSQ']['selected_requests'],16)
  for p,h in before.items():self.assertEqual(sha(p),h)
  self.assertNotIn('MUST_NOT_LEAK',(self.out/'ssq_requests.jsonl').read_text())
 def test_23_resume(self):
  self.prep();r=self.cmd('continue');self.assertEqual(r.returncode,0,r.stderr);self.assertIn('REUSED VERIFIED',r.stdout)
 def test_24_pilot_full_cache(self):
  r=self.cmd('pilot');self.assertEqual(r.returncode,0,r.stderr);self.prep();self.assertEqual(len((self.gh.parents[1]/'calls').read_text().splitlines()),4)
 def test_25_wrong_previous(self):
  r=self.cmd('check','--previous-dir',str(self.prev.parent/'all'));self.assertNotEqual(r.returncode,0);self.assertIn('Required input missing',r.stderr)
 def test_26_input_hash_change(self):
  with (self.run/'candidate_rankings.csv').open('a') as f:f.write('bad')
  r=self.cmd();self.assertNotEqual(r.returncode,0);self.assertIn('rankings SHA',r.stderr)
 def test_27_source_lock(self):
  import fcntl
  with (self.run/'.pipeline.lock').open('a+') as f:
   fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);r=self.cmd('check');self.assertNotEqual(r.returncode,0);self.assertIn('ACTIVE_LOCK',r.stderr)
 def test_28_rc0_without_export_fails(self):
  (self.gh.parents[1]/'fail').touch();r=self.cmd();self.assertNotEqual(r.returncode,0);self.assertIn('actual per-sample log',r.stderr)
 def test_29_changed_output(self):
  self.prep();(self.out/'ssq_requests.jsonl').write_text('{}\n');r=self.cmd('continue');self.assertNotEqual(r.returncode,0);self.assertIn('Output changed',r.stderr)
 def test_30_changed_plan(self):
  self.prep();r=self.cmd('run','--max-requests','2');self.assertNotEqual(r.returncode,0);self.assertIn('Plan/code/scope changed',r.stderr)
 def test_31_ask_consent(self):
  self.prep()
  with self.assertRaisesRegex(ValueError,'allow-send-code'):ask(self.out,ROOT/'config/llm.example.json','primary',False)
 def test_32_LLM_loopback_budget_and_eval(self):
  self.prep();calls=[]
  class Handler(BaseHTTPRequestHandler):
   def log_message(self,*a):pass
   def do_POST(self):
    body=json.loads(self.rfile.read(int(self.headers['Content-Length'])));calls.append(body);prompt=body['messages'][1]['content'];rid=prompt.splitlines()[0].split('=',1)[1];cs=json.loads(prompt.split('[QUERY]\n',1)[1])['candidates']
    obj={'request_id':rid,'abstain':False,'scores':[{'target_id':c['target_id'],'score':.9 if c['target_id'].endswith('B') else .1,'support_refs':['CODE:L1']} for c in cs],'explanation':'MOCK SERVICE test, not a real LLM'}
    data=json.dumps({'choices':[{'finish_reason':'stop','message':{'content':json.dumps(obj)}}]}).encode();self.send_response(200);self.end_headers();self.wfile.write(data)
  server=ThreadingHTTPServer(('127.0.0.1',0),Handler);t=threading.Thread(target=server.serve_forever,daemon=True);t.start()
  try:
   c=load(ROOT/'config/llm.example.json');c.update(endpoint='http://127.0.0.1:%d/v1/chat/completions'%server.server_address[1],model='MOCK_TEST_ONLY',max_calls=2);cp=self.root/'llm.json';dump(cp,c)
   self.assertEqual(ask(self.out,cp,'primary',True),0);self.assertEqual(ask(self.out,cp,'primary',True),0);self.assertEqual(len(calls),2)
   rs=list(jsonl(self.out/'ssq_requests.jsonl'));g=self.root/'gold.jsonl';write_jsonl(g,[{'sample_id':r['sample_id'],'query_id':r['query_id'],'target_semantic_type':r['semantic_type'],'target_id':next(t for t in r['candidate_ids'] if t.endswith('B'))} for r in rs]);evaluate(self.out,g,'primary','test')
   x=load(self.out/'evaluation/test/accuracy_report.json');self.assertEqual(sum(z['net_correct'] for z in x['modules'].values()),2);self.assertEqual(x['strict_gold_groups'],16);self.assertFalse(x['document_four_evidence_fusion_completed'])
  finally:server.shutdown();server.server_close();t.join(timeout=2)
 def test_33_Gold_missing_candidate_counts(self):
  self.prep();g=self.root/'g.jsonl';write_jsonl(g,[{'sample_id':'s0','query_id':'s0::q','target_semantic_type':'LayoutSem','target_id':'missing'}]);evaluate(self.out,g,'primary','base',True);x=load(self.out/'evaluation/base/accuracy_report.json');self.assertEqual(x['modules']['LayoutSem']['baseline_223']['candidate_coverage'],0)
 def test_34_background_launch(self):
  r=subprocess.run([sys.executable,'-S',str(ROOT/'launch.py'),'pilot','--run-dir',str(self.run),'--analyze-headless',str(self.gh),'--min-free-gib','0'],text=True,capture_output=True,timeout=10);self.assertEqual(r.returncode,0,r.stdout+r.stderr)
  state=self.run/'seraph_task1_v7_3_0/pilot3/state.json'
  for _ in range(150):
   if state.exists() and load(state).get('status') in ('PREPARED','PREPARED_WITH_GAPS','FAILED'):break
   time.sleep(.05)
  self.assertIn(load(state)['status'],('PREPARED','PREPARED_WITH_GAPS'))
 def test_35_fixed_wrapper_scope(self):
  r=subprocess.run(['bash',str(ROOT/'run20.sh'),'check','--run-dir',str(self.run)],text=True,capture_output=True);self.assertEqual(r.returncode,2)
 def test_36_foreign_parent_scope(self):
  dump(self.prev/'plan.json',{'samples':['bad']});r=self.cmd('check');self.assertNotEqual(r.returncode,0);self.assertIn('scope mismatch',r.stderr)
 def test_37_session_requires_real_model(self):
  with self.assertRaisesRegex(ValueError,'ACTUAL'):config(ROOT/'config/llm.example.json')
 def test_38_new_output_tag(self):
  r=self.cmd('pilot','--output-tag','pilot3_custom');self.assertEqual(r.returncode,0,r.stderr);self.assertTrue((self.run/'seraph_task1_v7_3_0/pilot3_custom/task1_report.json').exists())

@unittest.skipUnless(os.environ.get('SERAPH_TEST_ANALYZE_HEADLESS') and shutil.which('gcc') and shutil.which('nm'),'Live Ghidra not configured; mocked exporter tests do NOT establish live-Ghidra compatibility')
class Live(unittest.TestCase):
 def test_39_live_Ghidra(self):
  from seraph.codebridge import export
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);run,prev,exe,full,binary=fixture(root);nm=subprocess.check_output(['nm',str(full)],text=True);entry='0x'+next(l.split()[0] for l in nm.splitlines() if l.endswith(' f'))
   c={'base':root/'live','headless':Path(os.environ['SERAPH_TEST_ANALYZE_HEADLESS']),'ghidra_version':'live-test'}
   p=export(c,{'sample_id':'s0','stripped_path':str(binary)},[entry],ROOT/'ghidra_scripts/ExportSeraphCodeV730.py')
   self.assertEqual(load(p)['functions'][0]['status'],'OK')

if __name__=='__main__':unittest.main()
