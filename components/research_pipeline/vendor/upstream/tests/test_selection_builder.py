import json,subprocess,sys,tempfile,unittest
from pathlib import Path
class T(unittest.TestCase):
 def test_selection_gate(self):
  with tempfile.TemporaryDirectory() as td:
   d=Path(td); g=d/'g.jsonl'; b=d/'b.jsonl'; o=d/'out'
   gr=[]; br=[]
   for i in range(120):
    gr.append({'query_id':f'q{i}','gold_id':'g','sample_id':f's{i//4}','project_id':f'p{i//12}','strong':True})
    br.append({'query_id':f'q{i}','ranked_ids':['x','g'] if i<20 else ['g','x']})
   g.write_text(''.join(json.dumps(x)+'\n' for x in gr)); b.write_text(''.join(json.dumps(x)+'\n' for x in br))
   s=Path(__file__).resolve().parents[1]/'03_selection_validation'/'build_selection_validation_v7.py'
   cp=subprocess.run([sys.executable,str(s),'--gold',str(g),'--baseline',str(b),'--out-dir',str(o)],capture_output=True,text=True)
   self.assertEqual(cp.returncode,0,cp.stdout+cp.stderr)
   gate=json.load(open(o/'selection_gate.json'))
   self.assertTrue(gate['pass']); self.assertGreaterEqual(gate['baseline_top1_errors'],10)
if __name__=='__main__':unittest.main()
