import csv
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PIPE=ROOT/'vendor/pipeline'
UP=ROOT/'vendor/upstream/01_modified_upstream'
sys.path.insert(0,str(PIPE))

class CleanPackageTests(unittest.TestCase):
    def test_v702_api_provenance_survives(self):
        path=UP/'02_extract_nodes_edges_blocksem_v7.py'
        spec=importlib.util.spec_from_file_location('exv702',path)
        ex=importlib.util.module_from_spec(spec);spec.loader.exec_module(ex)
        obj={
            'id':'api_401000_0',
            'api_name':'printf',
            'callsite':'401000',
            'semantic_template':{
                'symbol_resolution':{
                    'resolution_kind':'external_or_thunk',
                    'api_name':'printf',
                    'callsite':'401000'
                }
            }
        }
        p=ex.infer_api_provenance(obj)
        self.assertEqual(p['api_source'],'symbol_table')
        self.assertTrue(p['deterministic_api_source'])
        data={'high_level_sem':{'api_sem':[obj]}}
        nodes,_=ex.build_nodes(data,'s','x.json')
        api=[n for n in nodes if n['node_type']=='APISem'][0]
        self.assertEqual(api['attrs']['api_source'],'symbol_table')

    def test_stage04_real_edge_lineage(self):
        import conflict_resolver_v6_5 as cr
        nodes={
            's::InstrSem::a':{'node_type':'InstrSem'},
            's::LayoutSem::b':{'node_type':'LayoutSem'},
            's::APISem::c':{'node_type':'APISem','api_name':'strlen','api_source':'import_table'},
        }
        c=cr.ConstraintCompiler(nodes,{})
        resolved={'edge_id':'origE','edge_class':'evidence','source_node':'s::InstrSem::a',
                  'target_node':'s::LayoutSem::b','strength':.9}
        raw={'edge_id':'origE','source_node':resolved['source_node'],'target_node':resolved['target_node'],
             'edge_kind':'evidence','evidence_type':'fixed_offset','strength':.9}
        x=c.compile_edge(raw,resolved)[0]
        self.assertEqual(x.constraint_id,'derived::origE::struct')
        self.assertEqual(x.source_edge_id,'origE')
        api=c.compile_api_self_constraints(['s::APISem::c'])[0]
        self.assertEqual(api.source_edge_id,'')

    def test_v56_evaluator_revision_contract(self):
        text=(PIPE/'11_evaluate_reasoning_v5_5.py').read_text(encoding='utf-8')
        self.assertIn('reasoning_revision_compatible',text)
        self.assertIn('v5_6_irreversible_confirmation_transactional_raw_sglobal',text)
        self.assertNotIn('"v5_5_revision_active": (',text)

    def test_selection_hash_is_deterministic(self):
        # Import control only to exercise the exact packaged rule.
        spec=importlib.util.spec_from_file_location('ctl',ROOT/'control.py')
        ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl)
        ids=['s3','s1','s2','s10']
        a=sorted(ids,key=lambda x:(ctl.selection_key(x),x))
        b=sorted(reversed(ids),key=lambda x:(ctl.selection_key(x),x))
        self.assertEqual(a,b)
        self.assertEqual(len(set(a)),len(ids))

if __name__=='__main__':
    unittest.main()
