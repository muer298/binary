#!/usr/bin/env python3
import argparse,json
p=argparse.ArgumentParser();p.add_argument('semantic_json');a=p.parse_args();x=json.load(open(a.semantic_json))
low=x.get('low_level_sem',{}) or {}; bs=x.get('blocksem_v7',{}) or {}
blocks=low.get('block_sem',[]) or []; instr=low.get('instr_sem',[]) or []
print(json.dumps({
 'schema':x.get('program_info',{}).get('schema_version'),
 'instr_sem':len(instr),'block_sem':len(blocks),
 'blocks_with_ordered_members':sum(bool(b.get('ordered_instruction_ids')) for b in blocks),
 'blocks_member_complete':sum(bool(b.get('member_list_complete')) for b in blocks),
 'instr_with_pcode_site_id':sum(bool(i.get('pcode_site_id')) for i in instr),
 'instr_with_value_ids':sum(bool(i.get('input_value_ids') or i.get('output_value_ids')) for i in instr),
 'blocksem_v7_stats':bs.get('stats',{})
},ensure_ascii=False,indent=2))
