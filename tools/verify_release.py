"""Read-only source integrity and packaging checks; never imports research code."""
import ast
import hashlib
import json
import re
from pathlib import Path

root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'release/source_manifest.json').read_text())
aliases=json.loads((root/'release/compatibility_aliases.json').read_text())
errors=[]
for row in manifest['files']:
    p=root/row['path']
    if not p.is_file():
        errors.append('missing '+row['path']);continue
    if hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:
        errors.append('hash mismatch '+row['path'])
    text=p.read_text()
    if p.suffix=='.py':
        try:ast.parse(text,filename=str(p))
        except SyntaxError as e:errors.append(str(e))
    for name in re.findall(r'packages/([\w.-]+)',text):
        if name not in aliases:errors.append('unresolved package '+name+' in '+row['path'])
for name,target in aliases.items():
    p=root/'packages'/name
    if not p.is_symlink() or p.resolve()!=(root/target).resolve():
        errors.append('invalid compatibility link '+name)
result={'status':'FAILED' if errors else 'PASSED','source_files':len(manifest['files']),
        'compatibility_links':len(aliases),'errors':errors,'research_code_executed':False,
        'end_to_end_experiment_validated':False}
print(json.dumps(result,ensure_ascii=False,indent=2))
raise SystemExit(bool(errors))
