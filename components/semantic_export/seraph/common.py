from __future__ import annotations
import csv, hashlib, json, math, os, re, shutil, sqlite3, time
from pathlib import Path

VERSION='7.2.2'
MODULES=('APISem','TypeSem','LayoutSem','InterfaceSem')
def stamp(): return time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20), b''): h.update(b)
    return h.hexdigest()
def canonical(x): return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(',',':'), allow_nan=False)
def digest(x): return hashlib.sha256(canonical(x).encode()).hexdigest()
def load(p):
    with Path(p).open(encoding='utf-8') as f:return json.load(f)
def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+'.tmp.'+str(os.getpid()))
    with t.open('w',encoding='utf-8') as f:
        json.dump(x,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(t,p)
def jsonl(p):
    with Path(p).open(encoding='utf-8') as f:
        for i,line in enumerate(f,1):
            if not line.strip():continue
            try:r=json.loads(line)
            except Exception as e:raise ValueError('%s:%s: %s'%(p,i,e))
            if not isinstance(r,dict):raise ValueError('%s:%s: expected object'%(p,i))
            yield r
def write_jsonl(p,rows):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8') as f:
        for r in rows:f.write(canonical(r)+'\n')
def csv_rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:
        yield from csv.DictReader(f)
def write_csv(p,rows,fields):
    with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader()
        for r in rows:w.writerow({k:canonical(v) if isinstance(v,(dict,list)) else v for k,v in r.items()})
def integer(v):
    if v is None or isinstance(v,bool):return None
    if isinstance(v,int):return v
    if isinstance(v,float):return int(v) if math.isfinite(v) and v==int(v) else None
    s=str(v).strip()
    if not re.fullmatch(r'[+-]?(?:0x[0-9a-fA-F]+|[0-9]+)',s):return None
    return int(s,16 if '0x' in s.lower() else 10)
def address(v):
    """Ghidra addresses: unprefixed strings are hexadecimal, NOT decimal."""
    if isinstance(v,int) and not isinstance(v,bool):return v
    if v is None:return None
    s=str(v).strip()
    if s.startswith('ram:'):s=s[4:]
    if not re.fullmatch(r'(?:0x)?[0-9a-fA-F]+',s):return None
    return int(s,16)
def first(d,*keys):
    for k in keys:
        v=d.get(k)
        if v is not None and v!='' and v!=[] and v!={}:return v
    return None
def reg_name(v):
    if v is None:return None
    s=str(v).replace('reg:','').replace('%','').strip().lower()
    # No ARM aliases here: runtime preflight rejects non-x86-64 strict identity.
    regs={'rax','rbx','rcx','rdx','rsi','rdi','rbp','rsp','rip'}|{'r'+str(i) for i in range(8,16)}
    return s if s in regs else None
DWARF_REGS={0:'rax',1:'rdx',2:'rcx',3:'rbx',4:'rsi',5:'rdi',6:'rbp',7:'rsp',8:'r8',9:'r9',10:'r10',11:'r11',12:'r12',13:'r13',14:'r14',15:'r15',16:'rip'}
def free_check(path,min_gib):
    f=shutil.disk_usage(path).free
    if f < min_gib*(1<<30):raise RuntimeError('Insufficient free space: %.2f GiB; require %.2f GiB. No input is deleted.'%(f/(1<<30),min_gib))
    return f

def connect_ro(path):
    db=sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro',uri=True,timeout=3)
    db.row_factory=sqlite3.Row;db.execute('PRAGMA query_only=ON');return db

# Task1 additions: strict JSON, explicit output-scope validation and locks.
VERSION='7.3.0'
import contextlib

def loads(text):
    def unique(items):
        d={}
        for k,v in items:
            if k in d:raise ValueError('Duplicate JSON key: '+k)
            d[k]=v
        return d
    def bad(x):raise ValueError('Nonfinite JSON value: '+x)
    return json.loads(text,object_pairs_hook=unique,parse_constant=bad)

def load(p):
    return loads(Path(p).read_text(encoding='utf-8'))

def tag_ok(s):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}',s) or s in ('assets','jobs','cache'):
        raise ValueError('Unsafe/reserved output tag: '+s)
    return s

def require(p):
    p=Path(p).expanduser().resolve()
    if not p.is_file():raise FileNotFoundError('Required input missing: '+str(p))
    return p

def inside(p,root):
    p=Path(p).resolve();root=Path(root).resolve()
    if p!=root and root not in p.parents:raise ValueError('Path outside configured run: '+str(p))
    return p

def hx(v):
    n=address(v)
    return None if n is None else '0x%x'%n

def finite(v):
    if v is None or isinstance(v,bool) or v=='':return None
    try:n=float(v)
    except (ValueError,TypeError):return None
    return n if math.isfinite(n) else None

@contextlib.contextmanager
def locked(path,shared=False,create=False):
    import fcntl
    p=Path(path)
    if create:p.parent.mkdir(parents=True,exist_ok=True)
    elif not p.is_file():raise FileNotFoundError('Source lock missing: '+str(p))
    f=p.open('a+' if create else 'rb')
    try:
        try:fcntl.flock(f.fileno(),(fcntl.LOCK_SH if shared else fcntl.LOCK_EX)|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('ACTIVE_LOCK: '+str(p))
        yield f
    finally:f.close()

def verify_outputs(cp):
    for p,h in cp['outputs'].items():
        if sha(require(p))!=h:raise ValueError('Output changed: '+p)

def record(p):
    p=require(p);s=p.stat()
    return {'sha256':sha(p),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns}
