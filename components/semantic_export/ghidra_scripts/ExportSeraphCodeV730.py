# -*- coding: utf-8 -*-
# Stripped-only pseudocode with direct token / P-code machine address anchors.
#@category Research2.SERAPH
#@runtime Jython
from __future__ import print_function
import json, os
from ghidra.app.decompiler import DecompInterface
from ghidra.app.decompiler.component import DecompilerUtils

def text(v):
    try:return unicode(v)
    except NameError:return str(v)

def addr(v):
    return None if v is None else '0x%x'%int(str(v),16)

def anchors(t):
    op=t.getPcodeOp()
    if op is not None:
        s=op.getSeqnum()
        return [{'address':addr(s.getTarget()),'kind':'pcode_target','pcode_order':int(s.getOrder()),'pcode_time':int(s.getTime())}]
    lo,hi=t.getMinAddress(),t.getMaxAddress()
    return [{'address':addr(lo),'kind':'token_single_address'}] if lo is not None and lo==hi else []

def lines(markup):
    out=[]
    for i,line in enumerate(DecompilerUtils.toLines(markup),1):
        parts=[text(line.getIndentString())];pos=len(parts[0]);tokens=[]
        for t in line.getAllTokens():
            s=text(t.getText());tokens.append({'text':s,'start_char':pos,'end_char':pos+len(s),'anchors':anchors(t)});parts.append(s);pos+=len(s)
        out.append({'line':i,'text':''.join(parts),'tokens':tokens})
    return out

def main():
    args=list(getScriptArgs())
    if len(args)!=4:raise ValueError('OUTPUT_JSON SAMPLE_ID ENTRIES_TXT TIMEOUT_SEC required')
    output,sid,entryfile,timeout=args[0],args[1],args[2],int(args[3])
    with open(entryfile) as f:entries=[s.strip() for s in f if s.strip()]
    if len(set(entries))!=len(entries):raise ValueError('Duplicate entries')
    dc=DecompInterface();dc.toggleCCode(True);dc.toggleSyntaxTree(True);dc.setSimplificationStyle('decompile')
    if not dc.openProgram(currentProgram):raise RuntimeError(str(dc.getLastMessage()))
    result=[]
    try:
        fm=currentProgram.getFunctionManager();space=currentProgram.getAddressFactory().getDefaultAddressSpace()
        for i,entry in enumerate(entries,1):
            if monitor.isCancelled():raise RuntimeError('Cancelled')
            fn=fm.getFunctionAt(space.getAddress(entry.replace('0x','')))
            row={'sample_id':sid,'function_entry':entry,'status':'NOT_DECOMPILED','lines':[]}
            if fn is None:row['status']='FUNCTION_ENTRY_NOT_FOUND'
            elif fn.isExternal():row['status']='EXTERNAL_FUNCTION'
            else:
                res=dc.decompileFunction(fn,timeout,monitor);row['message']=text(res.getErrorMessage())
                if not res.decompileCompleted():row['status']='DECOMPILE_INCOMPLETE'
                elif res.getCCodeMarkup() is None:row['status']='NO_CODE_MARKUP'
                else:
                    row['lines']=lines(res.getCCodeMarkup());row['status']='OK' if row['lines'] else 'NO_CODE_LINES'
            result.append(row);print('[SERAPH CODE]',i,'/',len(entries),entry,row['status'])
    finally:dc.dispose()
    parent=os.path.dirname(os.path.abspath(output))
    if not os.path.isdir(parent):os.makedirs(parent)
    with open(output,'w') as f:json.dump({'schema':'seraph_code_export_v7_3_0','sample_id':sid,'completed':True,'image_base':addr(currentProgram.getImageBase()),'requested_entries':entries,'functions':result},f,ensure_ascii=True,sort_keys=True)
    print('SERAPH_CODE_EXPORT_DONE',sid,'functions',len(result))

if __name__=='__main__':main()
