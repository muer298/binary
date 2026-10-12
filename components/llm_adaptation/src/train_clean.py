from __future__ import annotations
import argparse, hashlib, json, os, random, shutil, time
from collections import defaultdict
from pathlib import Path
from common import *
from common import _sha_file


def metrics(src_z,dec_z,src_groups,dec_groups):
    import torch
    sim=src_z @ dec_z.T
    def one(mat,qg,cg):
        out=[]
        for i,g in enumerate(qg):
            order=torch.argsort(mat[i],descending=True).tolist();rank=None
            for j,idx in enumerate(order,1):
                if cg[idx]==g:rank=j;break
            out.append(rank)
        n=len(out)
        return {'n':n,'R1':sum(r==1 for r in out)/n if n else None,
                'R5':sum(r is not None and r<=5 for r in out)/n if n else None,
                'MRR':sum(1/r for r in out if r)/n if n else None}
    return {'source_to_pseudoc':one(sim,src_groups,dec_groups),
            'pseudoc_to_source':one(sim.T,dec_groups,src_groups)}


def audit_pairs(pairs):
    if any(r.get('global_role')!='TRAIN' for r in pairs):
        raise RuntimeError('Clean Stage-1 pairs contain a non-GLOBAL-TRAIN role')
    split_rows={s:[r for r in pairs if r.get('split')==s] for s in ('train','val')}
    if sum(len(x) for x in split_rows.values())!=len(pairs):
        raise RuntimeError('Unexpected Stage-1 inner split')
    split_groups={s:{r['source_function_id'] for r in rr} for s,rr in split_rows.items()}
    group_splits=defaultdict(set);code_splits=defaultdict(set)
    for r in pairs:
        group_splits[r['source_function_id']].add(r['split'])
        code_splits[hashlib.sha256(r['source_code'].encode()).hexdigest()].add(r['split'])
    group_leaks={k:sorted(v) for k,v in group_splits.items() if len(v)>1}
    code_leaks={k:sorted(v) for k,v in code_splits.items() if len(v)>1}
    report={'rows':{s:len(v) for s,v in split_rows.items()},
            'unique_source_groups':{s:len(v) for s,v in split_groups.items()},
            'cross_split_source_groups':len(group_leaks),
            'cross_split_exact_source_code':len(code_leaks)}
    if any(not split_rows[s] for s in ('train','val')):
        raise RuntimeError('Both inner train and inner-dev must be non-empty: '+json.dumps(report))
    if group_leaks or code_leaks:
        raise RuntimeError('Cross-split leakage detected: '+json.dumps(report))
    return report,split_rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--gpu',type=int,required=True);a=ap.parse_args()
    cfg=load_json(a.config);cc=cfg['contrastive'];root=Path(cfg['output_root']);pairs=list(rows(root/'pairs.jsonl'))
    if len(pairs)<cfg['pair_minimum']:raise SystemExit('Not enough GLOBAL-TRAIN clean pairs; run prepare first.')
    prepared=load_json(root/'STAGE1_CLEAN_DATA_AUDIT.json')
    if prepared.get('status')!='PASS_ENGINEERING_SCALE' or _sha_file(root/'pairs.jsonl')!=prepared['clean_pairs_sha256']:
        raise RuntimeError('Prepared clean pair manifest changed or failed')
    data_audit,split_rows=audit_pairs(pairs)
    if data_audit['rows']!={'train':29,'val':11}:
        raise RuntimeError('Clean inner split size changed')
    outdir=root/'contrastive'
    if outdir.exists():raise SystemExit('Refusing to overwrite existing contrastive output: '+str(outdir))
    outdir.mkdir(parents=True)
    dump_json(outdir/'training_config.json',cfg)

    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID';os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    os.environ['TOKENIZERS_PARALLELISM']='false';os.environ['PYTHONNOUSERSITE']='1'
    import torch, torch.nn.functional as F
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from peft import LoraConfig,PeftModel,get_peft_model
    random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
    free_before,total_vram=torch.cuda.mem_get_info(0);gpu_name=torch.cuda.get_device_name(0);started=time.monotonic()

    tok=AutoTokenizer.from_pretrained(cfg['model_dir'],local_files_only=True,trust_remote_code=False)
    if tok.pad_token_id is None:
        if tok.eos_token_id is None:raise RuntimeError('Tokenizer has neither pad_token nor eos_token')
        tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(
        cfg['model_dir'],torch_dtype=torch.bfloat16,device_map={'':0},local_files_only=True,
        low_cpu_mem_usage=True,use_safetensors=True,trust_remote_code=False)
    model.config.use_cache=False;model.gradient_checkpointing_enable();model.enable_input_require_grads()
    target_modules=['q_proj','k_proj','v_proj','o_proj']
    model=get_peft_model(model,LoraConfig(
        r=cc['lora_r'],lora_alpha=cc['lora_alpha'],lora_dropout=cc['lora_dropout'],
        bias='none',task_type='CAUSAL_LM',target_modules=cc['lora_target_modules']))
    if target_modules!=cc['lora_target_modules']:
        raise RuntimeError('LoRA target module contract changed')
    proj=torch.nn.Linear(model.config.hidden_size,cc['projection_dim'],bias=False,device='cuda',dtype=torch.float32)
    trainable_names=[n for n,p in model.named_parameters() if p.requires_grad]
    if not trainable_names or any('lora_' not in n for n in trainable_names):
        raise RuntimeError('Base model is not fully frozen / LoRA targets missing')
    init_hash=hashlib.sha256()
    for name,p in model.named_parameters():
        if p.requires_grad:
            init_hash.update(name.encode());init_hash.update(p.detach().float().cpu().contiguous().numpy().tobytes())
    init_hash.update(proj.weight.detach().cpu().contiguous().numpy().tobytes())
    fresh_lora_initialization_sha256=init_hash.hexdigest()
    trainable=[p for p in list(model.parameters())+list(proj.parameters()) if p.requires_grad]
    opt=torch.optim.AdamW(trainable,lr=cc['learning_rate'])
    model_params=sum(p.numel() for p in model.parameters());model_trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    proj_params=sum(p.numel() for p in proj.parameters());proj_trainable=sum(p.numel() for p in proj.parameters() if p.requires_grad)
    total_params=model_params+proj_params;trainable_params=model_trainable+proj_trainable
    first_lora_name,first_lora=next((n,p) for n,p in model.named_parameters() if p.requires_grad)
    lora_before=first_lora.detach().float().cpu().clone();proj_before=proj.weight.detach().float().cpu().clone()
    torch.cuda.reset_peak_memory_stats(0)

    def enc(texts,grad=True):
        b=tok(texts,return_tensors='pt',padding=True,truncation=True,max_length=cc['max_length']);b={k:v.cuda() for k,v in b.items()}
        ctx=torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            o=model(**b,output_hidden_states=True,use_cache=False,return_dict=True)
            h=o.hidden_states[-1].float();mask=b['attention_mask'].unsqueeze(-1)
            z=(h*mask).sum(1)/mask.sum(1).clamp_min(1);z=F.normalize(proj(z),dim=-1)
        return z

    train=split_rows['train'];val=split_rows['val'];bygroup={}
    for r in train:bygroup.setdefault(r['source_function_id'],[]).append(r)
    groups=list(bygroup);history=[];step=0;selection_history=[];best_score=float('-inf');best_epoch=None
    def validate_inner_dev():
        model.eval();proj.eval();vg={}
        for r in val:vg.setdefault(r['source_function_id'],[]).append(r)
        src=[];sg=[];dec=[];dg=[]
        for g,rr in vg.items():src.append(rr[0]['source_code']);sg.append(g)
        for g,rr in vg.items():
            for r in rr:dec.append(r['pseudoc']);dg.append(g)
        zs=[];zd=[]
        for i in range(0,len(src),4):zs.append(enc(src[i:i+4],False).cpu())
        for i in range(0,len(dec),4):zd.append(enc(dec[i:i+4],False).cpu())
        return metrics(torch.cat(zs),torch.cat(zd),sg,dg)
    for epoch in range(cc['epochs']):
        random.Random(cfg['seed']+epoch).shuffle(groups)
        for off in range(0,len(groups),cc['batch_size']):
            gg=groups[off:off+cc['batch_size']]
            if len(gg)<2:continue
            batch=[random.choice(bygroup[g]) for g in gg]
            model.train();proj.train();opt.zero_grad(set_to_none=True)
            zs=enc([r['source_code'] for r in batch],True);zd=enc([r['pseudoc'] for r in batch],True)
            logits=zs@zd.T/cc['temperature'];y=torch.arange(len(batch),device='cuda')
            loss=(F.cross_entropy(logits,y)+F.cross_entropy(logits.T,y))/2
            if not torch.isfinite(loss):raise RuntimeError(f'Non-finite loss at step {step+1}: {loss.item()}')
            loss.backward();opt.step();step+=1
            item={'step':step,'epoch':epoch+1,'loss':float(loss.detach())};history.append(item)
            print(json.dumps({'TRAIN_STEP':item},ensure_ascii=False),flush=True)
        valm=validate_inner_dev()
        score=(valm['source_to_pseudoc']['MRR']+valm['pseudoc_to_source']['MRR'])/2
        if not (score==score and abs(score)!=float('inf')):
            raise RuntimeError('Non-finite inner-dev selection score')
        candidate=outdir/'checkpoint_candidates'/('epoch_%d'%(epoch+1))
        candidate.mkdir(parents=True,exist_ok=False)
        model.save_pretrained(candidate/'adapter')
        torch.save({'state_dict':proj.state_dict(),'in_dim':model.config.hidden_size,
                    'out_dim':cc['projection_dim']},candidate/'projection.pt')
        selection_history.append({'epoch':epoch+1,'optimizer_steps':step,'inner_dev':valm,
                                  'bidirectional_MRR':score,
                                  'lora_max_abs_delta_from_fresh_init':float((first_lora.detach().float().cpu()-lora_before).abs().max()),
                                  'projection_max_abs_delta_from_fresh_init':float((proj.weight.detach().float().cpu()-proj_before).abs().max())})
        if score>best_score:
            best_score=score;best_epoch=epoch+1
        print(json.dumps({'INNER_DEV_EPOCH':selection_history[-1]},ensure_ascii=False),flush=True)
    if step<=0 or best_epoch is None:
        raise RuntimeError('No optimizer step or selected checkpoint')
    train_seconds=time.monotonic()-started
    peak_allocated=torch.cuda.max_memory_allocated(0);peak_reserved=torch.cuda.max_memory_reserved(0)
    lora_delta=float((first_lora.detach().float().cpu()-lora_before).abs().max())
    proj_delta=float((proj.weight.detach().float().cpu()-proj_before).abs().max())
    if lora_delta<=0 or proj_delta<=0:raise RuntimeError(f'Parameter update check failed: lora={lora_delta} projection={proj_delta}')

    selected=outdir/'checkpoint_candidates'/('epoch_%d'%best_epoch)
    adapter=outdir/'adapter';shutil.copytree(selected/'adapter',adapter)
    projection_path=outdir/'projection.pt';shutil.copyfile(selected/'projection.pt',projection_path)
    selected_record=next(x for x in selection_history if x['epoch']==best_epoch)
    selected_metrics=selected_record['inner_dev']
    if selected_record['lora_max_abs_delta_from_fresh_init']<=0 or selected_record['projection_max_abs_delta_from_fresh_init']<=0:
        raise RuntimeError('Selected checkpoint did not update LoRA and projection')
    selection_report={'schema':'seraph_stage1_clean_checkpoint_selection_v0_1_5',
        'policy':cc['checkpoint_selection'],'selection_split':'stage1_inner_dev_from_GLOBAL_TRAIN_only',
        'selection_history':selection_history,'selected_epoch':best_epoch,
        'selected_bidirectional_MRR':best_score,
        'outer_DEV_opened':0,'CALIBRATION_opened':0,
        'COMPROMISED_TEST_V1_opened':0,'FINAL_TEST_V2_opened':0}
    dump_json(outdir/'checkpoint_selection_report.json',selection_report)

    # Mandatory reload smoke: fresh base + saved adapter + saved projection.
    del opt,trainable,proj,model;torch.cuda.empty_cache()
    reload_base=AutoModelForCausalLM.from_pretrained(
        cfg['model_dir'],torch_dtype=torch.bfloat16,device_map={'':0},local_files_only=True,
        low_cpu_mem_usage=True,use_safetensors=True,trust_remote_code=False)
    reload_model=PeftModel.from_pretrained(reload_base,adapter,is_trainable=False);reload_model.eval()
    checkpoint=torch.load(projection_path,map_location='cpu')
    reload_proj=torch.nn.Linear(checkpoint['in_dim'],checkpoint['out_dim'],bias=False,device='cuda',dtype=torch.float32)
    reload_proj.load_state_dict(checkpoint['state_dict']);reload_proj.eval()
    b=tok([val[0]['source_code']],return_tensors='pt',truncation=True,max_length=cc['max_length']);b={k:v.cuda() for k,v in b.items()}
    with torch.no_grad():
        o=reload_model(**b,output_hidden_states=True,use_cache=False,return_dict=True)
        h=o.hidden_states[-1].float();mask=b['attention_mask'].unsqueeze(-1)
        smoke_z=F.normalize(reload_proj((h*mask).sum(1)/mask.sum(1).clamp_min(1)),dim=-1)
    reload_smoke={'passed':bool(torch.isfinite(smoke_z).all()),'shape':list(smoke_z.shape)}
    if not reload_smoke['passed']:raise RuntimeError('Reload smoke produced non-finite representation')

    report={
        'schema':'seraph_stage1_clean_contrastive_training_report_v0_1_5','status':'DONE','model_dir':cfg['model_dir'],
        'adapter_dir':str(adapter),'projection_path':str(projection_path),'config_path':str(outdir/'training_config.json'),
        'pair_count':len(pairs),'data_audit':data_audit,'epochs':cc['epochs'],'batch_size':cc['batch_size'],
        'max_length':cc['max_length'],'learning_rate':cc['learning_rate'],'projection_dim':cc['projection_dim'],
        'temperature':cc['temperature'],'lora_r':cc['lora_r'],'lora_alpha':cc['lora_alpha'],
        'lora_dropout':cc['lora_dropout'],'lora_target_modules':target_modules,
        'model_params':model_params,'projection_params':proj_params,'total_params':total_params,
        'lora_trainable_params':model_trainable,'projection_trainable_params':proj_trainable,
        'trainable_params':trainable_params,'optimizer_steps':step,'history':history,
        'initial_loss':history[0]['loss'] if history else None,'final_loss':history[-1]['loss'] if history else None,
        'nan_or_inf':False,'inner_dev_validation':selected_metrics,'gpu_index':a.gpu,'gpu_name':gpu_name,
        'free_vram_before_bytes':free_before,'total_vram_bytes':total_vram,
        'peak_allocated_bytes':peak_allocated,'peak_reserved_bytes':peak_reserved,
        'training_seconds':train_seconds,
        'parameter_update_check':{'lora_parameter':first_lora_name,
            'lora_max_abs_delta':selected_record['lora_max_abs_delta_from_fresh_init'],
            'projection_max_abs_delta':selected_record['projection_max_abs_delta_from_fresh_init'],
            'final_epoch_lora_max_abs_delta':lora_delta,'final_epoch_projection_max_abs_delta':proj_delta},
        'reload_smoke':reload_smoke,'test_evaluated':False,
        'fresh_lora_initialization_sha256':fresh_lora_initialization_sha256,
        'initialization_source':'DeepSeek-Coder base + get_peft_model fresh LoRA; no old adapter loaded',
        'base_config_sha256':prepared['base_model_config_sha256'],
        'base_weight_sha256':prepared['base_model_weight_sha256'],
        'train_manifest_sha256':prepared['clean_pairs_sha256'],
        'inner_dev_identity_sha256':prepared['stage1_inner_dev_identity_sha256'],
        'selected_epoch':best_epoch,'selection_policy':cc['checkpoint_selection'],
        'outer_DEV_opened':0,'CALIBRATION_opened':0,
        'COMPROMISED_TEST_V1_opened':0,'FINAL_TEST_V2_opened':0,
        'adapter_sha256':_sha_file(adapter/'adapter_model.safetensors'),
        'projection_sha256':_sha_file(projection_path),
    }
    dump_json(outdir/'training_report.json',report)
    print(json.dumps({'CONTRASTIVE_TRAINING':'DONE','steps':step,'inner_dev':selected_metrics,'selected_epoch':best_epoch,'adapter':str(adapter),'reload_smoke':reload_smoke},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
