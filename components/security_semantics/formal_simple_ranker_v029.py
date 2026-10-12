#!/usr/bin/env python3
"""Formal group-CV tensor-only simple binary risk ranking for v0.2.9."""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import torch
from torch import nn


INPUT=Path("/input");WORK=Path("/work")
SEEDS=[20261010,20261011,20261012]
ARMS=["S0_STRUCTURAL","S1_STRUCTURAL_SERAPH","S2_SERAPH_HSYM"]
AUDIT_ARMS=["A_TYPESEM_ONLY","A_LAYOUTSEM_ONLY"]
FAMILIES=["LOGISTIC","MLP"]
SEM_NAMES=["used_TypeSem","used_InterfaceSem","used_LayoutSem","used_APISem","has_type_safety_predicate","type_predicate_executable","typesem_fact_count","known_signedness","has_layout_evidence","known_layout_boundary","interface_role_known","apisem_support","semantic_evidence_ref_count","semantic_support_item_count","semantic_confidence","semantic_available_bias"]
TYPE_IDX=[0,4,5,6,7];LAYOUT_IDX=[2,8,9]
STEPS=500;INNER_STEPS=300;FPR=.10


def sha(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for b in iter(lambda:f.read(4*1024*1024),b""):h.update(b)
 return h.hexdigest()
def dump(path,x):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(x,indent=2,sort_keys=True)+"\n")
def assert_numeric(x,trail="root"):
 if torch.is_tensor(x):
  if x.dtype.is_floating_point and not bool(torch.isfinite(x).all()):raise RuntimeError(f"nonfinite {trail}")
 elif isinstance(x,dict):
  for k,v in x.items():
   if not isinstance(k,str):raise RuntimeError(f"nonnumeric key {trail}")
   assert_numeric(v,f"{trail}.{k}")
 elif isinstance(x,(list,tuple)):
  for i,v in enumerate(x):assert_numeric(v,f"{trail}[{i}]")
 elif not isinstance(x,(int,float,bool,type(None))):raise RuntimeError(f"identity object {trail}: {type(x).__name__}")


def metric(y,s,threshold=.5):
 pred=[int(x>=threshold) for x in s];tp=sum(a and b for a,b in zip(y,pred));fp=sum((not a) and b for a,b in zip(y,pred));tn=sum((not a) and not b for a,b in zip(y,pred));fn=sum(a and not b for a,b in zip(y,pred));pr=tp/(tp+fp) if tp+fp else 0.;re=tp/(tp+fn) if tp+fn else 0.;sp=tn/(tn+fp) if tn+fp else 0.;f1=2*pr*re/(pr+re) if pr+re else 0.
 pos=[x for a,x in zip(y,s) if a];neg=[x for a,x in zip(y,s) if not a];auc=sum(1 if p>n else .5 if p==n else 0 for p in pos for n in neg)/(len(pos)*len(neg))
 groups={}
 for a,x in zip(y,s):groups.setdefault(float(x),[]).append(a)
 hit=seen=0;previous=ap=0.
 for x in sorted(groups,reverse=True):
  g=groups[x];hit+=sum(g);seen+=len(g);cur=hit/len(pos);ap+=(cur-previous)*(hit/seen);previous=cur
 brier=sum((a-x)**2 for a,x in zip(y,s))/len(y)
 return {"auroc":auc,"auprc":ap,"precision":pr,"recall":re,"f1":f1,"balanced_accuracy":(re+sp)/2,"brier":brier,"tp":tp,"fp":fp,"tn":tn,"fn":fn,"n":len(y),"positive":sum(y),"negative":len(y)-sum(y)}


def threshold_at_fpr(y,s):
 valid=[];neg=sum(not x for x in y);pos=sum(y)
 for t in [float("inf")]+sorted(set(s),reverse=True)+[float("-inf")]:
  fp=sum((not a) and x>=t for a,x in zip(y,s));tp=sum(a and x>=t for a,x in zip(y,s));fpr=fp/neg;rec=tp/pos
  if fpr<=FPR:valid.append((rec,-fpr,t))
 return max(valid)[2]


def summarize(graphs,membership):
 offsets=membership["bag_offsets"].tolist();indices=membership["candidate_indices"].tolist();features={k:[] for k in ARMS+AUDIT_ARMS};coverage=[]
 for bag in range(len(offsets)-1):
  gs=[graphs[i] for i in indices[offsets[bag]:offsets[bag+1]]]
  if not gs:raise RuntimeError("unexpected empty development bag")
  cs=torch.stack([g["cand_struct"] for g in gs]);sem=torch.stack([g["cand_semantic"] for g in gs])
  # Exclude TypeSem/InterfaceSem/LayoutSem/APISem node rows from S0.
  structural_nodes=[];hsym=[];avail=[]
  for g in gs:
   keep=g["x_struct"][:,1:5].sum(1)==0
   structural_nodes.append(g["x_struct"][keep]);hsym.append(g["x_hsym"]);avail.append(g["x_hsym_available"])
  node_mean=torch.cat(structural_nodes).mean(0)
  s0=torch.cat([torch.tensor([float(len(gs))]),cs[:,:6].sum(0),cs.mean(0),node_mean])
  sem_mean=sem.mean(0);sem_max=sem.max(0).values;sem_presence=(sem!=0).float().mean(0);sem_all=torch.cat([sem_mean,sem_max,sem_presence])
  hv=torch.cat(hsym);ha=torch.cat(avail);valid=ha[:,0]>0
  hvalid=hv[valid];hsummary=torch.cat([hvalid.mean(0),hvalid.std(0,unbiased=False),torch.tensor([float(valid.float().mean())])])
  features["S0_STRUCTURAL"].append(s0);features["S1_STRUCTURAL_SERAPH"].append(torch.cat([s0,sem_all]));features["S2_SERAPH_HSYM"].append(torch.cat([s0,sem_all,hsummary]))
  def subset(cols):
   ix=torch.tensor(cols);return torch.cat([s0,sem_mean[ix],sem_max[ix],sem_presence[ix]])
  features["A_TYPESEM_ONLY"].append(subset(TYPE_IDX));features["A_LAYOUTSEM_ONLY"].append(subset(LAYOUT_IDX))
  coverage.append({"candidate_count":len(gs),"semantic_nonzero_candidates":int((sem.abs().sum(1)>0).sum()),"semantic_field_nonzero_fraction":(sem!=0).float().mean(0).tolist(),"semantic_field_any":(sem!=0).any(0).long().tolist(),"hsym_node_coverage":float(valid.float().mean()),"hsym_nodes":int(valid.sum()),"total_nodes":int(valid.numel())})
 return {k:torch.stack(v).float() for k,v in features.items()},coverage


class Model(nn.Module):
 def __init__(self,dim,family):
  super().__init__();self.family=family
  self.net=nn.Linear(dim,1) if family=="LOGISTIC" else nn.Sequential(nn.Linear(dim,32),nn.ReLU(),nn.Linear(32,1))
 def forward(self,x):return self.net(x).squeeze(1)


def fit_model(x,y,family,seed,steps):
 torch.manual_seed(seed);m=Model(x.shape[1],family).to(x.device)
 if family=="LOGISTIC":nn.init.zeros_(m.net.weight);nn.init.zeros_(m.net.bias)
 lr=.03 if family=="LOGISTIC" else .003;wd=.001 if family=="LOGISTIC" else .0001
 opt=torch.optim.Adam(m.parameters(),lr=lr,weight_decay=wd);pos=float(y.sum());neg=len(y)-pos;lossfn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg/pos,device=x.device));losses=[];grads=[]
 initial={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
 for _ in range(steps):
  opt.zero_grad(set_to_none=True);loss=lossfn(m(x),y);loss.backward();grads.append(math.sqrt(sum(float(p.grad.detach().pow(2).sum().cpu()) for p in m.parameters() if p.grad is not None)));opt.step();losses.append(float(loss.detach().cpu()))
 update=max(float((v.detach().cpu()-initial[k]).abs().max()) for k,v in m.state_dict().items())
 return m,{"losses":losses,"gradients":grads,"max_update":update,"learning_rate":lr,"weight_decay":wd,"steps":steps}


class Calibrator(nn.Module):
 def __init__(self):super().__init__();self.raw_scale=nn.Parameter(torch.tensor(0.5413248546));self.bias=nn.Parameter(torch.zeros(()))
 def forward(self,z):return torch.nn.functional.softplus(self.raw_scale)*z+self.bias


def fit_calibrator(logits,labels):
 c=Calibrator().to(logits.device);opt=torch.optim.LBFGS(c.parameters(),lr=.2,max_iter=100,line_search_fn="strong_wolfe")
 def closure():opt.zero_grad();loss=nn.functional.binary_cross_entropy_with_logits(c(logits),labels);loss.backward();return loss
 opt.step(closure)
 with torch.no_grad():return c,{"positive_scale":float(torch.nn.functional.softplus(c.raw_scale).cpu()),"bias":float(c.bias.cpu())}


def train_fold(arm,family,outer_fold,seed,x,y,folds,groups,input_hashes):
 train=[i for i,f in enumerate(folds) if f!=outer_fold];held=[i for i,f in enumerate(folds) if f==outer_fold];device=x.device
 # Group-aware inner OOF predictions for Platt calibration and FPR threshold.
 inner_logits={};inner_audit=[]
 outer_groups=sorted(set(groups[i] for i in train))
 for inner_fold in range(3):
  val_groups={g for g in outer_groups if g%3==inner_fold};fit=[i for i in train if groups[i] not in val_groups];val=[i for i in train if groups[i] in val_groups]
  mu=x[fit].mean(0);sd=x[fit].std(0,unbiased=False).clamp(min=1e-6);model,log=fit_model((x[fit]-mu)/sd,y[fit],family,seed+inner_fold,INNER_STEPS)
  with torch.no_grad():pred=model((x[val]-mu)/sd).cpu().tolist()
  for i,z in zip(val,pred):inner_logits[i]=z
  inner_audit.append({"inner_fold":inner_fold,"fit_groups":sorted(set(groups[i] for i in fit)),"calibration_groups":sorted(val_groups),"fit_count":len(fit),"calibration_count":len(val),"final_loss":log["losses"][-1]})
 if set(inner_logits)!=set(train):raise RuntimeError("inner OOF coverage failure")
 z=torch.tensor([inner_logits[i] for i in train],device=device);yt=y[train];cal,cal_info=fit_calibrator(z,yt)
 with torch.no_grad():inner_prob=torch.sigmoid(cal(z)).cpu().tolist()
 fpr_t=threshold_at_fpr([int(v) for v in yt.cpu()],inner_prob)
 mu=x[train].mean(0);sd=x[train].std(0,unbiased=False).clamp(min=1e-6);model,log=fit_model((x[train]-mu)/sd,y[train],family,seed,STEPS)
 with torch.no_grad():raw_train=model((x[train]-mu)/sd);raw_held=model((x[held]-mu)/sd);uncal=torch.sigmoid(raw_held);prob=torch.sigmoid(cal(raw_held))
 run_id=f"fold{outer_fold}_seed{seed}_{family}_{arm}";ck=WORK/"checkpoints"/f"{run_id}.pt";ck.parent.mkdir(parents=True,exist_ok=True)
 torch.save({"model":model.state_dict(),"calibrator":cal.state_dict(),"mu":mu.cpu(),"sd":sd.cpu(),"arm":arm,"family":family,"input_hashes":input_hashes},ck)
 reload=Model(x.shape[1],family).to(device);reload.load_state_dict(torch.load(ck,map_location=device)["model"]);rc=Calibrator().to(device);rc.load_state_dict(torch.load(ck,map_location=device)["calibrator"])
 with torch.no_grad():again=torch.sigmoid(rc(reload((x[held]-mu)/sd)))
 diff=float((again-prob).abs().max());pred=WORK/"predictions"/f"{run_id}.pt";pred.parent.mkdir(parents=True,exist_ok=True)
 torch.save({"bag_indices":torch.tensor(held),"labels":y[held].cpu().long(),"raw_logits":raw_held.cpu(),"uncalibrated_probabilities":uncal.cpu(),"calibrated_probabilities":prob.cpu(),"inner_oof_indices":torch.tensor(train),"inner_oof_probabilities":torch.tensor(inner_prob),"fpr_threshold":fpr_t},pred)
 m=metric(y[held].long().cpu().tolist(),prob.cpu().tolist());u=metric(y[held].long().cpu().tolist(),uncal.cpu().tolist());fm=metric(y[held].long().cpu().tolist(),prob.cpu().tolist(),fpr_t)
 rec={"run_id":run_id,"arm":arm,"family":family,"fold":outer_fold,"seed":seed,"train_count":len(train),"held_count":len(held),"train_groups":sorted(set(groups[i] for i in train)),"held_groups":sorted(set(groups[i] for i in held)),"feature_dim":x.shape[1],"normalization_fit_on_outer_train_only":True,"class_weight_fit_on_train_only":True,"hyperparameters_predeclared":True,"inner_group_calibration":inner_audit,"calibrator":cal_info,"inner_oof_brier":metric([int(v) for v in yt.cpu()],inner_prob)["brier"],"held_uncalibrated":u,"held_calibrated":m,"fixed_fpr":{"threshold_from_inner_oof":fpr_t,"held_recall":fm["recall"],"held_fpr":fm["fp"]/fm["negative"]},"training":log,"checkpoint_sha256":sha(ck),"prediction_sha256":sha(pred),"reload_diff":diff,"reload_passed":diff<=1e-7,"gold_location_features":False,"raw_identity_features":False}
 dump(WORK/"records"/f"{run_id}.json",rec);return rec


def access_gate():
 paths=["/audit_ledger/METADATA_LEDGER.jsonl","/data/chaoni/mjy/resym_exp/training_runs/seraph_task2/v0_2_5/audit_ledger/METADATA_LEDGER.jsonl","/data/chaoni/mjy/resym_exp/training_runs/seraph_task2/v0_1_0/vulnerability_gold.jsonl","/proc/1/root/data/chaoni/mjy/resym_exp/training_runs/seraph_task2/v0_2_5/audit_ledger/METADATA_LEDGER.jsonl"]
 out={}
 for p in paths:
  try:Path(p).open("rb").close();out[p]="ACCESSIBLE"
  except (FileNotFoundError,PermissionError):out[p]="INACCESSIBLE"
 return out


def main():
 graphs=torch.load(INPUT/"tensor_graphs.pt",map_location="cpu");membership=torch.load(INPUT/"bag_membership.pt",map_location="cpu");sup=torch.load(INPUT/"supervision.pt",map_location="cpu");assert_numeric(graphs);assert_numeric(membership);assert_numeric(sup)
 gate=access_gate()
 if any(v!="INACCESSIBLE" for v in gate.values()):raise RuntimeError(gate)
 device=torch.device("cuda:6" if torch.cuda.is_available() and torch.cuda.device_count()>6 else "cpu");features,coverage=summarize(graphs,membership);features={k:v.to(device) for k,v in features.items()};y=sup["binary_labels"].to(device);folds=sup["fold_indices"].tolist();groups=sup["group_codes"].tolist();input_hashes={n:sha(INPUT/n) for n in ("tensor_graphs.pt","bag_membership.pt","supervision.pt")};records=[];started=time.time()
 for family in FAMILIES:
  for arm in ARMS:
   for seed in SEEDS:
    for fold in range(3):
     r=train_fold(arm,family,fold,seed,features[arm],y,folds,groups,input_hashes);records.append(r);print(json.dumps({"done":r["run_id"],"auc":r["held_calibrated"]["auroc"],"brier":r["held_calibrated"]["brier"]}),flush=True)
 for arm in AUDIT_ARMS:
  for seed in SEEDS:
   for fold in range(3):records.append(train_fold(arm,"LOGISTIC",fold,seed,features[arm],y,folds,groups,input_hashes))
 passed=len(records)==72 and all(r["reload_passed"] and r["training"]["max_update"]>0 and min(r["training"]["gradients"])>0 and all(math.isfinite(v) for v in r["training"]["losses"]) for r in records)
 dump(WORK/"V029_TRAINING_EXECUTION.json",{"passed":passed,"runs":len(records),"main_runs":54,"audit_runs":18,"device":str(device),"elapsed_seconds":time.time()-started,"seeds":SEEDS,"folds":3,"steps":STEPS,"inner_steps":INNER_STEPS,"arms":ARMS,"audit_arms":AUDIT_ARMS,"families":FAMILIES,"semantic_names":SEM_NAMES,"type_indices":TYPE_IDX,"layout_indices":LAYOUT_IDX,"coverage":coverage,"input_hashes":input_hashes,"access_gate":gate,"historical_holdouts_read":False,"target_candidate_match_available":False})
 if not passed:raise RuntimeError("v029 training gate failed")


if __name__=="__main__":main()
