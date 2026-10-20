"""Phase 7 expanded PlantVillage HOG/SVM validation and test experiment."""
from __future__ import annotations
import argparse, csv, hashlib, json, platform, sys, time
from collections import Counter, defaultdict
from importlib.metadata import version
from pathlib import Path
from typing import Any
import joblib, numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'ml/src')); sys.path.insert(0,str(Path(__file__).resolve().parent))
from agriguard_ml.preprocessing import BLOCK_NORM,CELLS_PER_BLOCK,HOG_FEATURE_COUNT,HOG_ORIENTATIONS,HOG_TRANSFORM_SQRT,IMAGE_SIZE,PIXELS_PER_CELL
import run_phase5_experiments as phase5
SEED=2026; SPLITS=('train','validation','test')
CLASSES={'Potato___Early_blight','Potato___Late_blight','Potato___healthy','Tomato___Early_blight','Tomato___Late_blight','Tomato___healthy'}
FEATURE_SPEC=phase5.FEATURE_SPEC

def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        r=csv.DictReader(f); return list(r),r.fieldnames or []
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def _json_default(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(f'Cannot serialize {type(value).__name__}')
def write_json(path,obj): path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=_json_default)+'\n',encoding='utf-8')

def verify_manifests(root,splitdir):
    summary=json.loads((splitdir/'summary.json').read_text(encoding='utf-8'))
    ledger,_=read_csv(splitdir/'component_assignments.csv')
    rows={}; hashes={}; seen={}; hsplit=defaultdict(set); csplit=defaultdict(set); gsplit=defaultdict(set)
    ledger_paths={}
    for item in ledger:
        for rel in json.loads(item['member_paths']):
            if rel in ledger_paths:raise ValueError(f'Path occurs twice in component ledger: {rel}')
            ledger_paths[rel]=(item['component_id'],item['split_assignment'])
    for s in SPLITS:
        p=splitdir/f'{s}.csv'; rr,fields=read_csv(p)
        need={'path','label','sha256','component_id','grouping_evidence','split_assignment','grouping_metadata_status','canonical_hash_member'}
        if not need.issubset(fields):raise ValueError(f'Missing manifest columns: {need-set(fields)}')
        if len(rr)!=summary['manifest_rows'][s]:raise ValueError(f'Row count differs from summary in {s}')
        hashes[s]=digest(p)
        for r in rr:
            rel=r['path'].replace('\\','/')
            if rel in seen:raise ValueError(f'Path appears in multiple manifests: {rel}')
            if r['split_assignment']!=s or r['label'] not in CLASSES or rel.split('/',1)[0]!=r['label']:raise ValueError(f'Label or split mismatch: {rel}')
            img=(root/Path(rel)).resolve()
            if not img.is_relative_to(root.resolve()) or not img.is_file():raise FileNotFoundError(img)
            with Image.open(img) as im: im.verify()
            if digest(img)!=r['sha256']:raise ValueError(f'SHA-256 mismatch: {rel}')
            if not r['component_id']:raise ValueError(f'Missing component: {rel}')
            seen[rel]=s; hsplit[r['sha256']].add(s); csplit[r['component_id']].add(s)
            if ledger_paths.get(rel)!=(r['component_id'],s):raise ValueError(f'Component ledger assignment mismatch: {rel}')
            for e in json.loads(r['grouping_evidence']):
                if e['type']=='leaf_group':gsplit[e['id']].add(s)
            if r['grouping_metadata_status']!='valid_leaf_group_id' and s!='train':raise ValueError(f'Unknown-group image not training-only: {rel}')
        rows[s]=rr
    if len(seen)!=summary['source_image_paths'] or set(ledger_paths)!=set(seen) or any(len(v)>1 for bucket in (hsplit,csplit,gsplit) for v in bucket.values()):raise ValueError('Manifest path coverage or grouping isolation failed')
    prior=set()
    for old in SPLITS:
        rr,_=read_csv(ROOT/'ml/data/splits'/f'{old}.csv')
        prior.update(r['path'].replace('\\','/') for r in rr)
    if any(seen.get(p)!='train' for p in prior):raise ValueError('Prior-manifest component is not training-only')
    pairs,_=read_csv(ROOT/'ml/.dev_artifacts/phase6_2/near_duplicate_candidates.csv')
    if len(pairs)!=11 or any(seen.get(p['image_a'])!=seen.get(p['image_b']) for p in pairs):raise ValueError('Candidate-pair grouping check failed')
    counts={s:dict(Counter(r['label'] for r in rr)) for s,rr in rows.items()}
    for s in SPLITS:
        for c,v in summary['class_counts'].items():
            if counts[s].get(c,0)!=v[f'{s}_paths']:raise ValueError(f'Class count mismatch {s}/{c}')
        if sum(r['canonical_hash_member']=='true' for r in rows[s])!=summary['unique_sha256_examples'][s]:raise ValueError(f'Deduplicated count mismatch: {s}')
    expected=CLASSES-{'Potato___healthy'}
    if set(counts['validation'])!=expected or set(counts['test'])!=expected:raise ValueError('Expected five-class validation/test sets')
    if {r['component_id'] for r in rows['validation']} & {r['component_id'] for r in rows['test']}:raise ValueError('Validation/test component overlap')
    checks={
        'all_paths_readable_and_hashes_match':True,
        'labels_match_class_directories':True,
        'component_ledger_matches_manifest_rows':True,
        'components_leaf_groups_and_exact_hashes_are_split_disjoint':True,
        'prior_manifest_components_training_only':True,
        'unknown_group_components_training_only':True,
        'all_11_candidate_pair_constraints_preserved':True,
        'per_class_counts_match_phase6_6_summary':True,
        'validation_test_have_five_classes_without_potato_healthy':True,
    }
    return rows,hashes,{'checks':checks,'class_counts':counts,'summary':summary,'prior_paths':len(prior),'component_count':len(ledger)}

def unique_rows(rows):return [r for r in rows if r['canonical_hash_member'].lower()=='true']
def evaluate(y,pred,labels):
    ids=np.arange(len(labels));pr,rc,f1,sup=precision_recall_fscore_support(y,pred,labels=ids,average=None,zero_division=0)
    present=np.flatnonzero(sup>0); macro=precision_recall_fscore_support(y,pred,labels=present,average='macro',zero_division=0)
    return {'accuracy':float(np.mean(y==pred)),'macro_precision':float(macro[0]),'macro_recall':float(macro[1]),'macro_f1':float(macro[2]),'macro_classes_with_support':[labels[i] for i in present],
      'per_class':{l:{'precision':float(pr[i]),'recall':float(rc[i]),'f1':float(f1[i]),'support':int(sup[i])} for i,l in enumerate(labels)},
      'confusion_matrix':{'labels':labels,'values':confusion_matrix(y,pred,labels=ids).astype(int).tolist()},'example_count':len(y),'predicted_potato_healthy_count':int(np.sum(pred==labels.index('Potato___healthy')))}
def extract(rows,root,tag,color):return phase5.extract_features(rows,root,tag,include_color=color)

def run(args):
    if args.output_dir.exists():raise FileExistsError(f'Refusing to overwrite existing output: {args.output_dir}')
    root=args.data_root.resolve(); splitdir=args.split_dir.resolve()
    rows,manifest_hashes,verification=verify_manifests(root,splitdir)
    labels=sorted(CLASSES); to_id={x:i for i,x in enumerate(labels)}
    fit={s:unique_rows(rows[s]) for s in SPLITS}
    omitted=[]
    for s in SPLITS:
        canonical={r['sha256']:r['path'] for r in fit[s]}
        for r in rows[s]:
            if r['canonical_hash_member'].lower()!='true':
                if r['sha256'] not in canonical:raise ValueError(f'No canonical hash row for {r["path"]}')
                omitted.append({'split':s,'sha256':r['sha256'],'label':r['label'],'retained_path':canonical[r['sha256']],'omitted_path':r['path']})
    if len(omitted)!=verification['summary']['extra_exact_duplicate_paths']:raise ValueError('Exact-duplicate omission count mismatch')
    specs={
      'hog_svm':{'features':'hog','svc':{'kernel':'rbf','C':1.0,'gamma':'scale','class_weight':'balanced','probability':False,'random_state':SEED}},
      'hog_hsv_svm':{'features':'hog_hsv','svc':{'kernel':'rbf','C':1.0,'gamma':'scale','class_weight':'balanced','probability':False,'random_state':SEED}},
    }
    cache=args.cache_dir.resolve(); cache.mkdir(parents=True,exist_ok=True)
    key=hashlib.sha256(json.dumps({'manifests':manifest_hashes,'train':[r['path'] for r in fit['train']],'validation':[r['path'] for r in fit['validation']],'features':FEATURE_SPEC},sort_keys=True).encode()).hexdigest()[:20]
    cp=cache/f'phase7_{key}.npz'; mp=cp.with_suffix('.json')
    if cp.is_file() and mp.is_file():
        meta=json.loads(mp.read_text(encoding='utf-8'))
        if meta.get('cache_key')!=key:raise ValueError('Feature cache key mismatch')
        ar=np.load(cp,allow_pickle=False); xth,xtc,ytl=ar['train_hog'],ar['train_hsv'],ar['train_labels'].tolist(); xvh,xvc,yvl=ar['validation_hog'],ar['validation_hsv'],ar['validation_labels'].tolist(); ext=meta['extraction_seconds']
        if ytl!=[r['label'] for r in fit['train']] or yvl!=[r['label'] for r in fit['validation']]:raise ValueError('Cached feature labels/order mismatch')
    else:
        xth,xtc,ytl,dt=extract(fit['train'],root,'Phase 7 train',True)
        xvh,xvc,yvl,dv=extract(fit['validation'],root,'Phase 7 validation',True)
        if xth.shape[1]!=HOG_FEATURE_COUNT or xvh.shape[1]!=HOG_FEATURE_COUNT or xtc.shape[1]!=32 or xvc.shape[1]!=32:raise ValueError('Unexpected Phase 5 feature dimensions')
        ext={'train_hog':dt['hog'],'train_hsv':dt['color_histogram'],'validation_hog':dv['hog'],'validation_hsv':dv['color_histogram'],'total':sum(dt.values())+sum(dv.values())}
        tmp=cp.with_suffix('.tmp.npz')
        with tmp.open('wb') as f:np.savez_compressed(f,train_hog=xth,train_hsv=xtc,train_labels=np.asarray(ytl),validation_hog=xvh,validation_hsv=xvc,validation_labels=np.asarray(yvl))
        tmp.replace(cp); write_json(mp,{'cache_key':key,'manifest_sha256':manifest_hashes,'feature_spec':FEATURE_SPEC,'extraction_seconds':ext,'train_paths':[r['path'] for r in fit['train']],'validation_paths':[r['path'] for r in fit['validation']]})
    ytr=np.asarray([to_id[x] for x in ytl]); yv=np.asarray([to_id[x] for x in yvl])
    args.output_dir.mkdir(parents=True)
    tmp_models=cache/f'models_{key}';tmp_models.mkdir(exist_ok=True)
    results={}
    for name,spec in specs.items():
        trainx=xth if spec['features']=='hog' else np.hstack((xth,xtc)).astype(np.float32,copy=False)
        validx=xvh if spec['features']=='hog' else np.hstack((xvh,xvc)).astype(np.float32,copy=False)
        model=Pipeline([('scaler',StandardScaler()),('svc',SVC(**spec['svc']))])
        start=time.perf_counter();model.fit(trainx,ytr);fit_s=time.perf_counter()-start
        start=time.perf_counter();pred=model.predict(validx).astype(np.int64);pred_s=time.perf_counter()-start
        m=evaluate(yv,pred,labels)
        results[name]={'feature_set':spec['features'],'feature_count':int(trainx.shape[1]),'svc_parameters':spec['svc'],'class_weight':'balanced from canonical training examples','validation':m,'training_seconds':fit_s,'validation_prediction_seconds':pred_s,'feature_extraction_seconds':ext,'distinct_examples_per_class':{'train':dict(Counter(ytl)),'validation':dict(Counter(yvl))},'validation_true_ids':yv.tolist(),'validation_predicted_ids':pred.tolist()}
        joblib.dump(model,tmp_models/f'{name}.joblib');del model
        print(f"{name}: validation accuracy {m['accuracy']:.4f}, macro F1 {m['macro_f1']:.4f}; fit {fit_s:.2f}s",flush=True)
    selected=max(results,key=lambda n:(results[n]['validation']['macro_f1'],results[n]['validation']['macro_recall'],results[n]['validation']['accuracy']))
    model=joblib.load(tmp_models/f'{selected}.joblib'); selected_spec=specs[selected]
    # Test features are not extracted until model selection is frozen.
    write_json(args.output_dir/'validation_results.json',{'experiment_id':args.experiment_id,'seed':SEED,'selected_candidate':selected,'selection_rule':'Highest validation macro-F1 over supported classes; ties by macro recall then accuracy. Test not used for selection.','potato_healthy_validation_support':0,'candidates':results,'feature_spec':FEATURE_SPEC,'manifest_sha256':manifest_hashes,'manifest_verification':verification,'feature_cache':str(cp)})
    xt_h,xt_c,ytl_test,dt=extract(fit['test'],root,'Phase 7 selected test',selected_spec['features']=='hog_hsv')
    xt=xt_h if selected_spec['features']=='hog' else np.hstack((xt_h,xt_c)).astype(np.float32,copy=False)
    yt=np.asarray([to_id[x] for x in ytl_test]); start=time.perf_counter();yp=model.predict(xt).astype(np.int64);test_pred_s=time.perf_counter()-start
    tm=evaluate(yt,yp,labels)
    joblib.dump(model,args.output_dir/'selected_model.joblib')
    with (args.output_dir/'test_predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=['path','label','true_class_id','predicted_class_id','predicted_label','correct','sha256','component_id'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r,a,b in zip(fit['test'],yt,yp):w.writerow({'path':r['path'],'label':r['label'],'true_class_id':int(a),'predicted_class_id':int(b),'predicted_label':labels[int(b)],'correct':str(int(a)==int(b)).lower(),'sha256':r['sha256'],'component_id':r['component_id']})
    test_payload={'metrics':tm,'true_ids':yt.tolist(),'predicted_ids':yp.tolist(),'true_labels':ytl_test,'predicted_labels':[labels[int(i)] for i in yp],'feature_extraction_seconds':sum(dt.values()),'prediction_seconds':test_pred_s};write_json(args.output_dir/'test_results.json',test_payload)
    # Export selected scikit-learn pipeline and compare labels on actual test feature vectors.
    onnx_result={'status':'failed'}
    try:
        from onnxruntime import InferenceSession
        from skl2onnx import to_onnx
        sample=xth[:1] if selected_spec['features']=='hog' else np.hstack((xth[:1],xtc[:1])).astype(np.float32,copy=False)
        onnx=to_onnx(model,sample.astype(np.float32),target_opset=17,options={'svc__zipmap':False})
        raw=onnx.SerializeToString(); session=InferenceSession(raw,providers=['CPUExecutionProvider']); inp=session.get_inputs()[0]
        expected_dim=HOG_FEATURE_COUNT+(32 if selected_spec['features']=='hog_hsv' else 0)
        if inp.type!='tensor(float)' or len(inp.shape)!=2 or inp.shape[1]!=expected_dim:raise ValueError(f'Unexpected ONNX input: {inp.type} {inp.shape}')
        out=session.run(None,{inp.name:xt.astype(np.float32,copy=False)})
        onnx_ids=np.asarray(out[0],dtype=np.int64).reshape(-1); miss=np.flatnonzero(onnx_ids!=yp)
        onnx_result={'status':'passed' if len(miss)==0 else 'mismatch','input_name':inp.name,'input_type':inp.type,'input_shape':inp.shape,'vector_count':len(yp),'matched_count':int(len(yp)-len(miss)),'mismatch_count':len(miss),'mismatch_indices':miss.tolist(),'python_class_ids':yp.tolist(),'onnx_class_ids':onnx_ids.tolist(),'class_id_to_label':{str(i):x for i,x in enumerate(labels)}}
        if not len(miss):(args.output_dir/'selected_model.onnx').write_bytes(raw)
    except Exception as e:onnx_result={'status':'failed','error':f'{type(e).__name__}: {e}'}
    # Record exact duplicate members excluded from feature fitting and evaluation.
    with (args.output_dir/'duplicate_handling.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['split','sha256','label','retained_path','omitted_path']);w.writeheader();w.writerows(omitted)
    class_map={str(i):x for i,x in enumerate(labels)}
    write_json(args.output_dir/'class_labels.json',{'class_id_to_label':class_map,'validation_test_supported_classes':results[selected]['validation']['macro_classes_with_support']})
    hog_cfg={'image_size':list(IMAGE_SIZE),'resize_interpolation':'Pillow bilinear','grayscale':'Pillow convert L to uint8 0..255; EXIF orientation not applied','orientations':HOG_ORIENTATIONS,'pixels_per_cell':list(PIXELS_PER_CELL),'cells_per_block':list(CELLS_PER_BLOCK),'block_norm':BLOCK_NORM,'transform_sqrt':HOG_TRANSFORM_SQRT,'feature_count':HOG_FEATURE_COUNT,'dtype':'float32'}
    feature_config={'hog':hog_cfg,'hsv_histogram':FEATURE_SPEC['color'],'combined_order':'HOG 8100 followed by H(16 bins), S(8 bins), V(8 bins)','combined_feature_count':HOG_FEATURE_COUNT+32,'scaling':'StandardScaler fitted on canonical training rows only'}
    write_json(args.output_dir/'feature_config.json',feature_config)
    duplicate_info={'manifest_path_counts':{s:len(rows[s]) for s in SPLITS},'unique_sha256_examples':{s:len(fit[s]) for s in SPLITS},'omitted_exact_duplicate_paths':len(omitted),'omission_file':'duplicate_handling.csv','policy':'Use one lexicographically canonical path per SHA-256. Exact duplicate copies omitted from fitting and evaluation. pHash-similar images were not removed.'}
    # Include prior metrics for context only; they are not used for model selection.
    history={}
    p3=ROOT/'ml/models/phase3/metrics.json'
    if p3.is_file():
        obj=json.loads(p3.read_text(encoding='utf-8'));history['phase3_test']=obj.get('test',{})
    p5_candidates=list((ROOT/'ml/models/phase5').rglob('metrics.json')) if (ROOT/'ml/models/phase5').exists() else []
    for p in p5_candidates:
        try:
            obj=json.loads(p.read_text(encoding='utf-8'))
            if 'test' in obj:history[f'phase5_test:{p.parent.name}']=obj['test']
        except Exception:pass
    metadata={'experiment_id':args.experiment_id,'selected_candidate':selected,'selection_rule':'Highest validation macro-F1 across supported classes; ties resolved by macro recall then accuracy. Test not used for selection.','seed':SEED,'dataset_root':str(root),'split_dir':str(splitdir),'manifest_sha256':manifest_hashes,'manifest_verification':verification,'class_id_to_label':class_map,'feature_set':selected_spec['features'],'feature_count':int(xt.shape[1]),'feature_config_file':'feature_config.json','pipeline':['StandardScaler','sklearn.svm.SVC'],'svc_parameters':selected_spec['svc'],'class_weight':'balanced from canonical training rows','duplicate_handling':duplicate_info,'validation_metrics':results[selected]['validation'],'test_metrics':tm,'historical_test_metrics_reference_only':history,'timing_seconds':{'feature_extraction_train_validation':ext,'fit_by_candidate':{k:v['training_seconds'] for k,v in results.items()},'validation_predict_by_candidate':{k:v['validation_prediction_seconds'] for k,v in results.items()},'selected_test_feature_extraction':sum(dt.values()),'test_predict':test_pred_s},'onnx':onnx_result,'library_versions':{'python':platform.python_version(),'numpy':version('numpy'),'scikit_learn':version('scikit-learn'),'scikit_image':version('scikit-image'),'Pillow':version('Pillow'),'onnxruntime':version('onnxruntime'),'skl2onnx':version('skl2onnx')},'limitations':['Validation/test contain only five classes; Potato healthy performance is unmeasured.','PlantVillage held-out results do not establish field-photo performance.','Prior Phase 3/5 metrics are historical and use different evaluation data.']}
    write_json(args.output_dir/'metadata.json',metadata)
    write_json(args.output_dir/'metrics.json',{'validation_candidates':{k:v['validation'] for k,v in results.items()},'selected_candidate':selected,'test':tm,'onnx_parity':onnx_result})
    write_report(args.output_dir/'report.md',args.experiment_id,results,selected,tm,onnx_result,metadata)
    print(json.dumps({'experiment_id':args.experiment_id,'output_dir':str(args.output_dir.resolve()),'selected_candidate':selected,'validation':{k:v['validation'] for k,v in results.items()},'test':tm,'onnx':onnx_result,'duplicates':duplicate_info},indent=2),flush=True)

def write_report(path,experiment_id,results,selected,test,onnx,meta):
    lines=[f'# AgriGuard Phase 7 — {experiment_id}','',f'Selected using validation only: `{selected}`. Test features were computed after model selection was frozen.','', '## Manifest and duplicate handling','',f"Manifest paths: train {meta['duplicate_handling']['manifest_path_counts']['train']}, validation {meta['duplicate_handling']['manifest_path_counts']['validation']}, test {meta['duplicate_handling']['manifest_path_counts']['test']}.",f"Exact duplicate paths excluded from fitting/metrics: {meta['duplicate_handling']['omitted_exact_duplicate_paths']} (details in `duplicate_handling.csv`). Similar pHash candidates were not excluded.",'', '## Validation comparison','', '| Candidate | Features | Accuracy | Macro precision | Macro recall | Macro F1 | Fit seconds | Feature extraction seconds |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for n,v in results.items():
        m=v['validation'];lines.append(f"| {n} | {v['feature_count']} | {m['accuracy']:.4f} | {m['macro_precision']:.4f} | {m['macro_recall']:.4f} | {m['macro_f1']:.4f} | {v['training_seconds']:.2f} | {v['feature_extraction_seconds']['total']:.2f} |")
    for n,v in results.items():
        m=v['validation'];lines.extend(['',f'### {n} per-class validation','', '| Class | Precision | Recall | F1 | Support |','|---|---:|---:|---:|---:|'])
        for c,q in m['per_class'].items():lines.append(f"| {c} | {q['precision']:.4f} | {q['recall']:.4f} | {q['f1']:.4f} | {q['support']} |")
        lines.extend(['','Confusion matrix (rows true, columns predicted), labels: '+', '.join(m['confusion_matrix']['labels']),'','```json',json.dumps(m['confusion_matrix']['values']),'```'])
    lines.extend(['','Macro validation scores cover the five classes with true validation support. Potato healthy has 152 training examples and zero validation/test examples, so it cannot be assessed; any predictions of that class appear in the confusion matrix.','', '## Selection','',f"`{selected}` was chosen by validation macro-F1, with macro recall and accuracy as tie-breakers. Per-class results and error matrices informed review; test metrics were not used.",'', '## Selected model test results','',f"- Accuracy: {test['accuracy']:.4f}",f"- Macro precision (five supported classes): {test['macro_precision']:.4f}",f"- Macro recall (five supported classes): {test['macro_recall']:.4f}",f"- Macro F1 (five supported classes): {test['macro_f1']:.4f}",f"- Examples: {test['example_count']}; predictions of unsupported Potato healthy: {test['predicted_potato_healthy_count']}",'','| Class | Precision | Recall | F1 | Support |','|---|---:|---:|---:|---:|'])
    for c,q in test['per_class'].items():lines.append(f"| {c} | {q['precision']:.4f} | {q['recall']:.4f} | {q['f1']:.4f} | {q['support']} |")
    lines.extend(['','Confusion matrix (rows true, columns predicted), labels: '+', '.join(test['confusion_matrix']['labels']),'','```json',json.dumps(test['confusion_matrix']['values']),'```','', 'Per-image predictions are in `test_predictions.csv`.','', '## ONNX','',f"Status: **{onnx['status']}**."])
    if onnx.get('status')=='passed':lines.append(f"Python and ONNX class IDs matched on {onnx['matched_count']}/{onnx['vector_count']} real held-out feature vectors.")
    if meta['feature_set']=='hog_hsv':lines.append('Browser deployment requires a matching TypeScript HSV histogram extractor and parity tests. The Phase 3 browser model was not replaced.')
    lines.extend(['','## Time and configuration','',f"- Training plus validation feature extraction: {meta['timing_seconds']['feature_extraction_train_validation']['total']:.2f}s (HOG {meta['timing_seconds']['feature_extraction_train_validation']['train_hog']+meta['timing_seconds']['feature_extraction_train_validation']['validation_hog']:.2f}s; HSV {meta['timing_seconds']['feature_extraction_train_validation']['train_hsv']+meta['timing_seconds']['feature_extraction_train_validation']['validation_hsv']:.2f}s).",f"- Fit seconds: `{json.dumps(meta['timing_seconds']['fit_by_candidate'])}`.",f"- RBF SVC with C={meta['svc_parameters']['C']}, gamma={meta['svc_parameters']['gamma']}, balanced class weights; seed {meta['seed']}. StandardScaler fit on training features only.",'', '## Historical comparison and limitations','', 'Phase 3/5 test metrics are historical and not directly comparable: those experiments used other splits, and prior-manifest components are now part of this expanded training set. The new evaluation is a five-class PlantVillage evaluation, not field-photo validation. Independent Potato healthy images reserved from model selection are required for a fair six-class evaluation.',''])
    path.write_text('\n'.join(lines),encoding='utf-8')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,default=ROOT/'ml/data/plantvillage-source/raw/color')
    p.add_argument('--split-dir',type=Path,default=ROOT/'ml/data/splits/phase6_6')
    p.add_argument('--cache-dir',type=Path,default=ROOT/'ml/.dev_artifacts/phase7/cache')
    p.add_argument('--output-dir',type=Path,default=ROOT/'ml/models/phase7/phase7_exp_20261019_v1')
    p.add_argument('--experiment-id',default='agriguard_phase7_exp_20261019_v1')
    args=p.parse_args()
    try:run(args)
    except Exception as e:
        print(f'Phase 7 stopped: {type(e).__name__}: {e}',file=sys.stderr);raise
if __name__=='__main__':main()
