"""Bounded, reproducible GPU diagnostic using existing SQuAD annotations only."""
from __future__ import annotations
import argparse,copy,hashlib,json,logging,os,random,time,traceback
from pathlib import Path
import torch,yaml
from dotenv import load_dotenv
from doc_to_atom.atomization.validation import validate_record
from doc_to_atom.config import load_and_resolve_config,validate_config,config_fingerprint_candidates
from doc_to_atom.data.io import read_jsonl,write_jsonl,validate_jsonl
from doc_to_atom.evaluation.baselines import BaseQueryOnly,BaseWithContext
from doc_to_atom.evaluation.cli import generate_rows,score_rows,enrich_rows,_checkpoint_directory
from doc_to_atom.training.checkpoint import load_training_checkpoint
from doc_to_atom.training.cli import build_parser,execute
from doc_to_atom.training.factory import build_system
from doc_to_atom.training.loop import _trainable_modules

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-root",required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    out=Path(args.output_root).resolve()
    out.mkdir(parents=True,exist_ok=True)
    manifest_path=out/"diagnostic_manifest.json"
    if manifest_path.exists():
        raise FileExistsError("Choose a new output root; diagnostic artifacts are never overwritten")
    seed=20261113
    original=root/"data/squad_molab_atoms.jsonl"
    raw_source=root/"data/squad_molab_raw.jsonl"
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    protected={str(p):digest(p) for p in (original,raw_source)}
    def save(name,value):
        p=out/name;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n")
    def phase(name,**details):
        save("status.json",{"phase":name,"updated":time.time(),**details})
        print("PHASE",name,details,flush=True)
    with manifest_path.open("x") as f:
        json.dump({"seed":seed,"input_sha256":protected,"purpose":"bounded diagnostic, not paper reproduction",
                   "code_sha256":{str(p.relative_to(root)):digest(p) for p in (root/"src/doc_to_atom").rglob("*.py")}},f)
    system=None
    try:
        phase("validate_and_prepare")
        load_dotenv(root.parent/".env",override=False);load_dotenv(root/".env",override=False)
        random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(4)
        assert torch.cuda.is_available(),"CUDA GPU required"
        records=[validate_record(r) for r in read_jsonl(original)]
        (out/"data").mkdir()
        write_jsonl(records,out/"data/canonical_atoms.jsonl")
        stats=validate_jsonl(out/"data/canonical_atoms.jsonl",source_path=raw_source,reject_span_mismatch=True)
        assert stats["errors"]==0 and stats["source_answer_mismatches"]==0
        save("data/full_validation.json",stats)
        rng=random.Random(seed)
        train=rng.sample([r for r in records if r.split=="train"],8)
        validation=rng.sample([r for r in records if r.split=="validation"],4)
        chosen=train+validation
        selected_ids={r.record_id for r in chosen}
        sources=[json.loads(l) for l in raw_source.read_text().splitlines() if l.strip()]
        selected_source=[r for r in sources if str(r["id"]) in selected_ids]
        dataset=out/"data/subset_atoms.jsonl";raw_subset=out/"data/subset_source.jsonl"
        write_jsonl(chosen,dataset)
        raw_subset.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in selected_source))
        subset_stats=validate_jsonl(dataset,source_path=raw_subset,reject_span_mismatch=True)
        assert subset_stats["errors"]==0
        save("data/selection.json",{"train":[r.record_id for r in train],
            "validation":[r.record_id for r in validation],"input_sha256":protected,"subset_validation":subset_stats})
        config=load_and_resolve_config(root/"configs/experiments/molab_aligned_qa_v2.yaml",require_runnable=True)
        config["profile"]={"name":"squad-minimax-bounded-e2e","family":"experiments"}
        config["deviations"]=[*config.get("deviations",[]),{
            "id":"BOUNDED-SQUAD-E2E-20261113","description":"8 train / 4 validation records, 16 steps, boundaries 4/8/12, batch 1.",
            "rationale":"Bounded GPU functional experiment; not the paper schedule.",
            "affected_metrics":["answer_f1","refusal_f1","training_loss"]}]
        config["training"].update(seed=seed,max_steps=16,max_epochs=10,micro_batch_size=1,
            gradient_accumulation_steps=1,effective_global_batch=1,warmup_steps=0,
            curriculum_boundaries=[4,8,12],initial_curriculum_step=0,
            validation_every_steps=4,checkpoint_every_steps=4,stage4_only_best=True)
        config["runtime"].update(device="cuda",wandb_mode="disabled",checkpoint_upload="none")
        config["evaluation"].update(max_new_tokens=32,routing_recall_k=8)
        validate_config(config,require_runnable=True)
        (out/"configs").mkdir()
        (out/"configs/diagnostic.yaml").write_text(yaml.safe_dump(config,sort_keys=False))
        smoke=copy.deepcopy(config)
        smoke["training"].update(max_steps=1,validation_every_steps=1,checkpoint_every_steps=1,stage4_only_best=False)
        (out/"configs/smoke.yaml").write_text(yaml.safe_dump(smoke,sort_keys=False))
        phase("load_model")
        system,tokenizer,targets=build_system(config,enable_reranker=True)
        system.eval()
        initial={n:p.detach().cpu().clone() for n,p in system.named_parameters() if p.requires_grad}
        torch.cuda.reset_peak_memory_stats()
        def train_run(name):
            arguments=build_parser().parse_args(["--config",str(out/f"configs/{name}.yaml"),
                "--data",str(dataset),"--output",str(out/name),"--wandb-mode","disabled"])
            return execute(arguments,system_factory=lambda *a,**kw:(system,tokenizer,targets))
        phase("gpu_smoke")
        smoke_result=train_run("smoke")
        assert smoke_result["training"]["optimizer_steps"]==1
        with torch.no_grad():
            for name,p in system.named_parameters():
                if name in initial:p.copy_(initial[name].to(p.device))
        torch.manual_seed(seed)
        phase("train_16_steps")
        trained=train_run("diagnostic")
        assert trained["training"]["optimizer_steps"]==16
        phase("evaluate_best")
        best=out/"diagnostic/checkpoints/best.json"
        assert best.is_file()
        state=load_training_checkpoint(_trainable_modules(system),_checkpoint_directory(best),
            expected_fingerprints={"config":config_fingerprint_candidates(config)},restore_rng=False)
        system.eval()
        with torch.inference_mode():predictions=generate_rows(system,tokenizer,validation,config)
        enrich_rows(predictions,validation)
        save("evaluation/doc_to_atom_predictions.json",predictions)
        metrics={"doc_to_atom":score_rows(predictions)}
        phase("evaluate_base_baselines")
        for name,baseline in [("base_query_only",BaseQueryOnly(system.base_model,tokenizer,32)),
                              ("base_with_context",BaseWithContext(system.base_model,tokenizer,32))]:
            rows=[]
            for record in validation:
                for qa in record.qa_pairs:
                    rows.append({"record_id":record.record_id,"split":record.split,
                        "question_id":qa.question_id,"question":qa.question,"gold":list(qa.answers),
                        "is_irrelevant":qa.is_irrelevant,"is_generated_irrelevant":qa.is_generated_irrelevant,
                        "answer_provenance":qa.answer_provenance,
                        "prediction":baseline.answer(record.context,qa.question)})
            enrich_rows(rows,validation)
            save(f"evaluation/{name}_predictions.json",rows)
            metrics[name]=score_rows(rows)
        assert protected=={str(p):digest(p) for p in (original,raw_source)}
        save("summary.json",{"status":"completed","seed":seed,"train_records":8,"validation_records":4,
            "optimizer_steps":16,"checkpoint_step":state.optimizer_step,"metrics":metrics,
            "gpu":torch.cuda.get_device_name(),"peak_allocated_bytes":torch.cuda.max_memory_allocated(),
            "model_revision":getattr(system.base_model.config,"_commit_hash",None),
            "warning":"Tiny diagnostic subset; metrics are not a full SQuAD or paper benchmark."})
        phase("completed",checkpoint_step=state.optimizer_step)
    except Exception as exc:
        phase("failed",error_type=type(exc).__name__,error=str(exc)[:500])
        traceback.print_exc();raise
    finally:
        if system is not None:system.injector.close()

if __name__=="__main__":
    logging.basicConfig(level=logging.INFO)
    main()
