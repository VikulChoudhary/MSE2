"""Load transaction data, train a fraud model, then deploy only if safe."""
import csv
import logging
import math
from pathlib import Path

from .config import Settings
from .database import Store
from .registry import Registry
from .training_model import FraudModel

LOG=logging.getLogger(__name__)
FEATURE_NAMES=["amount","hour","merchant_risk","international","transactions_24h","account_age_days"]
CLASS_NAMES=["legitimate","fraud"]
DATA_FILE=Path(__file__).resolve().parent/"data"/"synthetic_fraud_transactions.csv"


def dataset():
    """Deterministic stratified split, fixed 25% evaluation holdout."""
    with DATA_FILE.open(newline="",encoding="utf-8") as file:
        rows=list(csv.DictReader(file))
    grouped={0:[],1:[]}
    for row in rows:
        label=int(row["is_fraud"])
        grouped[label].append(([float(row[key]) for key in FEATURE_NAMES],label))
    if any(len(g)<8 for g in grouped.values()):
        raise ValueError("Need adequate examples of both classes")
    train=[];valid=[]
    for label in (0,1):
        for index,item in enumerate(grouped[label]):
            (valid if index%4==0 else train).append(item)
    return [r for r,_ in train],[r for r,_ in valid],[y for _,y in train],[y for _,y in valid]


def reference_profile(x):
    out={}
    for j,name in enumerate(FEATURE_NAMES):
        values=sorted(float(row[j]) for row in x)
        n=len(values)
        edges=sorted(set(values[round((n-1)*i/10)] for i in range(1,10)))
        counts=[0]*(len(edges)+1)
        for value in values:
            idx=0
            while idx<len(edges) and value>=edges[idx]:idx+=1
            counts[idx]+=1
        mean=sum(values)/n
        out[name]={"inner_edges":edges,"fractions":[v/n for v in counts],
                   "mean":mean,"std":math.sqrt(sum((v-mean)**2 for v in values)/n)}
    return out


class Trainer:
    def __init__(self,settings:Settings,store:Store,registry:Registry):
        self.settings,self.store,self.registry=settings,store,registry

    @staticmethod
    def evaluate(model,x,y):
        pred=model.predict(x)
        tp=sum(a==1 and b==1 for a,b in zip(y,pred))
        fp=sum(a==0 and b==1 for a,b in zip(y,pred))
        fn=sum(a==1 and b==0 for a,b in zip(y,pred))
        tn=sum(a==0 and b==0 for a,b in zip(y,pred))
        precision=tp/(tp+fp) if tp+fp else 0
        recall=tp/(tp+fn) if tp+fn else 0
        f1=2*precision*recall/(precision+recall) if precision+recall else 0
        return {"accuracy":round((tp+tn)/len(y),5),"fraud_precision":round(precision,5),
                "fraud_recall":round(recall,5),"fraud_f1":round(f1,5),
                "false_positives":fp,"false_negatives":fn,"true_positives":tp,
                "true_negatives":tn,"fraud_prevalence":round(sum(y)/len(y),5)}

    def evaluate_version(self,version:int|None=None)->dict:
        if version is None: model,version=self.registry.active_model()
        else:model=self.registry.load_version(version)
        _,xv,_,yv=dataset()
        return {"version":version,**self.evaluate(model,xv,yv)}

    def train(self,source="manual",deploy=True,use_feedback=True):
        xt,xv,yt,yv=dataset()
        feedback_used=0
        if use_feedback:
            holdout={tuple(row) for row in xv}
            extra=[(r["features"],r["actual"]) for r in self.store.labeled(500)
                   if tuple(r["features"]) not in holdout]
            for row,target in extra:
                xt.append(row);yt.append(target)
            feedback_used=len(extra)
        model=FraudModel().fit(xt,yt)
        metrics=self.evaluate(model,xv,yv)
        metrics.update({"training_rows":len(yt),"feedback_used":feedback_used})
        current=self.registry.active_version()
        baseline=self.store.version(current)["metrics"]["accuracy"] if current is not None else None
        meets_threshold=metrics["accuracy"]>=self.settings.candidate_accuracy_threshold
        fraud_recall_pass=metrics["fraud_recall"]>=self.settings.min_fraud_recall
        no_regression=baseline is None or metrics["accuracy"]>=baseline-self.settings.max_allowed_regression
        accepted=meets_threshold and fraud_recall_pass and no_regression
        if not meets_threshold:note="Rejected: accuracy below threshold"
        elif not fraud_recall_pass:note="Rejected: fraud recall below threshold"
        elif not no_regression:note="Rejected: candidate accuracy regresses against champion"
        elif not deploy:note="Passed gates; evaluated only (not deployed)"
        else:note="Passed quality gates and deployed"
        version=None
        if accepted and deploy:
            version=self.registry.register(model,metrics,reference_profile(xt),len(yt),source)
            self.registry.activate(version,reason=f"quality-gated {source}")
        params={"algorithm":"logistic_regression","backend":model.backend,"epochs":model.epochs,
                "threshold":model.threshold,"validation":"fixed stratified 25% synthetic fraud holdout",
                "feedback_used":feedback_used}
        self.store.add_experiment(source,params,metrics,accepted,note,version)
        self._track_mlflow(source,params,metrics,accepted,note,version)
        return {"accepted":accepted,"deployed":accepted and deploy,"version":version,"backend":model.backend,
                "metrics":metrics,"quality_gates":{"min_accuracy_passed":meets_threshold,
                "min_fraud_recall_passed":fraud_recall_pass,"nonregression_passed":no_regression},"note":note}

    def _track_mlflow(self,source,params,metrics,accepted,note,version):
        if not self.settings.mlflow_tracking_uri:return
        try:
            import mlflow
            mlflow.set_tracking_uri(self.settings.mlflow_tracking_uri)
            mlflow.set_experiment("HELIOS-synthetic-fraud-recovery")
            with mlflow.start_run(run_name=f"{source}-v{version or 'rejected'}"):
                mlflow.log_params({"source":source,**params})
                mlflow.log_metrics({k:v for k,v in metrics.items() if isinstance(v,(int,float))})
                mlflow.set_tags({"gate_accepted":str(accepted),"gate_note":note,
                                 "registry_version":str(version or 'none')})
        except Exception as exc:
            LOG.warning("Optional MLflow tracking unavailable: %s",exc)
