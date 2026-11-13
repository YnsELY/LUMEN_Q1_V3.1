"""Linux process supervisor. NEVER stops/deletes the RunPod itself.

Cost is an estimate from operator-supplied billing start/rate, not an invoice/API.
"""
from __future__ import annotations
from datetime import datetime,timezone
from pathlib import Path
import math,os,signal,subprocess,time
from .common import finite,integer,utc_now,write_json,read_json
from .environment import resource_sample

def budget_status(cfg,now=None):
    rate=finite(cfg['hourly_total_usd'],'hourly_total_usd',0)
    limit=finite(cfg['remaining_budget_at_start_usd'],'remaining_budget_at_start_usd',1e-9)
    reserve=finite(cfg.get('reserve_usd',1.),'reserve_usd',0)
    if reserve>=limit:raise ValueError('Reserve must be below balance at start')
    start=datetime.fromisoformat(cfg['pod_billing_start_utc'])
    if start.tzinfo is None:raise ValueError('Billing timestamp requires timezone')
    now=now or datetime.now(timezone.utc);elapsed=(now-start).total_seconds()
    if elapsed<0:raise ValueError('Billing start is in the future')
    duration=finite(cfg['max_pod_seconds'],'max_pod_seconds',.001)
    spent=elapsed*rate/3600
    remaining_seconds=min(duration-elapsed, ((limit-reserve-spent)*3600/rate) if rate else math.inf)
    return {'elapsed_since_pod_start_seconds':elapsed,'estimated_spent_usd':spent,
            'seconds_left_before_guard':remaining_seconds,'guard_reached':remaining_seconds<=0,
            'warning':'Stopping child does NOT stop GPU billing. Stop the pod in RunPod console/API separately.'}

def supervise(command,workdir,output,budget_cfg,*,poll_seconds=2,grace_seconds=10):
    if os.name!='posix':raise RuntimeError('Process-tree supervisor targets Linux pods')
    if not command or not all(isinstance(s,str) and s for s in command):raise ValueError('Command must be an argv list')
    poll_seconds=finite(poll_seconds,'poll_seconds',.01);grace_seconds=finite(grace_seconds,'grace_seconds',0)
    folder=Path(output);folder.mkdir(parents=True,exist_ok=True)
    if (folder/'process.json').exists():raise FileExistsError('Use a new supervisor output directory')
    initial=budget_status(budget_cfg)
    if initial['guard_reached']:raise TimeoutError('Budget already exhausted: child not started')
    start=time.monotonic();reason='completed';proc=None
    write_json(folder/'budget.input.json',budget_cfg)
    with (folder/'stdout.log').open('w',encoding='utf-8') as log:
        try:
            proc=subprocess.Popen(command,cwd=workdir,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            write_json(folder/'process.json',{'pid':proc.pid,'command':command,'cwd':str(Path(workdir).resolve()),'created_utc':utc_now()})
            while proc.poll() is None:
                status=budget_status(budget_cfg)
                write_json(folder/'heartbeat.json',{**resource_sample(),**status,'pid':proc.pid})
                if status['guard_reached']:
                    reason='budget_or_walltime_guard';break
                time.sleep(min(poll_seconds,max(.01,status['seconds_left_before_guard'])))
        except BaseException:
            reason='supervisor_interrupted';raise
        finally:
            if proc is not None and proc.poll() is None:
                try:os.killpg(proc.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:proc.wait(timeout=grace_seconds)
                except subprocess.TimeoutExpired:
                    try:os.killpg(proc.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    proc.wait()
            result={'exit_code':None if proc is None else proc.returncode,'reason':reason,'elapsed_seconds':time.monotonic()-start,
                    'pod_stopped':False,'billing_stopped':False,'final_cost_estimate':budget_status(budget_cfg)}
            write_json(folder/'result.json',result)
    return result
