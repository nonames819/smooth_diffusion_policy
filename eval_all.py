"""
Usage:
python eval_all.py --checkpoint_dir data/image/pusht/diffusion_policy_cnn/train_0/checkpoints -o data/pusht_eval_output
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from datetime import datetime
import glob

@click.command()
@click.option('-c', '--checkpoint_dir', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
def main(checkpoint_dir, output_dir, device):
    # if os.path.exists(output_dir):
    #     click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    output_dir = f'{output_dir}/{datetime.now().strftime("%Y.%m.%d/%H.%M.%S")}'
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # find all epoch checkpoints
    checkpoint_dir = pathlib.Path(checkpoint_dir)
    checkpoint_files = sorted(glob.glob(str(checkpoint_dir / "epoch*.ckpt")))
    
    if not checkpoint_files:
        print(f"No epoch checkpoints found in {checkpoint_dir}")
        return
    
    print(f"Found {len(checkpoint_files)} checkpoints to evaluate")
    
    # evaluate each checkpoint
    for checkpoint in checkpoint_files:
        checkpoint_name = pathlib.Path(checkpoint).stem
        print(f"\nEvaluating checkpoint: {checkpoint_name}")
        
        # create output subdirectory for this checkpoint
        checkpoint_output_dir = os.path.join(output_dir, checkpoint_name)
        pathlib.Path(checkpoint_output_dir).mkdir(parents=True, exist_ok=True)
        
        # load checkpoint
        payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
        cfg = payload['cfg']
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg, output_dir=checkpoint_output_dir)
        workspace: BaseWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        
        # get policy from workspace
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model
        
        device = torch.device(device)
        policy.to(device)
        policy.eval()
        
        # run eval
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=checkpoint_output_dir)
        runner_log = env_runner.run(policy)
        
        # dump log to json
        json_log = dict()
        json_log['ckpt_path'] = checkpoint
        for key, value in runner_log.items():
            if isinstance(value, wandb.sdk.data_types.video.Video):
                json_log[key] = value._path
            else:
                json_log[key] = value
        out_path = os.path.join(checkpoint_output_dir, 'eval_log.json')
        json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)
        
        print(f"Evaluation completed for {checkpoint_name}")

if __name__ == '__main__':
    main()
