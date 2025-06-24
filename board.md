**Remember to change the project name of wandb**

* Train

- use repo config

* robomimic
CUDA_VISIBLE_DEVICES=7 python train.py --config-dir=diffusion_policy/config --config-name=train_diffusion_transformer_ddim_hybrid_workspace.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='ckpt/${now:%Y.%m.%d}/${now:%H.%M.%S}_${task_name}_${name}' task="mimicgen_stack" logging.mode=offline

* mimicgen
CUDA_VISIBLE_DEVICES=3 python train.py --config-dir=diffusion_policy/config --config-name=train_diffusion_transformer_ddim_hybrid_workspace.yaml training.seed=42 hydra.run.dir='ckpt/${now:%Y.%m.%d}/${now:%H.%M.%S}_${task_name}_${name}' logging.group="mimicgen" task="stack_mimic" training.device=cuda:0

* cs
CUDA_VISIBLE_DEVICES=2 python train.py --config-dir=diffusion_policy/config --config-name=train_diffusion_transformer_ddim_hybrid_workspace.yaml training.seed=42 hydra.run.dir='ckpt/${now:%Y.%m.%d}/${now:%H.%M.%S}_${task_name}_${name}' logging.group="cs" task="stack_cs" training.device=cuda:0




* Eval 

python eval.py --checkpoint ckpt/2025.06.16/20.05.57_can_image_train_diffusion_transformer_hybrid/checkpoints/epoch=1350-test_mean_score=1.000.ckpt --output_dir eval_output --device cuda:7

CUDA_VISIBLE_DEVICES=7 python eval_all.py --checkpoint_dir ckpt/2025.05.08/04.31.55_train_diffusion_unet_hybrid_square_image/checkpoints --output_dir ckpt/eval_output/square_eval_output --device cuda:0 

**Remember to change the project name of wandb, the val freq and the ckpt freq**

**In some specific systems, the way how robomimic initial should be modified, especially how mujoco check the device for rendering, file: [PREFIX]/envs/robodiff/lib/python3.9/site-packages/robomimic/envs/env_robosuite.py: line 73**