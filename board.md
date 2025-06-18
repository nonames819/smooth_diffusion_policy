**Remember to change the project name of wandb**

* Train pusht
python train.py --config-dir=. --config-name=image_pusht_diffusion_policy_cnn.yaml training.seed=42 training.device=cuda:1 hydra.run.dir='ckpt/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

* Train tool_hang
python train.py --config-dir=. --config-name=image_tool_hang_diffusion_policy_cnn.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='ckpt/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

* Train square
- use download config
CUDA_VISIBLE_DEVICES=5 python train.py --config-dir=. --config-name=image_square_diffusion_policy_cnn.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='ckpt/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
CUDA_VISIBLE_DEVICES=6 python train.py --config-dir=. --config-name=image_square_diffusion_policy_transformer.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='ckpt/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

- use repo config
CUDA_VISIBLE_DEVICES=6 python train.py --config-dir=diffusion_policy/config --config-name=train_diffusion_transformer_ddim_hybrid_workspace.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='ckpt/${now:%Y.%m.%d}/${now:%H.%M.%S}_${task_name}_${name}' task="can_image_abs"

- eval 

python eval.py --checkpoint ckpt/2025.05.20/20.09.44_train_diffusion_unet_hybrid_square_image/checkpoints/epoch=1150-test_mean_score=0.900.ckpt --output_dir ckpt/eval_output/square_eval_output --device cuda:0

python eval.py --checkpoint ckpt/official/data/experiments/low_dim/square_ph/diffusion_policy_cnn/train_0/checkpoints/epoch=1750-test_mean_score=1.000.ckpt --output_dir ckpt/eval_output/square_eval_output --device cuda:0

CUDA_VISIBLE_DEVICES=7 python eval_all.py --checkpoint_dir ckpt/2025.05.08/04.31.55_train_diffusion_unet_hybrid_square_image/checkpoints --output_dir ckpt/eval_output/square_eval_output --device cuda:0 

**Remember to change the project name of wandb, the val freq and the ckpt freq**

**In some specific systems, the way how robomimic initial should be modified, especially how mujoco check the device for rendering, file: [PREFIX]/envs/robodiff/lib/python3.9/site-packages/robomimic/envs/env_robosuite.py: line 73**