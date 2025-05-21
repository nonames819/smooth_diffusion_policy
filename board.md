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
CUDA_VISIBLE_DEVICES=3 python train.py --config-dir=diffusion_policy/config --config-name=train_diffusion_unet_ddim_hybrid_workspace.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='ckpt/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

- eval 

python eval.py --checkpoint ckpt/2025.05.08/00.57.49_train_diffusion_unet_hybrid_square_image/checkpoints/epoch=3050-test_mean_score=0.940.ckpt --output_dir ckpt/eval_output/square_eval_output --device cuda:0

python eval_test_time.py --checkpoint ckpt/2025.05.08/00.57.49_train_diffusion_unet_hybrid_square_image/checkpoints/epoch=3050-test_mean_score=0.940.ckpt --output_dir ckpt/eval_output/square_eval_output --device cuda:0


python eval.py --checkpoint ckpt/official/data/experiments/low_dim/square_ph/diffusion_policy_cnn/train_0/checkpoints/epoch=1750-test_mean_score=1.000.ckpt --output_dir ckpt/eval_output/square_eval_output --device cuda:0

**Remember to change the project name of wandb, the val freq and the ckpt freq**