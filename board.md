**Remember to change the project name of wandb**

* Train pusht
python train.py --config-dir=. --config-name=image_pusht_diffusion_policy_cnn.yaml training.seed=42 training.device=cuda:1 hydra.run.dir='ckpt/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

* Train tool_hang
python train.py --config-dir=. --config-name=image_tool_hang_diffusion_policy_cnn.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='ckpt/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

* Train square
python train.py --config-dir=. --config-name=image_square_diffusion_policy_cnn.yaml training.seed=42 training.device=cuda:1 hydra.run.dir='ckpt/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

**Remember to change the project name of wandb, the val freq and the ckpt freq**