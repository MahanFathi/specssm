#!/bin/bash
python main.py --gin_file=base.gin --gin_file=task/listops.gin --gin_file=ssm/s5.gin --gin_file=size/tiny.gin --gin_param=train.Trainer.use_wandb=False
