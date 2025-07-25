#instruction to run the script

# conda create -n stllm python=3.8
# conda env list
# conda activate stllm
# pip install -r requirements.txt

# python -m torch.utils.bottleneck train.py --data taxi_drop > bottleneck_profile.txt

# python train.py --data taxi_drop > taxi_drop_train_with_gat.log
# python train.py --data taxi_drop > taxi_drop_train_with_gat.log --resume ./logs/xtaxi_drop/checkpoint_epoch_10.pth
# python test.py --data taxi_drop --checkpoint ./logs/xtaxi_drop/best_model.pth > taxi_drop_test_with_gat.log
# python test.py --data taxi_drop --checkpoint ./logs/xPEMS07/best_model.pth > PEMS07_test_with_gat.log

#Zero-shot testing for bike_drop and bike_pick datasets
# python test_zero_shot.py --data bike_drop --checkpoint ./logs/xtaxi_drop/best_model.pth > bike_drop_zero_shot_test.log
# python test_zero_shot.py --data bike_pick --checkpoint ./logs/xtaxi_drop/best_model.pth > bike_pick_zero_shot_test.log
