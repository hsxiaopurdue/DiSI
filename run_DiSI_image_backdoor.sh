for expected_batchsize in 500
do

for EPOCH in 10
do

for lr in 0.1
do

python train_DiSI_image_backdoor.py --expected_batchsize $expected_batchsize --EPOCH $EPOCH --lr $lr --log_dir logs_DiSI_image_backdoor --attack AdapBlend --dataset cifar100 --model vit --accum_num 10 --accum_enabled True --defense majority --use_abl_loss False --abl_drop_ratio 0.2

done
done
done
