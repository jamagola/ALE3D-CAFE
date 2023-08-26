#!/usr/bin/env bash

#BSUB -G heas
#BSUB -q pbatch
#BSUB -nnodes 1
#BSUB -W 12:00

PYTHON=~/anaconda3/envs/skl_env/bin/python
DAT=~/draft_s/ALE3DCAFE_PANDAS_aniso_40
# DAT=~/jobs_s/

# cell shape 31 x 31 x 78
dat=cafeMINI_aniso40_hardrule_

device=0
for rot in 1; do # 0 = no rotation matrix encoding, like before; 1=encodes to 9 (3x3)
if [[ $rot == 1 ]]; then
  rot_opt='--data_preprocess=NPS.model.cafe_rnn_model.encode_rotmat'
  nf_out=11 # 
else
  rot_opt=''
  nf_out=5 # 
fi
for ker in 3; do # fix to 3?
for n_in in 3; do # tune
for n_out in 5; do # tune
for nhid in 64; do # tune
for wd in 1e-3 1e-2; do # tune
    for batch in 1; do # tune
    for nmp in 2; do # tune
    for model in predrnn_v1; do # ????
        for noise in 0; do 
        for lr in 1e-2 1e-4 ; do # tune
        for dx in 1; do
        for unet in 1; do # tune
        for RNN in 1; do
            for loss in L2 ; do
            DIR=~/draft_s/test/experiment/${dat}_RNN${model}-batch${batch}_lr${lr}_nin${n_in}_nout${n_out}_noise${noise}_loss${loss}_nmp${nmp}_nhid${nhid}_ker${ker}_RNN${RNN}_dx${dx}_unet${unet}_wd${wd}_rot${rot}
            mkdir -p $DIR
            echo $DIR $device
            CUDA_VISIBLE_DEVICES=$device $PYTHON -m NPS.main --dir=$DIR \
 --data=$DAT --dataset=longclip --nfeat_in=$((nf_out+2)) --frame_shape=28,28,76 --nfeat_out=$nf_out --pointgroup=1 --data_slice=':,:,:28,:28,:-2,:' \
 --dim=3 --channel_first $rot_opt \
 --model=NPS.model.ConvRNN --lat_model=$model --act=silu --nfeat_hid=$nhid --n_mpassing=$nmp --kernel_size=$ker --RNN=$RNN \
 --dx=$dx \
 --optimizer=adamw --wd=$wd \
 --data_aug=slice --slice_op=28,28,28 \
 --batch=$batch --lr=$lr --n_in=${n_in} --n_out=${n_out} --noise_op="" --nepoch=100 --epoch_size=500 --n_out_valid=30 --loss=$loss \
 --print_freq=500 --valid_freq=1 --scheduler=plateau --lr_decay_patience=8 --n_traj_out=10 \
 --n_out_predict=100 --clip_step_valid=5 --mode=train &>>$DIR/log &
            device=$(((device+1)%4))
done
done
done
done
done
done
done
done
done
done
done
done
done
done
done
wait
