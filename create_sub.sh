#!/bin/bash

out_dir=$1
epoch=$2

rm -rf submission
mkdir submission
cp custom_player.py submission/
cp q_learn_player.py submission/
cp mcts_player.py submission/
cp hand_eval.py submission/
cp $out_dir/$epoch/Q.npy submission/
cp $out_dir/$epoch/N.npy submission/
cp $out_dir/config.json submission/
zip -r submission.zip submission
