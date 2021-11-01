
#2405,2410,3368,6622,6632,3238,3365,6735,6745
case $1 in 
  "")
    python main.py -backbone resnet12 -shot 5 -way 5 -mode meta_eval -dataset tieredimagenet -gpu 1 -ind_for_viz 3946 2085 2090 2119 2124 2129 2116 2121 2131 -only_viz -exp_id tieredimagenet -model_id baseline -rise -seed 1
    python main.py -backbone resnet12 -shot 5 -way 5 -mode meta_eval -dataset tieredimagenet -gpu 1 -ind_for_viz 3946 2085 2090 2119 2124 2129 2116 2121 2131 -only_viz -exp_id tieredimagenet -model_id baseline -noise_tunnel -seed 1
    python main.py -backbone resnet12 -shot 5 -way 5 -mode meta_eval -dataset tieredimagenet -gpu 1 -ind_for_viz 3946 2085 2090 2119 2124 2129 2116 2121 2131 -only_viz -exp_id tieredimagenet -model_id baseline -seed 1
    ;;
  "norm")
    python main.py -backbone resnet12 -shot 5 -way 5 -mode meta_eval -dataset tieredimagenet -gpu 1 -exp_id tieredimagenet -model_id baseline -rise -seed 1
    ;;
  "new_inds")
    python main.py -backbone resnet12 -shot 5 -way 5 -mode meta_eval -dataset tieredimagenet -gpu 1 -ind_for_viz 2405 2410 3368 6622 6632 3238 3365 6735 6745 -only_viz -exp_id tieredimagenet -model_id baseline -rise -seed 1
    python main.py -backbone resnet12 -shot 5 -way 5 -mode meta_eval -dataset tieredimagenet -gpu 1 -ind_for_viz 2405 2410 3368 6622 6632 3238 3365 6735 6745 -only_viz -exp_id tieredimagenet -model_id baseline -noise_tunnel -seed 1
    python main.py -backbone resnet12 -shot 5 -way 5 -mode meta_eval -dataset tieredimagenet -gpu 1 -ind_for_viz 2405 2410 3368 6622 6632 3238 3365 6735 6745 -only_viz -exp_id tieredimagenet -model_id baseline -seed 1
    python main.py -backbone resnet12 -shot 5 -way 5 -mode meta_eval -dataset tieredimagenet -gpu 1 -exp_id tieredimagenet -model_id baseline -rise -seed 1
    ;;
  "norm")
    python main.py -backbone resnet12 -shot 5 -way 5 -mode meta_eval -dataset tieredimagenet -gpu 1 -exp_id tieredimagenet -model_id baseline -rise -seed 1
    ;;
  "br_npa_eval")
    python main.py -backbone resnet12 -shot 5 -way 5 -mode meta_eval -dataset tieredimagenet -gpu 2 -attention br_npa
    ;;
  "br_npa_eval_biggererLR")
    python main.py -backbone resnet12 -shot 5 -way 5 -mode meta_eval -dataset tieredimagenet -gpu 2 -attention br_npa -base_lr 0.8
    ;;
  "br_npa_eval_biggerLR")
    python main.py -backbone resnet12 -shot 5 -way 5 -mode meta_eval -dataset tieredimagenet -gpu 2 -attention br_npa -base_lr 0.4
    ;;
  "br_npa_eval_smallerLR")
    python main.py -backbone resnet12 -shot 5 -way 5 -mode meta_eval -dataset tieredimagenet -gpu 2 -attention br_npa -base_lr 0.025
    ;;
  "br_npa")
    python main.py -backbone resnet12 -shot 5 -way 5 -mode meta_train -dataset tieredimagenet -gpu 2 -attention br_npa -val_episode 500 
    ;;
  "br_npa_opt")
    python main.py -backbone resnet12 -shot 5 -way 5 -mode meta_train -dataset tieredimagenet -gpu 2 -attention br_npa -optuna       -exp_id tieredimagenet -model_id nodist -max_epoch 50 -val_episode 500 -num_workers 4
    ;;
  "br_npa_opt_more")
    python main.py -backbone resnet12 -shot 5 -way 5 -mode meta_train -dataset tieredimagenet -gpu 2 -attention br_npa -optuna       -exp_id tieredimagenet -model_id nodist -max_epoch 50 -val_episode 500 -num_workers 4 -more_params -optuna_trial_nb 40
    ;;
  "br_npa_dist")
    python main.py -backbone resnet12 -shot 5 -way 5 -mode meta_train -dataset tieredimagenet -gpu 0 -attention br_npa -optuna -dist -exp_id tieredimagenet -model_id dist -max_epoch 50 -val_episode 500  -num_workers 4
    ;;
  "br_npa_dist_more")
    python main.py -backbone resnet12 -shot 5 -way 5 -mode meta_train -dataset tieredimagenet -gpu 1 -attention br_npa -optuna -dist -exp_id tieredimagenet -model_id dist -max_epoch 50 -val_episode 500  -num_workers 4 -more_params -optuna_trial_nb 40 -seed 1
    ;;
  "br_npa_dist_more_val")
    python main.py -backbone resnet12 -shot 5 -way 5 -mode meta_train -dataset tieredimagenet -gpu 1 -attention br_npa -optuna -dist -exp_id tieredimagenet -model_id dist -max_epoch 50 -val_episode 500  -num_workers 4 -more_params -optuna_trial_nb 40 -seed 1 -test_on_val
    ;;
  "bcnn")
    python main.py -backbone resnet12 -shot 5 -way 5 -mode meta_eval -dataset tieredimagenet -gpu 2 -attention bcnn
    ;;
 "bcnn_opt")
    python main.py -backbone resnet12 -shot 5 -way 5 -mode meta_train -dataset tieredimagenet -gpu 0 -attention bcnn -optuna       -exp_id tieredimagenet -model_id nodist_bcnn -max_epoch 50 -val_episode 500  -num_workers 4 -seed 1 -optuna_trial_nb 18
    ;;
  "cross_att")
    python main.py -backbone resnet12 -shot 5 -way 5 -mode meta_eval -dataset tieredimagenet -gpu 1 -attention cross
    ;;
 "crossAtt_opt")
    python main.py -backbone resnet12 -shot 5 -way 5 -mode meta_train -dataset tieredimagenet -gpu 1 -attention cross -optuna       -exp_id tieredimagenet -model_id nodist_cross -max_epoch 50 -val_episode 500  -num_workers 4 -seed 1
    ;;
 "crossAtt_opt_more")
    python main.py -backbone resnet12 -shot 5 -way 5 -mode meta_train -dataset tieredimagenet -gpu 1 -attention cross -optuna       -exp_id tieredimagenet -model_id nodist_cross -max_epoch 200 -val_episode 500  -num_workers 4 -seed 1 -optuna_trial_nb 40
    ;;
 "crossAtt_opt_loss")
    python main.py -backbone resnet12 -shot 5 -way 5 -mode meta_train -dataset tieredimagenet -gpu 0 -attention cross -optuna       -exp_id tieredimagenet -model_id nodist_cross_loss -max_epoch 50 -val_episode 500  -num_workers 4 -seed 1 -cross_att_loss -query 13
    ;;
 "crossAtt_opt_loss_more")
    python main.py -backbone resnet12 -shot 5 -way 5 -mode meta_train -dataset tieredimagenet -gpu 0 -attention cross -optuna       -exp_id tieredimagenet -model_id nodist_cross_loss -max_epoch 50 -val_episode 500  -num_workers 4 -seed 1 -cross_att_loss -query 13 -optuna_trial_nb 40 -opt_loss -more_params
    ;;
  "*")
    echo "no such model"
    ;;
esac






