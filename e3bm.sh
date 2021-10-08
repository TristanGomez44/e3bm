
case $1 in
  "")
    python main.py -backbone resnet12 -shot 5 -way 5 -mode meta_eval -dataset tieredimagenet -gpu 2
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
    python main.py -backbone resnet12 -shot 5 -way 5 -mode meta_train -dataset tieredimagenet -gpu 2 -attention br_npa -optuna       -exp_id tieredimagenet -model_id nodist -max_epoch 50 -val_episode 500 
    ;;
  "br_npa_dist")
    python main.py -backbone resnet12 -shot 5 -way 5 -mode meta_train -dataset tieredimagenet -gpu 0 -attention br_npa -optuna -dist -exp_id tieredimagenet -model_id dist -max_epoch 50 -val_episode 500 
    ;;
  "bcnn")
    python main.py -backbone resnet12 -shot 5 -way 5 -mode meta_eval -dataset tieredimagenet -gpu 2 -attention bcnn
    ;;
  "cross_att")
    python main.py -backbone resnet12 -shot 5 -way 5 -mode meta_eval -dataset tieredimagenet -gpu 1 -attention cross
    ;;
  "*")
    echo "no such model"
    ;;
esac




