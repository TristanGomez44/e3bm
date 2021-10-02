
case $1 in
  "")
    python main.py -backbone resnet12 -shot 5 -way 5 -mode meta_eval -dataset tieredimagenet -gpu 2
    ;;
  "br_npa_eval")
    python main.py -backbone resnet12 -shot 5 -way 5 -mode meta_eval -dataset tieredimagenet -gpu 2 -attention br_npa
    ;;
  "br_npa")
    python main.py -backbone resnet12 -shot 5 -way 5 -mode meta_train -dataset tieredimagenet -gpu 2 -attention br_npa
    ;;
  "bcnn")
    python main.py -backbone resnet12 -shot 5 -way 5 -mode meta_eval -dataset tieredimagenet -gpu 2 -attention bcnn
    ;;
  "*")
    echo "no such model"
    ;;
esac




