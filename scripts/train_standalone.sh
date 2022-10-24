echo "================================================================================================"
echo "Please run the script as: "
echo "bash train_standalone.sh [PROJECT_PATH] [CONFIG_PATH]"
echo "For example: bash train_standalone.sh /home/nonlocal_mindspore /home/nonlocal_mindspore/src/config/nonlocal.yaml"
echo "================================================================================================"
set -e
if [ $# -lt 2 ]; then
  echo "Usage: bash train_standalone.sh [PROJECT_PATH] [CONFIG_PATH]"
exit 1
fi

PYTHON_PATH=$1
CONFIG_PATH=$2

export PYTHONPATH=$PYTHON_PATH
python $PYTHON_PATH/train.py -c $CONFIG_PATH  --is_distribute 0  \
    >  train_standalone.log 2>&1
