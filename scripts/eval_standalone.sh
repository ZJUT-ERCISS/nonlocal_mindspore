echo "================================================================================================"
echo "Please run the script as: "
echo "bash eval_standalone.sh [PROJECT_PATH] [CONFIG_PATH]"
echo "For example: bash eval_standalone.sh /home/nonlocal_mindspore /home/nonlocal_mindspore/src/config/nonlocal.yaml"
echo "================================================================================================"
set -e
if [ $# -lt 2 ]; then
  echo "Usage: bash eval_standalone.sh [PROJECT_PATH] [CONFIG_PATH]"
exit 1
fi

PYTHON_PATH=$1
CONFIG_PATH=$2

export PYTHONPATH=$PYTHON_PATH
python $PYTHON_PATH/eval.py -c $CONFIG_PATH >  eval_standalone.log 2>&1
