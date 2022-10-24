echo "================================================================================================"
echo "Please run the script as: "
echo "bash train_distribute.sh [PROJECT_PATH] [CONFIG_PATH]"
echo "For example: bash train_distribute.sh /home/nonlocal_mindspore /home/nonlocal_mindspore/src/config/nonlocal.yaml"
echo "================================================================================================"
set -e
if [ $# -lt 2 ]; then
  echo "Usage: bash train_distribute.sh [PROJECT_PATH] [CONFIG_PATH]"
exit 1
fi

PYTHON_PATH=$1
CONFIG_PATH=$2

export PATH=/usr/local/openmpi-4.0.3/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/openmpi-4.0.3/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHON_PATH
mpirun -n 8 --allow-run-as-root python $PYTHON_PATH/train.py -c $CONFIG_PATH --is_distribute 1 >  train_distributed.log 2>&1
