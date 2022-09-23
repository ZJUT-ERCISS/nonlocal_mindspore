# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
""" nonlocal eval script. """

import argparse
from email.policy import default

from mindspore import context, load_checkpoint, load_param_into_net
from mindspore import nn

from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindspore.train import Model


from src.models.nonlocal3d import nonlocal3d
from src.data.kinetics400 import Kinetic400
from src.data.transforms import VideoCenterCrop
from src.data.transforms import VideoShortEdgeResize, VideoRescale, VideoReOrder, VideoNormalize
from mindspore.train.callback import Callback


class PrintEvalStep(Callback):
    """ print eval step """
    def step_end(self, run_context):
        cb_params = run_context.original_args()
        print("eval: {}/{}".format(cb_params.cur_step_num, cb_params.batch_num))



def nonlocal_eval(args_opt):
    """nonlocal eval."""
    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target)

    # Data Pipeline.
    dataset_eval = Kinetic400(path=args_opt.dataset_path,
                              batch_size=args_opt.batch_size,
                              split='val',
                              shuffle=False,
                              seq=32,
                              num_parallel_workers=args_opt.num_parallel_workers,
                              seq_mode='interval',
                              frame_interval=6
                              )
    transforms = [VideoShortEdgeResize(size=256, interpolation='bicubic'),
                  VideoCenterCrop([256, 256]),
                  VideoRescale(shift=0),
                  VideoReOrder([3, 0, 1, 2]),
                  VideoNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.255])
                  ]
    dataset_eval.transform = transforms
    dataset_eval = dataset_eval.run()

    # Create model.
    network = nonlocal3d(keep_prob=1.0)

    if args_opt.pretrained:
        param_dict = load_checkpoint(args_opt.pretrained_model_dir)
        load_param_into_net(network, param_dict)


    # Define loss function.
    network_loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")

    # Define eval metrics.
    eval_metrics = {'Top_1_Accuracy': nn.Top1CategoricalAccuracy(),
                    'Top_5_Accuracy': nn.Top5CategoricalAccuracy()}

    print_cb = PrintEvalStep()

    # Init the model.
    model = Model(network, loss_fn=network_loss, metrics=eval_metrics)

    # Begin to train.
    print('[Start eval `{}`]'.format(args_opt.model_name))
    result = model.eval(dataset_eval, 
                        dataset_sink_mode=args_opt.dataset_sink_mode,
                        callbacks=[print_cb]
                        )
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nonlocal Eval.")
    parser.add_argument('--device_target', type=str, default="GPU", choices=["Ascend", "GPU", "CPU"])
    parser.add_argument('--run_distribute', type=bool, default=False,help='Distributed parallel training.')
    parser.add_argument('--seq', type=int, default=32, help='Number of frames of captured video.')
    parser.add_argument("--dataset_path", type=str, default="/data/kinetics-dataset")
    parser.add_argument("--model_name", type=str, default="nonlocal")
    parser.add_argument('--num_classes', type=int, default=400, help='Number of classification.')
    parser.add_argument("--device_id", type=int, default=3)
    parser.add_argument("--pretrained", type=bool, default=True, help="Load pretrained model.")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of batch size.")
    parser.add_argument("--pretrained_model_dir", type=str,
                        default="/home/cxhuang/nonlocal/best/nonlocal_ckpt_7/nonlocal-1_4975.ckpt",
                        help="Location of Pretrained Model.")
    parser.add_argument("--dataset_sink_mode", default=False, help="The dataset sink mode.")
    parser.add_argument("--num_parallel_workers", type=int, default=1, help="Number of parallel workers.")

    args = parser.parse_known_args()[0]

    nonlocal_eval(args)
