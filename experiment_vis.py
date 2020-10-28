"""
Example of running a single experiment of unet in the head and neck data.
The json config of the main model is 'examples/json/unet-sample-config.json'
All experiment outputs are stored in '../../hn_perf/logs'.
After running 3 epochs, the performance of the training process can be accessed
as log file and perforamance plot.
In addition, we can peek the result of 42 first images from prediction set.
"""

from deoxys.experiment import Experiment
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("log_folder")

    args = parser.parse_args()

    exp = Experiment(
        log_base_path=args.log_folder
    )

    # 42 images for 2d, 15 images for 3d
    img_num = 42

    if '3d' in args.log_folder:
        img_num = 15

    # Plot performance
    exp.plot_performance().plot_prediction(
        masked_images=[i for i in range(img_num)])
