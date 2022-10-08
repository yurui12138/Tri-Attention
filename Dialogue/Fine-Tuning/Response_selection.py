import time
import argparse
import pickle
import os
from BERT_finetuning import NeuralNetwork
from setproctitle import setproctitle

setproctitle('BERT_FP')

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#Dataset path.
FT_data={
    'ubuntu': '/mnt/public/home/lvwp/process/UDC_/FP/FP/ubuntu_data/ubuntu_dataset_1M_len.pkl',
    'douban': '/mnt/public/home/lvwp/process/UDC_/FP/FP/douban_data/douban_dataset_1M_len.pkl',
    'e_commerce': '/mnt/public/home/lvwp/process/UDC_/FP/FP/e_commerce_data/e_commerce_dataset_1M_len.pkl'
}
print(os.getcwd())
## Required parameters
parser = argparse.ArgumentParser()
parser.add_argument("--task",
                    default='ubuntu',
                    type=str,
                    help="The dataset used for training and test.")

parser.add_argument("--is_training",
                    action='store_true',
                    help="Training model or testing model?")

parser.add_argument("--batch_size",
                    default=8,                 #32
                    type=int,
                    help="The batch size.")
parser.add_argument("--learning_rate",
                    default=1e-5,
                    type=float,
                    help="The initial learning rate for Adamw.")
parser.add_argument("--epochs",
                    default=3,
                    type=float,
                    help="Total number of training epochs to perform.")
parser.add_argument("--save_path",
                    default="/mnt/public/home/lvwp/process/UDC_/FP/FP/Fine-Tuning/FT_checkpoint/",
                    type=str,
                    help="The path to save model.")
parser.add_argument("--score_file_path",
                    default="/mnt/public/home/lvwp/process/UDC_/FP/FP/Fine-Tuning/scorefile.txt",
                    type=str,
                    help="The path to save model.")
parser.add_argument("--do_lower_case", action='store_true', default=True,
                    help="Set this flag if you are using an uncased model.")
args = parser.parse_args()
args.save_path += args.task + '.' + "0.pt"
args.score_file_path = args.score_file_path
# load bert


print(args)
print("Task: ", args.task)


def train_model(train, dev, test, deep_number):
    model = NeuralNetwork(args=args, deep_number=deep_number)
    model.fit(train, dev, test)


def test_model(test):
    model = NeuralNetwork(args=args)
    model.load_model(args.save_path)
    model.evaluate(test, is_test=True)


if __name__ == '__main__':
    deep_list = [1,2,3,4,5,6]
    start = time.time()
    with open(FT_data[args.task], 'rb') as f:
        train, dev, test = pickle.load(f, encoding='ISO-8859-1')
    # print(train['cr'][12])
    # for i in range(len(train['sep_len'])):
    #     print(train['sep_len'][i])



    if args.is_training==True:
        for deep in deep_list:
            print("deep_number: " + str(deep) + "\n")
            train_model(train, dev, test, deep)
            # test_model(test)
    else:
        test_model(test)

    end = time.time()
    print("use time: ", (end - start) / 60, " min")