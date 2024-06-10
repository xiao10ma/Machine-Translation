# Facial Expression Recognition

## Installation
```bash
git clone https://github.com/xiao10ma/ANN-hm.git
cd ANN-hm
pip install -r requirements.txt
```

## How to run?

Move the data in to the data directory, it looks like this:
```bash
data
├── test.jsonl
├── train_100k.jsonl
├── train_10k.jsonl
└── valid.jsonl
```

Then, you can run the project with just(default use GRU):
```bash
python train.py
```

You can choose different model(GRU, LSTM) in the main function.
```python
MODEL = 'GRU'  # 'LSTM'
```

To visualize the training process, you can use tensorboard:
```bash
tensorboard --logdir={record_path}
```

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for train.py</span></summary>

  #### --record_path / -r
  Path to the record, you can use tensorboard to visualize it.
  #### --model_path / -m 
  Path where the trained model should be stored (```trained_model/{Modelname}``` by default).
  #### --embedded_size
  Integer to set the embedded word vector size.
  #### --hidden_size
  Integer to set the hidden output size.
  #### --save_ep
  Every save_ep epochs, the program will save the trained model. Default 50.
  #### --save_latest_ep
  Every save_latest_ep epochs, the program will save the trained model. Default 10.

</details>
<br>

If you have any questions, please contact me through email. My email: mazp@mail2.sysu.edu.cn
