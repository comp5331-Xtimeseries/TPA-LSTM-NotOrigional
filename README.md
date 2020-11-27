# Pytorch Implementation of TPA-LSTM.

The code was adapted from https://github.com/jingw2/demand_forecast

## Requirements
Please install [Pytorch](https://pytorch.org/) before run it, and

```python
pip install -r requirements.txt
```

## To excute
Use either of the command to train, evaluate and get the result of the specific dataset
```
./run_elec.sh
./run_exchange.sh
./run_solar.sh
./run_traffic.sh
```

# The structure of the folder
- ./data store the downloaded dataset
- ./model store the trained model
- ./result store the training, evaluation and testing result
- main.py execute the training, evaluation and testing process
- tpaLSTM.py has the model structure  

