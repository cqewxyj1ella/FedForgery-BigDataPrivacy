# FedForgery: Generalized Face Forgery Detection with Residual Federated Learning

The zip file contains the source code we used in this paper to test the accuracy of face forgery detection in the Hybrid-domain forgery dataset.

## Dependencies

* Anaconda3 (Python3.6, with Numpy etc.)
* Pytorch 1.10.0
* tensorboardX

More details about dependencies are shown in requirements.txt.
```shell
pip install -r requirements.txt
```

## Datasets

[Hybrid-domain forgery dataset] Combine four diverse forgery subtypes of the FF++ dataset and the WildDeepfake dataset,
into the whole dataset,
with five different artifact types.
The training set contains 20,000 images, where true images have the same number of fake images;
the ratio of the training set and testing set is kept at 7: 3.

[Larger dataset download from Kaggle](https://www.kaggle.com/datasets/arunnishanthan/split-23-40)

We use `split-23`. There are 35964 images for training, 6998 images for evaluation, and 6994 images form testing.

## Usage

### Download original pretrained model

| Model   | Download                                                     |
| ------- | ------------------------------------------------------------ |
| FedForgery | [MEGA](https://mega.nz/file/Ba0R1C4I#nRVi0u5Am9zuK_5TOvms8eCsYyxHyqLqwoj1aOgbH80) |

After downloading the pretrained model, we should put the model to `./pretrained`

### Download dataset

| Dataset Name | Download                                                   | Images  |
| ------------ | ---------------------------------------------------------- | ------- |
| Hybrid-domain forgery dataset | [Hybrid-domain forgery- dataset](https://mega.nz/file/9b9GGQqL#cfNu3PQ05Ssg68OHakK-h_Ghm97E2stD3vojmhNYxuU) | 4,2800 |
| Kaggle dataset | [Kaggle train eval test](https://www.kaggle.com/datasets/arunnishanthan/split-23-40) | 49956 in total |

After downloading the whole dataset, you can unzip **test.zip** to the `./testset`.


### Train central model

Simply call this command in the terminal:
```shell
python train_central.py
```

If you want to modify arguments, please check `options.py`


### Train FL model

You need to first open a terminal to start server process:
```shell
python train_flower_server.py 
```

Then, open another two terminals, to start client processes:
```shell
# terminal 1 for client 1
python train_flower_client.py --epochs 2 --cid 0 --K 2

# terminal 2 for client 2
python train_flower_client.py --epochs 2 --cid 1 --K 2
```

`cid` means the id of this client process, and starts from 0; `K` means how many clients in total.

If you want to add more clients, you can modify `min_xxx_clients` in `DP/train_flower_server.py`:

```python
# Define strategy
strategy = SaveModelStrategy(
    evaluate_metrics_aggregation_fn=weighted_average,
    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=2,
)
```

### Train FL model with DP protection
To do this, you need to start flower process by simulation:
```shell
# in the root folder of this repository
flower-simulation --server-app DP.server:app --client-app DP.client:app --num-supernodes 10
```

`--num-supernodes` means how many clients you want to simulate.

The server application starts according to the strategy specified in `DP/server.py`;
and client applications start according to the code `DP/client.py` and `DP/task.py`.

You may need to change `MIN_CLIENTS` and `NORM` parameters set in `DP/task.py`.

Besides, it's possible that the code won't run successfully.
You may need to modify Flower's source code.

* Change maximum message length in `grpc.py`:
Here's the path of the code you would modify:
```shell
# path 1 to modify the code
~/miniconda3/envs/<your venv name>/lib/python3.10/site-packages/flwr/common/grpc.py
```
Here's the line you would modify:
```python
# code to modify: Line 25
GRPC_MAX_MESSAGE_LENGTH: int = 2_073_741_824  # == 2 * 1024 * 1024 * 1024
```
* Change stochastic round function
Here's the path of the code you would modify:
```shell
# path 1 to modify the code
~/miniconda3/envs/<your venv name>/lib/python3.10/site-packages/flwr/common/secure_aggregation/quantization.py
```
Here's the line you would modify:
```python
# code to modify: starts from Line 24, the whole function
def _stochastic_round(arr: NDArrayFloat) -> NDArrayInt:
    ret: NDArrayInt = np.ceil(arr).astype(np.int32)
    # rand_arr: NDArrayFloat = np.random.rand(*ret.shape)
    # traverse ret and change value -= 1 if rand_arr < arr - ret
    shape = ret.shape
    ret2 = ret.flatten()
    arr2 = arr.flatten()
    # AttributeError: 'float' object has no attribute 'flatten'
    for i in range(len(ret2)):
        rand_num = np.random.rand()
        if rand_num < arr2[i] - ret2[i]:
            ret2[i] -= 1
    ret = ret2.reshape(shape)
    ret = ret.astype(np.int32)
    # np.int32 is not suitable for assignment
    return ret
```

### Test the model

```
./run_test.sh
```

But you should modify some relative path of model checkpoint in the test.py

## BibTeX
```

@article{liu2023fedforgery,
  title={FedForgery: generalized face forgery detection with residual federated learning},
  author={Liu, Decheng and Dang, Zhan and Peng, Chunlei and Zheng, Yu and Li, Shuang and Wang, Nannan and Gao, Xinbo},
  journal={IEEE Transactions on Information Forensics and Security},
  year={2023},
  publisher={IEEE}
}
```
