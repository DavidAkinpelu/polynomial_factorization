# Transformer based Character Sequence to Sequence
A simple transformer model for solving algebraic expressions. The model learns to expand single variable polynomials, where the model takes the factorized sequence as input and predict the expanded sequence. The model was trained on 1 million datasets of polynomial factors and their expansions.  It's strictly for educative purposes.

## Installation

```
$ pip install -r requirements.txt
```

# Train

```
$ python utils\train.py
```
The train module can take several arguments such as `--file`, `--model_name`, `--batch_size`, `--num_epochs`, `--num_encoder_layers`, `--num_decoder_layers`, `--emb_size`, `--n_head`, and `--ffn_hid_dim`.

Although the datasets is not provided, to train the model, the dataset should be named `data.txt` and be copied in the `data` folder.


# Test

If the `test.txt` or `train_txt` is located in the `root` folder, the model can be test using the following command:

```
$ python utils\test.py -t test.txt
```


