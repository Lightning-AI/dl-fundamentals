See https://github.com/Lightning-AI/lightning-hpo



## Run old Lightning-HPO as 

```
python -m lightning run app sweeper.py
```



## Run Trainer Studio App as



```
cd ../lightning-hpo && pip install -e .
```

```
python -m lightning run app app.py
```

```
lightning run sweep mlp_cli2.py \
--n_trials=3 \
--simultaneous_trials=1 \
--logger="tensorboard" \
--direction=maximize \
--cloud_compute=cpu-medium \
--framework="pytorch_lightning" \
--model.lr="log_uniform(0.001, 0.1)" \
--model.hidden_units='categorical(["[50, 100]", "[100, 200]"])' \
--data.batch_size="categorical([32, 64])" \
--trainer.max_epochs="int_uniform(1., 3.)"
```
