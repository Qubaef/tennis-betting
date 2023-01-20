## Tennis match prediction for betting
This project aims to train a neural network to predict the outcomes of tennis matches. It utilizes historical data on past matches, player statistics, betting odds and other relevant information to train the model.


## Data generation
To generate data run:
```
py data_generation/src/data_generator.py
```

## Configuration
Configuration is yaml based, project is integrated with wandb, templates are available in:
```
ml/config/run/
ml/config/sweep/
```

## Training
To run training simply run:
```
py tennis-betting/ml/src/training.py --config [CONFIG_PATH]
```

## Results
So far best run using most advantageous historical odds from dataset:
https://wandb.ai/pg-pug-tennis-betting/tennis-betting/runs/yyslrq1n
