import click

from .dataset import RandomDataset
from .tokenizer import Tokenizer
from .nn import Transformer, LSTM
from .trainer import Trainer
from .util import create_logger, load_config


@click.command()
@click.argument('config')
@click.option('--explain', type=str, default=None)
def main(config, explain=None):
    logger = create_logger()
    config = load_config(config)
    tokenizer = Tokenizer(**config['tokenizer'])
    logger.info('Loading dataset...')
    dataset = RandomDataset.from_config(config, tokenizer=tokenizer)
    logger.info(f'Size: {len(dataset)}')
    val_size = config['dataset']['val_size']
    dataset_train, dataset_val = dataset.split(val_size)
    logger.info(f'Train: {len(dataset_train)}')
    logger.info(f'Val: {len(dataset_val)}')
    logger.info('Training tokenizer...')
    tokenizer.train(dataset_train.text)
    logger.info(f'Chars: {tokenizer.num_chars}')
    if explain is not None:
        dataset_train.explain(explain)
        exit(0)
    model = LSTM.from_config(config, tokenizer=tokenizer)  
    trainer = Trainer(
        logger=logger,
        dataset_train=dataset_train,
        dataset_val=dataset_val,
        model=model,
        **config['train']
    )
    logger.info('Training...')
    trainer.run()

if __name__ == '__main__':
    main()
