from argparse import ArgumentParser

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from ogb.graphproppred import Evaluator

from molnet_geometric_lightning.model import Net, MolData


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--batch_size', type=int, default=32,
    )
    parser.add_argument(
        '--dataset_name', type=str, default='hiv'
    )
    parser.add_argument(
        '--dataset_root', type=str, default='data'
    )
    parser.add_argument(
        '--n_runs', type=int, default=1,
    )
    parser.add_argument(
        '--early_stopping', type=int, default=None,
    )

    parser = Net.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    mol_data = MolData(
        root=args.dataset_root,
        name=args.dataset_name,
        batch_size=args.batch_size,
    )

    evaluator = Evaluator(f'ogbg-mol{args.dataset_name}')

    for _ in range(args.n_runs):
        if args.early_stopping is not None:
            early_stopping = EarlyStopping(
                monitor=f'{evaluator.eval_metric}',
                mode='min' if 'rmse' in evaluator.eval_metric else 'max',
                patience=50,
            )
            trainer = Trainer.from_argparse_args(args, callbacks=[early_stopping])
        else:
            trainer = Trainer.from_argparse_args(args)

        trainer.checkpoint_callback.monitor = f'{evaluator.eval_metric}'
        trainer.checkpoint_callback.mode = 'min' if 'rmse' in evaluator.eval_metric else 'max'
        trainer.checkpoint_callback.save_top_k = 1
        model = Net(
            task_type=mol_data.task_type,
            num_tasks=mol_data.num_tasks,
            evaluator=evaluator,
            conf=args,
        )

        trainer.fit(model, mol_data)
        trainer.test()
        del model
        del trainer
        del early_stopping