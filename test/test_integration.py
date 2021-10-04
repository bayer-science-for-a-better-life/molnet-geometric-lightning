from molnet_geometric_lightning.train import parse_args, train


def test_integration():
    command_line_args = [
        '--BRO=0.001',
        '--gini=5.0',
        '--max_epochs=1',
        '--n_conv_layers=1',
        '--embedding_dim=32',
        '--dataset_name=esol',
    ]
    args = parse_args(command_line_args)
    train(args)
