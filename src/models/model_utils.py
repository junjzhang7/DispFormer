from models.my_models.DispFormer import DispFormer


def build_model(args):
    model = DispFormer(
        args,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
    )
    return model
