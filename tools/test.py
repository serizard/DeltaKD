import argparse
from utils import load_model
from engine import evaluate
import json


def main(args):
    model = load_model(args.checkpoint)

    model.eval() 

    eval_metrics = evaluate(model, args)
    print(json.dumps(eval_metrics, indent=4))

    with open(args.checkpoint.replace('pth', 'json'), 'w') as f:
        json.dump(eval_metrics, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
    args = parser.parse_args()
    main(args)

