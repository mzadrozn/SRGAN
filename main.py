from fire import Fire
from scripts.srganSR import SrganSR


def main(config = "train"):
    app = SrganSR(config)
    app.train()


if __name__ == '__main__':
    Fire(main)