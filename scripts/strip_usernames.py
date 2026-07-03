import json

import click


@click.command()
@click.argument("in_")
@click.argument("out")
def main(in_, out):
    with open(in_) as f:
        j = json.load(f)
    for i, c in enumerate(j):
        if "username" in c:
            j[i] = {k: v for k, v in c.items() if k != "username"}
    with open(out, "w") as f:
        json.dump(j, f)


if __name__ == "__main__":
    main()
