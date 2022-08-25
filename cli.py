import click

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.group(context_settings=CONTEXT_SETTINGS)
def cli():
    pass

@cli.command(help="Corrects unstitched 3D stacks in LSM or TIFF")
@click.option('--input', '-i', default='', help='Input file path')
@click.argument('input_path')
def stack(input_path, input):
    click.echo("Hello, {}".format(input))

@cli.command(help="Corrects stitched 2D image in CZI")
@click.option('--input', '-i', default='', help='Input file path')
@click.argument('input_path')
def slice(input_path, input):
    click.echo("goodbye, {}".format(input))
