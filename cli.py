import click
from .src.run_shading_correction import *

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.group(context_settings=CONTEXT_SETTINGS)
def cli():
    pass

@cli.command(help="Corrects unstitched 3D stacks in LSM or TIFF")
@click.argument('input_path')
@click.option('--output_path', '-o', default='', help='Output file path')
def stack(input_path, output_path):
    stack_shading_correction(input_path, output_path)

@cli.command(help="Corrects stitched 2D image in CZI")
@click.argument('input_path')
@click.option('--output_path', '-o', default='', help='Output file path')
def slice(input_path, input):
    stack_shading_correction(input_path, output_path)
