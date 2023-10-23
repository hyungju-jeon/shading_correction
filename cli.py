import click
from src.run_shading_correction import *

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.group(context_settings=CONTEXT_SETTINGS)
def cli():
    pass


@cli.command(help="Corrects unstitched 3D stacks in LSM or TIFF")
@click.argument("input_path")
@click.option("--output_path", "-o", default="", help="Output file path")
def stack(input_path, output_path):
    stack_shading_correction(input_path, output_path)


@cli.command(help="Corrects stitched 2D image in CZI")
@click.argument("input_path")
@click.option("--output_path", "-o", default="", help="Output file path")
def slice(input_path, output_path):
    stack_shading_correction(input_path, output_path)


@cli.command(help="Corrects 3D Imaris Stack")
@click.argument("input_path")
@click.option("--output_path", "-o", default="", help="Output file path")
@click.option("--traing_channels", "-t", default="0,1", help="Channels to train on")
@click.option("--correct_channels", "-c", default="0,1", help="Channels to correct")
@click.option("--reference_channels", "-r", default=0, help="Reference channel")
@click.option(
    "--num_slices", "-n", default=40, help="Number of slices per stack to use for training"
)
def imaris3D(
    input_path, output_path, traing_channels, correct_channels, reference_channels, num_slices
):
    traing_channels = [int(x) for x in traing_channels.split(",")]
    correct_channels = [int(x) for x in correct_channels.split(",")]

    print(correct_channels)
    imaris_shading_correction(
        input_path, output_path, traing_channels, correct_channels, reference_channels, num_slices
    )
