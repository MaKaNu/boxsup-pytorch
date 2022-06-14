"""Generates Dummy Images Based on simple Forms."""
from argparse import ArgumentParser
from pathlib import Path
from random import randint

from PIL import Image, ImageDraw
from bs4 import BeautifulSoup


ASCII_CHARS = "@#S%?*+;:,. "
OBJECT_TYPES = ["circle", "poly3", "poly4", "poly5", "poly6"]

XML_ANNOTATION = """
    <annotation>
        <folder></folder>
        <filename></filename>
        <path></path>
        <source>
            <database>Unknown</database>
        </source>
        <size>
            <width>20</width>
            <height>20</height>
            <depth>1</depth>
        </size>
        <segmented>0</segmented>
    </annotation>
"""

XML_BASE_OBJECT = """
        <object>
            <name></name>
            <pose>Unspecified</pose>
            <truncated>1</truncated>
            <difficult>0</difficult>
            <bndbox>
                <xmin></xmin>
                <ymin></ymin>
                <xmax></xmax>
                <ymax></ymax>
            </bndbox>
        </object>
"""


def pixels_to_ascii(image):
    pixels = image.getdata()
    chars = "".join([ASCII_CHARS[pixel//5500] for pixel in pixels])
    return chars


def create_bbox_of_object(center, radius, object_type):
    base_object = BeautifulSoup(XML_BASE_OBJECT, "html.parser")

    # Specify Values of BBox
    xmin = max(0, center[0] - radius)
    ymin = max(0, center[1] - radius)
    xmax = max(0, center[0] + radius)
    ymax = max(0, center[1] + radius)
    base_object.object.select_one('name').string = object_type
    base_object.object.bndbox.xmin.string = str(xmin)
    base_object.object.bndbox.ymin.string = str(ymin)
    base_object.object.bndbox.xmax.string = str(xmax)
    base_object.object.bndbox.ymax.string = str(ymax)

    return base_object


def main():
    parser = ArgumentParser("Generate Dummy Data")
    parser.add_argument(
        '-p', '--path',
        type=Path,
        help="Path to imagefolder",
        default=Path("./boxsup_pytorch/data/datasets/dummy")
    )
    parser.add_argument(
        '-n', '--num_images',
        type=int,
        help="Number of images to create",
        default=5
    )
    parser.add_argument(
        '-d', '--debug',
        type=bool,
        help="Defines if debug mode enabled.",
        default=False
    )
    parser.add_argument(
        '--phase',
        type=str,
        choices=['train', 'val', 'test'],
        help="Specify the phase for which the data will be created.",
        default='train'
    )

    args = parser.parse_args()

    # Create Image
    image_width = 20

    for image_idx in range(args.num_images):
        im = Image.new(mode="I", size=(image_width, image_width))
        draw = ImageDraw.Draw(im)

        # Define save path
        num_image_formated = str(image_idx+1).zfill(len(str(args.num_images)))
        image_name = f"dummy_{num_image_formated}.png"
        xml_name = f"dummy_{num_image_formated}.xml"
        save_path = args.path / args.phase
        save_file = save_path / image_name
        xml_file = save_path / xml_name

        # Create annotation File
        annotation = BeautifulSoup(XML_ANNOTATION, "html.parser")

        # Specify Values of Header
        annotation.annotation.folder.string = save_file.parent.name
        annotation.annotation.filename.string = save_file.name
        annotation.annotation.path.string = str(save_file.resolve())

        # Create Objects
        min_objects = 1
        max_objects = 4
        num_objects = randint(min_objects, max_objects)
        if args.debug:
            print(f'Num_objects: {num_objects}')
            print("-"*10)
        for object_idx in range(num_objects):
            max_gray_value = 65355
            min_gray_value = 4000
            graytone = randint(min_gray_value, max_gray_value)
            object_type = OBJECT_TYPES[randint(0, len(OBJECT_TYPES)-1)]
            if args.debug:
                print(f'{object_idx}: {object_type}')
            center = (randint(0, image_width), randint(0, image_width))
            radius = randint(2, 7)

            # Append Object to Annotation
            annotation.annotation.append(
                create_bbox_of_object(center, radius, object_type)
            )

            if object_type == "circle":
                radius = randint(2, 7)
                x0 = center[0] - radius
                y0 = center[1] - radius
                x1 = center[0] + radius
                y1 = center[1] + radius
                draw.ellipse([x0, y0, x1, y1], fill=graytone)
            else:
                bounding_circle = (*center, radius)
                n_sides = int(object_type[-1])
                rot = randint(0, 180)
                draw.regular_polygon(bounding_circle, n_sides, rot, fill=graytone)

        # Save Image
        im.save(save_file, format='png')

        # Save XML
        with open(xml_file, "w") as file:
            file.write(annotation.prettify())

        if args.debug:
            # Convert Image to Ascii
            ascii_pixels = pixels_to_ascii(im)

            # FORMAT
            pixel_count = len(ascii_pixels)
            ascii_image = "\n".join(
                ascii_pixels[i:(i+image_width)] for i in range(0, pixel_count, image_width)
            )

            print(ascii_image)


if __name__ == "__main__":
    main()
