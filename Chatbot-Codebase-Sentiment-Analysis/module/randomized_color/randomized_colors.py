import random

def random_color():
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))

def color_brightness(hex_color):
    rgb = [int(hex_color[i:i+2], 16) for i in (1, 3, 5)]
    return (rgb[0] * 299 + rgb[1] * 587 + rgb[2] * 114) / 1000

def is_valid_color(hex_color):
    brightness = color_brightness(hex_color)
    return 30 < brightness < 230  # Filter out very dark and very light colors

def get_valid_random_color():
    while True:
        color = random_color()
        if is_valid_color(color):
            return color