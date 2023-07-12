import copy
import json
import os
import pickle
import random
import string
import time
from concurrent.futures.thread import ThreadPoolExecutor

import freetype
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from cv2 import cv2
from tqdm import tqdm

WINDOWS_INVALID_CHAR = {"<": "less than", ">": "more than", ":": "colon ", '"': "double quote", ".": "point",
                        "\\": "backslash", "/": "slash", "|": "bar", "?": "question mark", "*": "asterisk"}

def load_attrib():
    # All text characteristics for writing
    with open('raw_data/text_atrib.json') as tf:
        TXT_ATRIB = json.load(tf)
    return TXT_ATRIB


def save_model(model):
    with open("model_trained.p", "wb") as pf:
        pickle.dump(model, pf)


def to_categorical(y_classes: np.ndarray):
    characters = np.array(list(string.ascii_lowercase +
        string.ascii_uppercase +
        string.digits
    ))
    characters.sort()
    one_hot_array = np.zeros((y_classes.size, characters.size))
    one_hot_array[np.arange(y_classes.size), np.searchsorted(characters, y_classes)] = 1
    return one_hot_array


def separate_characters(image, model, threshold=0.65):
    letters_predicted = []
    class_index_predicted = []
    bboxes = get_bbox(image)

    for bbox in bboxes:
        x, y, w, h = bbox
        letter_image = image[y:y+h, x:x+w]
        letter, class_index = predict_image(letter_image, model, threshold=threshold)
        letters_predicted.append(letter)
        class_index_predicted.append(class_index)
    return letters_predicted, class_index_predicted

def predict_image(letter_image, model, threshold=0.65):

    letter_image = cv2.resize(letter_image, (28, 28))

    letter_image = 255 - letter_image
    letter_image[letter_image < 100] = 0
    letter_image[letter_image > 100] = 255

    letter_image = pre_processing(letter_image)
    letter_image = letter_image.reshape(1, 28, 28, 1)

    predictions = model.predict(letter_image)
    class_index = np.argmax(predictions, axis=1)[0]

    probVal = np.amax(predictions)

    if probVal > threshold:
        return from_categorical(class_index), class_index

    return (None, None)


def from_categorical(prediction: int):
    characters = np.array(list(string.ascii_lowercase +
                               string.ascii_uppercase +
                               string.digits
                               ))
    characters.sort()

    return characters[prediction]


def pre_processing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img


def get_symbol_name(letter):
    if letter in WINDOWS_INVALID_CHAR.keys():
        group_name = WINDOWS_INVALID_CHAR[letter]
    else:
        group_name = letter
    return group_name


def get_symbol_from_name(name):
    inv_map = {v: k for k, v in WINDOWS_INVALID_CHAR.items()}

    if name in inv_map.keys():
        return inv_map[name]

    return name[0]


def record_data():
    paths = {}

    for root, _, files in tqdm(os.walk(f"../data/pil_data")):
        if not len(files):
            continue

        letter = get_symbol_from_name(root.split('/')[2])
        if letter not in paths.keys():
            paths.update({letter: []})

        for file in files:
            file_path = os.path.join(root, file)
            paths[letter].append(file_path)

    with open("record_piaal.json", 'w') as rec:
        json.dump(paths, rec, indent=4)

s = set()
def get_bbox(img):

    bbox = []
    # Turn image grayscale, Otsu's threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dilate = cv2.dilate(thresh, kernel, iterations=3)

    # Find contours, obtain bounding box coordinates, and extract ROI
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    # Filter using contour area and extract ROI
    for c in cnts:

        x, y, w, h = cv2.boundingRect(c)
        bbox.append((x, y, w, h))
        return bbox

    cv2.destroyAllWindows()
    return bbox


def get_image_path(dir_path, file_name, home_dir=None):
    try:
        try:
            if str(dir_path[1]).lower() in ("bold", "italic", "bold italic"):
                dir_path = os.path.join(f"{dir_path[0]}", dir_path[1].upper(), dir_path[2])
            elif dir_path[1] >= 16:
                dir_path = os.path.join(dir_path[0], "ITALIC", dir_path[2])
            else:
                dir_path = os.path.join(dir_path[0], "REGULAR", dir_path[2])

            if home_dir is not None:
                dir_path = os.path.join(home_dir, dir_path)

        except TypeError:
            dir_path = os.path.join(home_dir, dir_path[0], "REGULAR", dir_path[2])

        if not os.path.exists(dir_path):
            time.sleep(random.randint(0, 3))
            os.makedirs(dir_path, exist_ok=True)

    except PermissionError:
        dir_path = os.path.join(WINDOWS_INVALID_CHAR[dir_path[0]], "REGULAR", dir_path[2])
        os.makedirs(dir_path, exist_ok=True)

    file_path = os.path.join(dir_path, file_name)

    return file_path


def write_letter(letter, image, formats, home_dir=None):
    # Load all attributes
    TXT_ATRIB = load_attrib()

    # Get symbol name
    group_name = get_symbol_name(letter)
    try:
        # Iterate array with all formats combination
        for i, form in enumerate(formats):
            img = copy.deepcopy(image)

            # Get fonts features for drawing the letter
            font = form[0]
            thickness = form[2].astype(int)
            font_size = form[1].astype(float)

            if isinstance(image, Image.Image):
                # Prepare image for drawing
                draw = ImageDraw.Draw(img)

                # Get font type
                font_path = os.path.join(TXT_ATRIB['pil']['FONTS_PATH'], font)

                # Set text font and font size
                f = ImageFont.truetype(font_path, int(font_size) * 10)

                # Get text dimesions
                _, _, w, h = draw.textbbox((0, 0), letter[0], font=f)
                ascent, descent = f.getmetrics()

                # Compute text position for drawin
                org = (img.width / 2, img.height / 2 - h / font_size - ascent / font_size / 2 + descent / font_size)

                # Draw letter
                draw.text(org, letter[0], font=f, fill=(0, 0, 0), anchor='mm',
                          spacing=0, align='center', stroke_width=0)

                # Get font type name
                type_font = str(freetype.Face(font_path).style_name, "UTF-8")

                # Get image path
                image_path = get_image_path((group_name, type_font,
                                             os.path.splitext(font)[0]),
                                            file_name=f'{group_name}{i}.jpg',
                                            home_dir=home_dir)

                # Conert image to numpy array for saving
                img = np.array(img)
            else:
                # Prepare the font features data
                font = int(font)
                line_type = form[3].astype(int)
                italic = form[4].astype(int)

                # Get text size
                (text_width, text_height), baseline = cv2.getTextSize(letter[0], font | italic,
                                                                      font_size, thickness)

                # Compute text position
                x = int((img.shape[1] - text_width) / 2)
                y = int((img.shape[0] - text_height) / 2) + baseline + int(font_size * 10)

                # Draw text
                cv2.putText(img, letter[0], (x, y), font | italic,
                            font_size, (0, 0, 0), thickness, line_type)

                # Get image path
                image_path = get_image_path((group_name, italic, TXT_ATRIB['cv2']['FONTS_DICT'][str(font)]),
                                            file_name=f'{group_name}{i}.jpg',
                                            home_dir=home_dir)

            # Save image
            cv2.imwrite(image_path, img)
    except Exception as e:
        print(e, e.__traceback__)
        raise e


def create_text_image(letter, formats, home_dir=None, system=True):

    np_empty = np.full((100, 100, 3), 255, dtype=np.uint8)

    if system:
        np_empty = Image.fromarray(np_empty)

    time.sleep(random.randint(1, 10))
    write_letter(letter, np_empty, formats=formats, home_dir=home_dir)


def parallel_run(it=None, func=None, *args):
    try:
        # Create a thread pool with 10 worker threads
        with ThreadPoolExecutor(max_workers=10) as executor:
            # Submit a job for each character and wait for all jobs to complete
            # The args are function name with all function variables needed
            futures = [executor.submit(func, i, *args) for i in it]
            for future in futures:
                future.result()

    except Exception as e:
        print(e)
    finally:
        executor.shutdown()


def prepare_generator(library_name: str, home_dir=None):
    # Get all character for image generating
    all_chars = list(string.ascii_lowercase + string.digits + string.punctuation)
    all_chars += list(map(lambda l: f"{l}_", list(string.ascii_uppercase)))

    # Load fonts attributes
    attr = load_attrib()[library_name]

    # Get library formats
    if library_name == 'cv2':
        formats = np.array(np.meshgrid(attr['FONTS'], attr['FONT_SIZES'],
                                           attr['THICKNESS'], attr['LINE_TYPES'],
                                           attr['ITALIC'])).T.reshape(-1, 5)
        system = False
    else:
        formats = np.array(np.meshgrid(attr['FONTS'], attr['FONT_SIZES'],
                                           attr['THICKNESS'])).T.reshape(-1, 3)
        system = True

    # Start parallel processing
    parallel_run(all_chars, create_text_image, (formats, home_dir, system))


