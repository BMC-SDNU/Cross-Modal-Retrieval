import pickle
import numpy as np
import cv2
import os
import utils
from GAN_args import get_parser

parser = get_parser()
opts = parser.parse_args()


language = {0: "en", 1: "de-en", 2: "ru-en", 3: "de", 4: "ru", 5: "fr"}
language_suffix = '' if language[opts.language]=='en' else '_'+language[opts.language]
dataset = utils.Layer.merge2(['layer2', 'layer1'+language_suffix, 'det_ingrs'], opts.r1m_path)
ind2id = [entry['id'] for entry in dataset]


def get_recipe_info_from_ID(recipe_id, image_path):
    try:
        recipe = OneRecipe.from_json(dataset[ind2id.index(recipe_id)], image_path)
        return recipe
    except:
        return None


def load_VOCAB(data_path):
    global VOCAB_INGR
    if not VOCAB_INGR:
        VOCAB_INGR = pickle.load(open(data_path + "/vocab_ingr.pkl", "rb"))


def get_recipe_info(recipe_id, data_path, image_path):
    recipe = get_recipe_info_from_ID(recipe_id, image_path)
    return recipe


class OneRecipe(object):
    def __init__(self):
        self.img = None
        self.is_image_opencv = False
        self.title = None
        self.ingrs = None
        self.intrs = None
        self.recipe_id = None
        self.recipe_class = None


    @classmethod
    def from_json(cls, data, IMG_PATH):
        new_obj = object.__new__(OneRecipe)
        new_obj.recipe_id = data["id"]
        new_obj.title = data["title"+language_suffix]
        new_obj.ingrs = [line['text'] for line in data["ingredients"+language_suffix]]
        new_obj.intrs = [line['text'] for line in data["instructions"+language_suffix]]

        image_path = [data["images"][0]["id"][i] for i in range(4)]
        image_path = os.path.join(*image_path)
        image_path = os.path.join(IMG_PATH, data['partition'], image_path, data['images'][0]["id"])
        new_obj.img = cv2.imread(image_path)
        new_obj.is_image_opencv = True
        return new_obj


    def get_recipe(self):
        txt = self.title
        txt = txt + "\nIngredients:\n"
        for i in self.ingrs:
            txt += " - " + i + "\n"
        txt += "Instructions:\n"
        for k, i in enumerate(self.intrs):
            txt += f" - {k+1} " + i + "\n"
        return txt


    def get_opencv_image(self):
        if self.is_image_opencv:
            return self.img
        else:
            # un-normalize image:
            img = np.copy(self.img)
            for c in range(3):
                img[:,:,c] = self.img[:,:,c] * (STD[c]*255) + MEAN[c]* 255
            img = img.astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return img


    def show_image(self, window_name=""):
        img = self.get_opencv_image()
        cv2.imshow(window_name if window_name else self.title, img)


    def get_recipe_filename(self):
        ret = str(self.recipe_id) + "-" + self.title.replace(".", "").replace("/", "").replace("\\", "").replace("*", "").replace("\"", "").replace(":", "").replace("?", "").replace("|", "").replace("<", "").replace(">", "").replace(" ", "_")
        return ret


    def save_recipe(self, dir_path):
        # save image
        img = self.get_opencv_image()
        cv2.imwrite(dir_path + "/image.jpg", img)
        # save recipe
        txt = self.get_recipe()
        with open(dir_path + "/full_recipe.txt", "w") as f:
            f.write(txt)


    def __str__(self):
        return self.get_recipe()

    def __repr__(self):
        return self.__str__()