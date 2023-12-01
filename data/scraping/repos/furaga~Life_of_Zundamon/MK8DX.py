import time
import openai
import os
import re
from pathlib import Path
from glob import glob
import cv2
import numpy as np
import clip
import torch

if __name__ == "__main__":
    import digit_ocr
else:
    from . import digit_ocr


omote_item_dict_ = {}
ura_item_dict_ = {}
place_dict_ = {}
center_dict_ = {}


def cv2pil(image):
    from PIL import Image

    """OpenCV型 -> PIL型"""
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image


def imread_safe(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
    try:
        n = np.fromfile(filename, dtype)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        print(e)
        return None


ITEM_IMAGE_SIZE = 153


def get_clip_features(img):
    with torch.no_grad():
        p = cv2pil(img)
        image = clip_preprocess(p).unsqueeze(0).to(device)
        features = model.encode_image(image)
        return features.cpu().numpy()


def load_item_images(item_dir: Path):
    for type, item_dict in [("表", omote_item_dict_), ("裏", ura_item_dict_)]:
        all_img_paths = list(item_dir.glob(f"{type}/*.jpg"))
        for img_path in all_img_paths:
            img = imread_safe(str(img_path))
            #            img = cv2.resize(img, (ITEM_IMAGE_SIZE, ITEM_IMAGE_SIZE))

            mask_path = item_dir / "mask" / (img_path.stem.split("_")[0] + ".png")
            if mask_path.exists():
                mask = imread_safe(str(mask_path), cv2.IMREAD_GRAYSCALE).astype(
                    np.uint8
                )
            else:
                mask = np.zeros(img.shape[:2], np.uint8)
                mask.fill(255)

            if type == "裏":
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

            # マスクする
            rgb = cv2.bitwise_and(img, img, mask=mask)
            # cv2.imshow("rgb", rgb)
            # cv2.waitKey(0)

            feat = get_clip_features(rgb)
            feat /= np.linalg.norm(feat)
            item_dict[img_path.stem] = feat

    print("Loaded", len(omote_item_dict_), "omote item images.")
    print("Loaded", len(ura_item_dict_), "ura item images.")


def load_place_images(place_dir: Path):
    all_img_paths = list(place_dir.glob("*.png"))
    for img_path in all_img_paths:
        img = imread_safe(str(img_path))
        feat = get_clip_features(img)
        feat /= np.linalg.norm(feat)
        place_dict_[img_path.stem] = feat

    print("Loaded", len(place_dict_), "place images.")


def load_finish_images(finish_dir: Path):
    all_img_paths = list(finish_dir.glob("*.png"))
    for img_path in all_img_paths:
        img = imread_safe(str(img_path))
        center_dict_[img_path.stem] = img
    print("Loaded", len(center_dict_), "finish tempalte images.")


device, model, clip_preprocess = None, None, None


def init(root_dir: Path):
    global device, model, clip_preprocess
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, clip_preprocess = clip.load("ViT-B/32", device)
    load_item_images(root_dir / "items")
    load_place_images(root_dir / "place_rembg")
    load_finish_images(root_dir / "finish")


def match(img_feat, ref_feat):
    similarity = img_feat @ ref_feat.T
    return similarity[0][0]


def detect_items(img):
    h, w = img.shape[:2]
    x1 = int(113 / 1280 * w)
    x2 = int(215 / 1280 * w)
    y1 = int(65 / 720 * h)
    y2 = int(167 / 720 * h)
    omote = img[y1:y2, x1:x2]
    omote_feat = get_clip_features(omote)
    omote_feat /= np.linalg.norm(omote_feat)

    x1 = int(48 / 1280 * w)
    x2 = int(112 / 1280 * w)
    y1 = int(38 / 720 * h)
    y2 = int(102 / 720 * h)
    ura = img[y1:y2, x1:x2]
    ura_feat = get_clip_features(ura)
    ura_feat /= np.linalg.norm(ura_feat)

    def adhoc_correction(score, item_name):
        if item_name.startswith("トリプル"):
            # XXXよりトリプルXXXのスコアがなぜか大きく出がちなので、トリプル系にデバフを掛ける
            return score - 0.02
        return score

    omote_ls = []
    ura_ls = []

    for name, ref_feat in omote_item_dict_.items():
        name = name.split("_")[0].replace("2", "3")  # 例: バナ2→バナ3
        omote_score = match(omote_feat, ref_feat)
        omote_score = adhoc_correction(omote_score, name)
        omote_ls.append([omote_score, name])

    for name, ref_feat in ura_item_dict_.items():
        name = name.split("_")[0].replace("2", "3")  # 例: バナ2→バナ3
        ura_score = match(ura_feat, ref_feat)
        ura_score = adhoc_correction(ura_score, name)
        ura_ls.append([ura_score, name])

    omote_ls = sorted(omote_ls)
    ura_ls = sorted(ura_ls)
    # print("[omote]", omote_ls[-2:])
    # print("[ura  ]",ura_ls[-2:])

    res_omote = omote_ls[-1]
    res_ura = ura_ls[-1]

    # TODO: 緑と緑を間違えやすいので、緑甲羅と判別されたら色を確認する

    return res_omote, res_ura


# 現在の順位
def detect_place(img):
    h, w = img.shape[:2]
    x1 = int(1600 / 1920 * w)
    x2 = int(1820 / 1920 * w)
    y1 = int(840 / 1080 * h)
    y2 = int(1030 / 1080 * h)
    place_img = img[y1:y2, x1:x2]
    place_img_feat = get_clip_features(place_img)
    place_img_feat /= np.linalg.norm(place_img_feat)

    ls = []
    for name, ref_feat in place_dict_.items():
        name = name.split("_")[0]
        ls.append([match(place_img_feat, ref_feat), name])

    ls = sorted(ls)
    # print(ls[-2:])

    return ls[-1]


# "FINISH"
def detect_finish(img):
    h, w = img.shape[:2]
    x1 = int(444 / 1920 * w)
    x2 = int(1480 / 1920 * w)
    y1 = int(350 / 1080 * h)
    y2 = int(590 / 1080 * h)
    center_img = img[y1:y2, x1:x2]

    if w != 1920 or h != 1080:
        # 画像サイズ違ったらリサイズ
        fx = 1920 / w
        fy = 1080 / h
        center_img = cv2.resize(center_img, None, fx=fx, fy=fy)

    ls = []
    for name, tmpl in center_dict_.items():
        name = name.split("_")[0]
        result = cv2.matchTemplate(center_img, tmpl, cv2.TM_CCORR_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        ls.append([max_val, name])

    ls = sorted(ls)
    # print(ls[-2:])

    return ls[-1]


def detect_number(img, verbose):
    coin_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, value = digit_ocr.detect_digit(coin_img, verbose)
    # print(ret, value)
    return ret, value


# コイン枚数
def detect_coin(img):
    h, w = img.shape[:2]
    x1 = int(133 / 1920 * w)
    x2 = int(214 / 1920 * w)
    y1 = int(972 / 1080 * h)
    y2 = int(1032 / 1080 * h)
    return detect_number(img[y1:y2, x1:x2], False)


# 何周目か
def detect_lap(img):
    h, w = img.shape[:2]
    x1 = int(300 / 1920 * w)
    x2 = int(345 / 1920 * w)
    y1 = int(972 / 1080 * h)
    y2 = int(1032 / 1080 * h)
    return detect_number(img[y1:y2, x1:x2], False)


def imwrite_safe(filename, img, params=None):
    try:
        ext = os.path.splitext(filename)[1]
        result, n = cv2.imencode(ext, img, params)

        if result:
            with open(filename, mode="w+b") as f:
                n.tofile(f)
            return True
        else:
            return False
    except Exception as e:
        print(e)
        return False


def crop_and_save_items():
    all_img_paths = Path("../data/mk8dx_images/items_capture").glob("表/*.png")
    for img_path in all_img_paths:
        img = imread_safe(str(img_path))

        h, w = img.shape[:2]

        margin = 10
        x1 = int((113 - margin) / 1280 * w)
        x2 = int((215 + margin) / 1280 * w)
        y1 = int((65 - margin) / 720 * h)
        y2 = int((167 + margin) / 720 * h)
        omote = img[y1:y2, x1:x2]
        imwrite_safe("表/" + img_path.stem + ".jpg", omote)

        # x1 = int(48 / 1280 * w)
        # x2 = int(112 / 1280 * w)
        # y1 = int(38 / 720 * h)
        # y2 = int(102 / 720 * h)
        # ura = img[y1:y2, x1:x2]
        # imwrite_safe("裏/" + img_path.stem + ".jpg", ura)


# crop_and_save_items()

if __name__ == "__main__":

    def main() -> None:
        init(Path("../data/mk8dx_images"))

        img_list = list(Path("../record").glob("*.png"))
        for i, img_path in enumerate(img_list):
            # if i < 400:
            #    continue
            img = cv2.imread(str(img_path))

            # since = time.time()
            # ret = detect_finish(img)
            # print(ret)
            # print(f"({i})[detect_finish] Elapsed {time.time() - since:.2f} sec")

            since = time.time()
            ret = detect_items(img)
            print(ret)
            print(f"({i})[detect_items] Elapsed {time.time() - since:.2f} sec")

            # since = time.time()
            # ret = detect_place(img)
            # print(ret)
            # print(f"({i})[detect_place] Elapsed {time.time() - since:.2f} sec")

            # since = time.time()
            # ret = detect_coin(img)
            # since = time.time()
            # ret = detect_lap(img)
            # print(f"({i})[detect coin/lap] Elapsed {time.time() - since:.2f} sec")

            # 大きくて画面に入らないので小さく
            img_resize = cv2.resize(img, None, fx=0.5, fy=0.5)
            cv2.imshow("screenshot", img_resize)
            if ord("q") == cv2.waitKey(0 if ret[0][0] > 0.81 else 1):
                break

    main()
