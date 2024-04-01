import yaml
from utils.draw import draw_box


with open('assets/coco_chinese.yaml', 'r', encoding='utf-8') as f:
    names = yaml.load(f, Loader=yaml.FullLoader)


annotated_frame = draw_box(
    frame,
    det,
    names,
    line_width=2,
    font_size=20,
    pil=True,
    example=names[0]
    )