import os
import sys

if "APP_PATH" in os.environ:
    os.chdir(os.environ["APP_PATH"])
    # fix sys.path for import
    sys.path.append(os.getcwd())

print(os.getenv("XDG_CACHE_HOME"))

import gradio as gr

from tabled.assignment import assign_rows_columns
from tabled.fileinput import load_pdfs_images
from tabled.formats.markdown import markdown_format
from tabled.inference.detection import detect_tables
from tabled.inference.recognition import get_cells, recognize_tables

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["IN_STREAMLIT"] = "true"
import pypdfium2

import io
import tempfile
from PIL import Image

from tabled.inference.models import load_detection_models, load_recognition_models, load_layout_models

def load_models():
    return load_detection_models(), load_recognition_models(), load_layout_models()

def run_table_rec(image, highres_image, text_line, models, skip_detection=False, detect_boxes=False):
    if not skip_detection:
        table_imgs, table_bboxes, _ = detect_tables([image], [highres_image], models[2])
    else:
        table_imgs = [highres_image]
        table_bboxes = [[0, 0, highres_image.size[0], highres_image.size[1]]]

    table_text_lines = [text_line] * len(table_imgs)
    highres_image_sizes = [highres_image.size] * len(table_imgs)
    cells, needs_ocr = get_cells(table_imgs, table_bboxes, highres_image_sizes, table_text_lines, models[0], detect_boxes=detect_boxes)

    table_rec = recognize_tables(table_imgs, cells, needs_ocr, models[1])
    cells = [assign_rows_columns(tr, im_size) for tr, im_size in zip(table_rec, highres_image_sizes)]

    out_data = []
    for idx, (cell, pred, table_img) in enumerate(zip(cells, table_rec, table_imgs)):
        md = markdown_format(cell)
        out_data.append((md, table_img))
    return out_data

def open_pdf(pdf_file):
    return pypdfium2.PdfDocument(pdf_file)

def count_pdf(pdf_file):
    doc = open_pdf(pdf_file)
    return len(doc)

def get_page_image(pdf_file, page_num, dpi=96):
    doc = open_pdf(pdf_file)
    renderer = doc.render(
        pypdfium2.PdfBitmap.to_pil,
        page_indices=[page_num - 1],
        scale=dpi / 72,
    )
    png = list(renderer)[0]
    png_image = png.convert("RGB")
    return png_image

def get_uploaded_image(in_file):
    return Image.open(in_file).convert("RGB")


models = load_models()

with gr.Blocks(title="Tabled") as demo:
    gr.Markdown("""
    # Tabled Demo

    This app will let you try tabled, a table detection and recognition model.  It will detect and recognize the tables.

    Find the project [here](https://github.com/VikParuchuri/tabled).
    """)

    with gr.Row():
        with gr.Column():
            in_file = gr.File(label="PDF file or image:", file_types=[".pdf", ".png", ".jpg", ".jpeg", ".gif", ".webp"])
            in_num = gr.Slider(label="Page number", minimum=1, maximum=100, value=1, step=1)
            in_img = gr.Image(label="Page of Image", type="pil", sources=None)

            skip_detection_ckb = gr.Checkbox(label="Skip table detection", info="Use this if tables are already cropped (the whole PDF page or image is a table)")
            detect_boxes_ckb = gr.Checkbox(label="Detect cell boxes", info="Detect table cell boxes vs extract from PDF.  Will also run OCR.")

            tabled_btn = gr.Button("Run Tabled")
        with gr.Column():
            result_img = gr.Gallery(label="Result images", show_label=False, columns=[1], rows=[10], object_fit="contain", height="auto")
            result_md = gr.Markdown(label="Result markdown")

        def show_image(file, num=1):
            if file.endswith('.pdf'):
                count = count_pdf(file)
                img = get_page_image(file, num)
                return [
                    gr.update(visible=True, maximum=count),
                    gr.update(value=img)]
            else:
                img = get_uploaded_image(file)
                return [
                    gr.update(visible=False),
                    gr.update(value=img)]

        in_file.upload(
            fn=show_image,
            inputs=[in_file],
            outputs=[in_num, in_img],
        )
        in_num.change(
            fn=show_image,
            inputs=[in_file, in_num],
            outputs=[in_num, in_img],
        )

        # Run OCR
        def text_rec_img(in_file, page_number, skip_detection, detect_boxes):
            images, highres_images, names, text_lines = load_pdfs_images(in_file, max_pages=1, start_page=page_number - 1)

            out_data = run_table_rec(images[0], highres_images[0], text_lines[0], models, skip_detection=skip_detection, detect_boxes=detect_boxes)

            if len(out_data) == 0:
                gr.Warning(f"No tables found on page {page_number}", duration=5)

            table_md = ""
            table_imgs = []
            for idx, (md, table_img) in enumerate(out_data):
                table_md += f"\n## Table {idx}\n"
                table_md += md
                table_imgs.append(table_img)
            return gr.update(rows=[len(table_imgs)], value=table_imgs), table_md
        tabled_btn.click(
            fn=text_rec_img,
            inputs=[in_file, in_num, skip_detection_ckb, detect_boxes_ckb],
            outputs=[result_img, result_md]
        )

demo.launch()
