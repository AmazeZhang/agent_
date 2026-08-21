import json
import os
from io import BytesIO

import pyarrow as pa
import pyarrow.parquet as pq

from PIL import Image
from tqdm import tqdm


def _build_bbox_map(values, bboxes):
    if isinstance(bboxes, dict):
        return bboxes or {"x1": "none"}
    if not isinstance(values, list) or not isinstance(bboxes, list):
        return {"x1": "none"}
    out = {}
    for v, b in zip(values, bboxes):
        out[str(v)] = b
    return out or {"x1": "none"}


def _normalize_type(source):
    if not source:
        return None
    if source.startswith("chartqa_"):
        return source[len("chartqa_") :]
    return source


def _to_figure_path(image_path):
    if not image_path:
        return None
    if image_path.startswith("data/ChartQA/"):
        return image_path
    marker = "ChartQA/ChartQA Dataset/"
    if marker in image_path:
        suffix = image_path.split(marker, 1)[1]
        return "data/ChartQA/" + suffix
    return image_path


def _resolve_image_path(image_path, repo_root):
    candidates = []
    script_dir = os.path.dirname(__file__)
    if image_path:
        candidates.append(os.path.join(script_dir, image_path))
        candidates.append(os.path.join(repo_root, image_path))
        candidates.append(image_path)
    figure_path = _to_figure_path(image_path)
    if figure_path:
        candidates.append(os.path.join(repo_root, figure_path))
        candidates.append(os.path.join(script_dir, figure_path))
        candidates.append(figure_path)
    for p in candidates:
        if p and os.path.exists(p):
            return p
    raise FileNotFoundError(f"image not found, tried: {candidates}")


def _process_split(split, repo_root):
    with open(f"chartqa_vcot/{split}.jsonl", "r") as file:
        data = [json.loads(line) for line in file]

    figure_id = []
    figure_path = []
    images = []
    query = []
    prompt = []
    answer = []
    metadata = []

    for record in tqdm(data):
        figure_id.append(record["id"])

        full_prompt = f"""<image> # USER REQUEST #: {record.get("question")}
# USER Bounding Box Info: x_values_bbox, storing x values and coordinates. y_values_bbox, storing x values and coordinates. The x values in the image are: {record.get("x_values", [])}. The y values in the image are: {record.get("y_values", [])}.
# USER IMAGE stored in image_1, as PIL image."""
        query.append(record.get("question"))
        prompt.append(full_prompt)

        ans = record.get("answer")
        if isinstance(ans, list):
            ans = "|||".join([str(a) for a in ans])
        answer.append(ans)

        image_path = record.get("image")
        figure_path.append(_to_figure_path(image_path))

        x_bbox_map = _build_bbox_map(record.get("x_values", []), record.get("x_values_bbox", []))
        y_bbox_map = _build_bbox_map(record.get("y_values", []), record.get("y_values_bbox", []))
        meta = {
            "type": _normalize_type(record.get("source")),
            "figure_bbox": record.get("figure_bbox"),
            "x_values_bbox": x_bbox_map,
            "y_values_bbox": y_bbox_map,
        }
        metadata.append(json.dumps(meta, ensure_ascii=False))

        img_path = _resolve_image_path(image_path, repo_root)
        with Image.open(img_path) as img:
            buffer = BytesIO()
            img.save(buffer, format=img.format)
            image_bytes = buffer.getvalue()
            images.append([image_bytes])

    merged_dict = {
        "metadata": metadata,
        "figure_id": figure_id,
        "figure_path": figure_path,
        "query": query,
        "prompt": prompt,
        "answer": answer,
        "images": images,
    }

    output_dir = os.path.join(repo_root, "datasets")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{split}_full.parquet")

    reference_path = os.path.join(repo_root, "datasets_back", f"{split}_full.parquet")
    reference_metadata = None
    if os.path.exists(reference_path):
        reference_metadata = pq.read_schema(reference_path).metadata

    table = pa.table(
        {
            "metadata": pa.array(metadata, type=pa.string()),
            "figure_id": pa.array(figure_id, type=pa.string()),
            "figure_path": pa.array(figure_path, type=pa.string()),
            "query": pa.array(query, type=pa.string()),
            "prompt": pa.array(prompt, type=pa.string()),
            "answer": pa.array(answer, type=pa.string()),
            "images": pa.array(images, type=pa.list_(pa.binary())),
        }
    ).replace_schema_metadata(reference_metadata or {})
    pq.write_table(table, output_path)


repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for split in ["train", "val"]:
    _process_split(split, repo_root)

'''
split_ds = ds.train_test_split(test_size=0.1, seed=492)

train_dataset = split_ds['train']
test_dataset = split_ds['test']

# Save as parquet files
train_dataset.to_parquet("train.parquet")
test_dataset.to_parquet("test.parquet")'''
