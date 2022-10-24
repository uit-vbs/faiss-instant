import logging

import numpy as np
import pandas as pd
import torch
from flask import Blueprint, current_app, jsonify, request
from jsonschema import ValidationError, validate
from werkzeug.exceptions import BadRequest

from faiss_index import FaissIndex
from lavis.models import load_model_and_preprocess


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO
)

blueprint = Blueprint("faiss_index", __name__)

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

model, _, txt_processor = load_model_and_preprocess(
    name="blip_feature_extractor",
    model_type="base",
    is_eval=True,
    device=device,
)

metadata_df = pd.read_csv('/opt/faiss-instant/src/faiss_index/transition_metadata.csv')

def embedding_text_vector(text_query):
    text_input = txt_processor["eval"](text_query)
    sample_text = {"image": "", "text_input": [text_input]}
    features_text = model.extract_features(sample_text, mode="text")
    features_text = features_text.text_embeds_proj[:, 0, :].cpu().numpy().astype(np.float32)
    
    return features_text

@blueprint.record_once
def record(setup_state):
    resources_path = setup_state.app.config.get("RESOURCES_PATH")
    use_gpu = setup_state.app.config.get("USE_GPU")
    blueprint.faiss_index = FaissIndex(resources_path, logger, use_gpu)

@blueprint.route("/embedding_vector", methods=["POST"])
def embedding_vector():
    try:
        json = request.get_json(force=True)
        validate(
            json,
            {
                "type": "object",
                "required": ["text_query"],
                "properties": {
                    "text_query": {"type": "string"},
                },
            },
        )
        if 'text_query' not in json:
            return jsonify([])
        features_text = embedding_text_vector(json["text_query"])
    
        results = {"vector": features_text.tolist()}

        return jsonify(results)
    
    except (BadRequest, ValidationError) as e:
        logger.info("Bad request", e)
        return "Bad request", 400

    except Exception as e:
        logger.info("Server error", e)
        return "Server error", 500

@blueprint.route("/search", methods=["POST"])
def search():
    try:
        json = request.get_json(force=True)
        validate(
            json,
            {
                "type": "object",
                "required": ["k", "vector"],
                "properties": {
                    "k": {"type": "integer", "minimum": 1},
                    "vector": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "number"}},
                    },
                },
            },
        )
        if 'vector' not in json:
            return jsonify([])

        results = blueprint.faiss_index.search(json['vector'], json['k'])

        results_lst = []
        for index, score in results[0].items():
            results_dict = {}
            results_dict.update(metadata_df.iloc[int(index)].to_dict())
            results_dict["score"] = score
            results_lst.append(results_dict)

        return jsonify(results_lst)

    except (BadRequest, ValidationError) as e:
        logger.info("Bad request", e)
        return "Bad request", 400

    except Exception as e:
        logger.info("Server error", e)
        return "Server error", 500


@blueprint.route("/explain", methods=["POST"])
def explain():
    try:
        json = request.get_json(force=True)
        validate(
            json,
            {
                "type": "object",
                "required": ["vector", "id"],
                "properties": {
                    "id": {"type": "string", "minimum": 1},
                    "vector": {
                        "type": "array",
                        "items": {
                            "type": "number",
                        },
                    },
                },
            },
        )
        result = blueprint.faiss_index.explain(json["vector"], json["id"])
        return jsonify(result)

    except (BadRequest, ValidationError) as e:
        logger.info("Bad request", e)
        return "Bad request", 400

    except Exception as e:
        logger.info("Server error", e)
        return "Server error", 500


@blueprint.route("/reload", methods=["POST"])
def reload():
    data = request.get_json()
    index_name = None
    use_gpu = False
    if data is not None:
        assert "index_name" in data, "Please specify 'index_name' in the URL arguments."
        index_name = data["index_name"]
        use_gpu = data.setdefault("use_gpu", False)
    blueprint.faiss_index.load(index_name, use_gpu)
    return f"Faiss index ({index_name}) reloaded\n", 200


@blueprint.route("/index_list", methods=["GET"])
def index_list():
    index_list = blueprint.faiss_index.parse_index_list()
    index_loaded = blueprint.faiss_index.index_loaded
    device = blueprint.faiss_index.device
    results = {"index loaded": index_loaded, "device": device, "index list": index_list}
    return jsonify(results)


@blueprint.route("/reconstruct", methods=["GET"])
def reconstruct():
    _id = request.args.get("id")
    result = {"vector": blueprint.faiss_index.reconstruct(_id).tolist()}
    return jsonify(result)
