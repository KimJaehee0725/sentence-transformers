from __future__ import annotations

from dataclasses import field
from typing import Any

import torch
from typing_extensions import deprecated

from sentence_transformers.base.model_card import BaseModelCardCallback, BaseModelCardData
from sentence_transformers.base.modules import Router
from sentence_transformers.sentence_transformer.modules import StaticEmbedding


class SentenceTransformerModelCardCallback(BaseModelCardCallback):
    def on_log(self, args, state, control, model, logs, **kwargs):
        super().on_log(args, state, control, model, logs, **kwargs)

        # Set the ir_model flag so we can generate the model card with the encode_query/encode_document methods
        keys = {"loss"} & set(logs)
        if model.model_card_data.ir_model is None:
            for key in keys:
                if "ndcg" in key:
                    model.model_card_data.ir_model = True


class SentenceTransformerModelCardData(BaseModelCardData):
    ir_model: bool | None = field(default=None, init=False, repr=False)

    def try_to_set_base_model(self):
        super().try_to_set_base_model()
        if isinstance(self.model[0], StaticEmbedding) and self.base_model is None:
            if self.model[0].base_model:
                self.set_base_model(self.model[0].base_model)

    def register_model(self, model):
        super().register_model(model)

        if self.ir_model is not None:
            return

        if Router in [module.__class__ for module in model.children()]:
            self.ir_model = True
            return

        for ir_prompt_name in ["query", "document", "passage", "corpus"]:
            if ir_prompt_name in model.prompts and len(model.prompts[ir_prompt_name]) > 0:
                self.ir_model = True
                return

    def extract_dataset_metadata(self, dataset, dataset_metadata, loss, dataset_type):
        validated_datasets = super().extract_dataset_metadata(dataset, dataset_metadata, loss, dataset_type)
        if dataset_type == "train":
            if self.ir_model is None:
                if isinstance(dataset, dict):
                    column_names = set(
                        column for sub_dataset in dataset.values() for column in sub_dataset.column_names
                    )
                else:
                    column_names = set(dataset.column_names)
                if {"query", "question"} & column_names:
                    self.ir_model = True
        return validated_datasets

    def run_usage_snippet(self) -> dict[str, Any]:
        if self.predict_example is None:
            if self.ir_model:
                self.predict_example = [
                    "Which planet is known as the Red Planet?",
                    "Venus is often called Earth's twin because of its similar size and proximity.",
                    "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
                    "Saturn, famous for its rings, is sometimes mistaken for the Red Planet.",
                ]
            else:
                self.predict_example = [
                    "The weather is lovely today.",
                    "It's so sunny outside!",
                    "He drove to the stadium.",
                ]

        if not self.generate_widget_examples:
            return

        if self.ir_model:
            query_embeddings = self.model.encode_query(
                self.predict_example[0], convert_to_tensor=True, show_progress_bar=False
            )
            document_embeddings = self.model.encode_document(
                self.predict_example[1:], convert_to_tensor=True, show_progress_bar=False
            )
            similarity = self.model.similarity(query_embeddings, document_embeddings)
        else:
            self.predict_example = self.predict_example[:3]  # Limit to 3 examples for standard similarity
            embeddings = self.model.encode(self.predict_example, convert_to_tensor=True, show_progress_bar=False)
            similarity = self.model.similarity(embeddings, embeddings)

        with torch._tensor_str.printoptions(precision=4, sci_mode=False):
            self.similarities = "\n".join(f"# {line}" for line in str(similarity.cpu()).splitlines())

    def get_model_specific_metadata(self) -> dict[str, Any]:
        metadata = super().get_model_specific_metadata()
        similarity_fn_name = "Cosine Similarity"
        if self.model.similarity_fn_name:
            similarity_fn_name = {
                "cosine": "Cosine Similarity",
                "dot": "Dot Product",
                "euclidean": "Euclidean Distance",
                "manhattan": "Manhattan Distance",
            }.get(self.model.similarity_fn_name, self.model.similarity_fn_name.replace("_", " ").title())
        metadata.update(
            {
                "output_dimensionality": self.model.get_embedding_dimension(),
                "similarity_fn_name": similarity_fn_name,
            }
        )
        return metadata

    def generate_usage_snippet(self) -> str:
        if not self.ir_model:
            return super().generate_usage_snippet()

        display = self.predict_example_display or self.predict_example
        examples = display or [
            "Which planet is known as the Red Planet?",
            "Venus is often called Earth's twin because of its similar size and proximity.",
            "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
            "Saturn, famous for its rings, is sometimes mistaken for the Red Planet.",
        ]

        model_id = self.model_id or "sentence_transformers_model_id"
        output_dim = self._get_snippet_output_dimensionality()
        num_docs = len(examples) - 1

        lines = [
            "from sentence_transformers import SentenceTransformer",
            "",
            "# Download from the 🤗 Hub",
            f'model = SentenceTransformer("{model_id}")',
            "# Run inference",
            "queries = [",
            f"    {self._format_snippet_value(examples[0])},",
            "]",
            "documents = [",
        ]
        for item in examples[1:]:
            lines.append(f"    {self._format_snippet_value(item)},")
        lines.extend(
            [
                "]",
                "query_embeddings = model.encode_query(queries)",
                "document_embeddings = model.encode_document(documents)",
                "print(query_embeddings.shape, document_embeddings.shape)",
                f"# [1, {output_dim}] [{num_docs}, {output_dim}]",
                "",
                "# Get the similarity scores for the embeddings",
                "similarities = model.similarity(query_embeddings, document_embeddings)",
            ]
        )
        if self.similarities:
            lines.append("print(similarities)")
            lines.append(self.similarities)
        else:
            lines.extend(
                [
                    "print(similarities.shape)",
                    f"# [1, {num_docs}]",
                ]
            )

        return "```python\n" + "\n".join(lines) + "\n```"


@deprecated(
    "The `ModelCardCallback` has been renamed to `SentenceTransformerModelCardCallback` and the former is now deprecated. Please use `SentenceTransformerModelCardCallback` instead."
)
class ModelCardCallback(SentenceTransformerModelCardCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
