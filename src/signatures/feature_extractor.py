import dspy
from .utils import get_python_type_from_field


def create_extractor_signature(features, preamble = ""):
    class_name = "FeatureExtractionSignature"
    class_attrs = {"__annotations__": {}}
    to_skip = []
    for i, field in enumerate(features):
        field_name = field.name

        try:
            py_type = get_python_type_from_field(field)
        except TypeError as e:
            # print(f"Skipping field '{field_name}' due to error: {e}")
            to_skip.append(i)
            continue

        dspy_field_instance = dspy.OutputField(desc = field.description)

        class_attrs[field_name] = dspy_field_instance
        class_attrs["__annotations__"][field_name] = py_type

    prompt = "Extract the following features from the provided text, following each feature's extraction_query."

    for i, field in enumerate(features):
        if i in to_skip:
            continue

        prompt += f"\n- {field.name} ({field.type}): Extraction Query: {field.extraction_query}"

    class_attrs["__doc__"] = prompt

    class_attrs["text"] = dspy.InputField(desc = "The input text to extract features from.")
    class_attrs["__annotations__"]["text"] = str

    DynamicSignature = type(class_name, (dspy.Signature, ), class_attrs)
    return DynamicSignature


class ExtractorModule(dspy.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, text, features):
        annotator = dspy.Predict(create_extractor_signature(features))
        result = annotator(text = text)
        return result
