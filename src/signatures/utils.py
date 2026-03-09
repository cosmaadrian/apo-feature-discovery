from typing import List, Literal, Optional
from pydantic import Field, BaseModel


class Feature(BaseModel):
    name: str = Field(description = "The feature name (snake_case, descriptive)")
    type: Literal['int', 'float', 'bool', 'Literal'] = Field(description = "The pydantic type for this feature.")
    description: str = Field(description = "A concise description of what this feature represents.")
    extraction_query: str = Field(description = "The LLM extraction query to extract this feature from text samples.")
    literal_values: Optional[List[str]] = Field(default = None, description = "For Literal types, the allowed values.")

    def to_dspy_field_code(self) -> str:
        if self.type == 'Literal' and self.literal_values:
            type_annotation = (f"Literal[{', '.join(repr(v) for v in self.literal_values)}]")
        else:
            type_annotation = self.type.value

        return f'{self.name}: {type_annotation} = dspy.{field_type}(desc="{self.description}")'


def get_python_type_from_field(field):
    """Converts a Feature into a Python type for annotations."""
    type_str = field.type
    type_map = {
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
    }

    if field.type == 'Literal' and field.literal_values:
        return Literal[tuple(field.literal_values)]

    if type_str in type_map:
        return type_map[type_str]

    raise TypeError(f"Unsupported field type for dynamic class creation: '{type_str}'; {field}")


####################################################################################################################################
####################################################################################################################################
####################################################################################################################################