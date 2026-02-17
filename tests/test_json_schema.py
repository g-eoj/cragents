from inline_snapshot import snapshot
from pydantic import BaseModel
from pydantic_ai import BinaryImage, DeferredToolRequests, ToolOutput, _output

from cragents._utils import build_json_schema


def test_json_schema_default_string():
    output_schema = _output.OutputSchema.build(str)
    assert build_json_schema(output_schema) == {"type": "string"}


def test_json_schema_single_tool_output_bool():
    output_schema = _output.OutputSchema.build(ToolOutput(bool))
    result = build_json_schema(output_schema)
    assert result == {"properties": {"response": {"type": "boolean"}}, "required": ["response"], "type": "object"}


def test_json_schema_deduplicated_tool_outputs():
    output_schema = _output.OutputSchema.build([ToolOutput(bool, name="one"), ToolOutput(bool, name="two")])
    result = build_json_schema(output_schema)
    # Identical schemas are deduplicated — result is a single schema, not anyOf
    assert result == {"properties": {"response": {"type": "boolean"}}, "required": ["response"], "type": "object"}


def test_json_schema_multiple_distinct_tool_outputs():
    output_schema = _output.OutputSchema.build([ToolOutput(bool), ToolOutput(int)])
    result = build_json_schema(output_schema)
    assert result == snapshot(
        {
            "anyOf": [
                {"properties": {"response": {"type": "boolean"}}, "required": ["response"], "type": "object"},
                {"properties": {"response": {"type": "integer"}}, "required": ["response"], "type": "object"},
            ]
        }
    )


def test_json_schema_mixed_tool_and_text_output():
    output_schema = _output.OutputSchema.build([ToolOutput(bool), str])
    result = build_json_schema(output_schema)
    assert result == snapshot(
        {
            "anyOf": [
                {"type": "string"},
                {"properties": {"response": {"type": "boolean"}}, "required": ["response"], "type": "object"},
            ]
        }
    )


def test_json_schema_object_output_processor():
    # A plain BaseModel output goes through the ObjectOutputProcessor branch
    class Item(BaseModel):
        name: str
        value: int

    output_schema = _output.OutputSchema.build(Item)
    result = build_json_schema(output_schema)
    assert result == {
        "properties": {"name": {"type": "string"}, "value": {"type": "integer"}},
        "required": ["name", "value"],
        "title": "Item",
        "type": "object",
    }


def test_json_schema_allows_deferred_tools():
    output_schema = _output.OutputSchema.build([ToolOutput(bool), DeferredToolRequests])
    result = build_json_schema(output_schema)
    # Result should be anyOf including the DeferredToolRequests schema
    assert "anyOf" in result
    schemas = result["anyOf"]
    # At least one schema should have a "calls" property (DeferredToolRequests)
    assert any("calls" in s.get("properties", {}) for s in schemas)


def test_json_schema_allows_image():
    output_schema = _output.OutputSchema.build([ToolOutput(bool), BinaryImage])
    result = build_json_schema(output_schema)
    assert "anyOf" in result
    schemas = result["anyOf"]
    # The image schema only has 'data' and 'media_type' keys
    assert any(set(s.keys()) == {"data", "media_type"} for s in schemas)


def test_json_schema_defs_from_shared_model():
    # Two ToolOutputs that both reference a shared sub-model trigger $defs in merged schema
    class Address(BaseModel):
        street: str

    class Person(BaseModel):
        address: Address

    class Company(BaseModel):
        headquarters: Address

    output_schema = _output.OutputSchema.build([ToolOutput(Person), ToolOutput(Company)])
    result = build_json_schema(output_schema)
    assert "anyOf" in result
    assert "$defs" in result
    assert "Address" in result["$defs"]
