from inline_snapshot import snapshot

from cragents import Anchor, Constrain, Free, Think, UseTools
from cragents._utils import build_grammar, make_guided_extra_body

# ── build_grammar ──────────────────────────────────────────────────────────────


def test_grammar_single_anchor():
    grammar = build_grammar([Anchor("hello ")])
    assert grammar == snapshot("""\
start: "hello "
FREE: /[\\S\\s]*/
NL: /\\n/""")


def test_grammar_multiple_anchors():
    grammar = build_grammar([Anchor("foo "), Anchor("bar ")])
    assert grammar == snapshot("""\
start: "foo " "bar "
FREE: /[\\S\\s]*/
NL: /\\n/\
""")


def test_grammar_single_free():
    grammar = build_grammar([Free()])
    assert grammar == snapshot("""\
start: FREE
FREE: /[\\S\\s]*/
NL: /\\n/\
""")


def test_grammar_anchor_then_free():
    grammar = build_grammar([Anchor("Result: "), Free()])
    assert grammar == snapshot("""\
start: "Result: " FREE
FREE: /[\\S\\s]*/
NL: /\\n/\
""")


def test_grammar_single_constrain():
    grammar = build_grammar([Constrain(max_newlines=2, max_char_captures=3)])
    assert grammar == snapshot("""\
start: block_1
block_1: p_1{1,2}
p_1: s_1{1,3} NL NL
s_1[lazy]: /[^\\.\\n]+/ ( "." )
FREE: /[\\S\\s]*/
NL: /\\n/\
""")


def test_grammar_constrain_custom_chars():
    grammar = build_grammar([Constrain(max_newlines=1, max_char_captures=2, chars_to_capture="!?")])
    assert grammar == snapshot("""\
start: block_1
block_1: p_1{1,1}
p_1: s_1{1,2} NL NL
s_1[lazy]: /[^!\\?\\n]+/ ( "!" | "?" )
FREE: /[\\S\\s]*/
NL: /\\n/\
""")


def test_grammar_constrain_regex_special_chars():
    # "." is the default and must be escaped to "\\." in the character class
    grammar = build_grammar([Constrain(max_newlines=1, max_char_captures=1, chars_to_capture=".")])
    assert grammar == snapshot("""\
start: block_1
block_1: p_1{1,1}
p_1: s_1{1,1} NL NL
s_1[lazy]: /[^\\.\\n]+/ ( "." )
FREE: /[\\S\\s]*/
NL: /\\n/\
""")


def test_grammar_multiple_constrains_uid_increments():
    grammar = build_grammar([Constrain(1, 1), Constrain(2, 3)])
    assert grammar == snapshot("""\
start: block_1 block_2
block_1: p_1{1,1}
p_1: s_1{1,1} NL NL
s_1[lazy]: /[^\\.\\n]+/ ( "." )
block_2: p_2{1,2}
p_2: s_2{1,3} NL NL
s_2[lazy]: /[^\\.\\n]+/ ( "." )
FREE: /[\\S\\s]*/
NL: /\\n/\
""")


def test_grammar_think_empty_sequence():
    grammar = build_grammar([Think([])])
    assert grammar == snapshot("""\
start: <think> NL </think>
FREE: /[\\S\\s]*/
NL: /\\n/\
""")


def test_grammar_think_with_anchor():
    grammar = build_grammar([Think([Anchor("step one ")])])
    assert grammar == snapshot("""\
start: <think> NL "step one " </think>
FREE: /[\\S\\s]*/
NL: /\\n/\
""")


def test_grammar_think_with_constrain():
    grammar = build_grammar([Think([Constrain(1, 2)])])
    assert grammar == snapshot("""\
start: <think> NL block_1 </think>
block_1: p_1{1,1}
p_1: s_1{1,2} NL NL
s_1[lazy]: /[^\\.\\n]+/ ( "." )
FREE: /[\\S\\s]*/
NL: /\\n/\
""")


def test_grammar_think_with_free():
    grammar = build_grammar([Think([Free()])])
    assert grammar == snapshot("""\
start: <think> NL FREE </think>
FREE: /[\\S\\s]*/
NL: /\\n/\
""")


def test_grammar_think_uid_shared_with_outer():
    # Outer Constrain gets uid=1, inner Think's Constrain gets uid=2
    grammar = build_grammar([Constrain(1, 1), Think([Constrain(2, 3)])])
    assert grammar == snapshot("""\
start: block_1 <think> NL block_2 </think>
block_1: p_1{1,1}
p_1: s_1{1,1} NL NL
s_1[lazy]: /[^\\.\\n]+/ ( "." )
block_2: p_2{1,2}
p_2: s_2{1,3} NL NL
s_2[lazy]: /[^\\.\\n]+/ ( "." )
FREE: /[\\S\\s]*/
NL: /\\n/\
""")


def test_grammar_think_custom_tokens():
    grammar = build_grammar([Think([], start_token="<|think|>", stop_token="<|/think|>")])
    assert grammar == snapshot("""\
start: <|think|> NL <|/think|>
FREE: /[\\S\\s]*/
NL: /\\n/\
""")


def test_grammar_use_tools_explicit_schema():
    schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
    grammar = build_grammar([UseTools(json_schema=schema)])
    assert grammar == snapshot("""\
start: <tool_call> tool_call </tool_call>
tool_call: "{\\"name\\": \\"" FUNCTION_NAME "\\", \\"arguments\\": " tool_schema "}\\n"
tool_schema: %json {"type": "object", "properties": {"x": {"type": "integer"}}}
FUNCTION_NAME: /[a-zA-Z0-9_]+/
FREE: /[\\S\\s]*/
NL: /\\n/""")


def test_grammar_use_tools_explicit_tool_names():
    grammar = build_grammar([UseTools(json_schema={"type": "string"}, tool_names=["search", "fetch"])])
    assert grammar == snapshot("""\
start: <tool_call> tool_call </tool_call>
tool_call: "{\\"name\\": \\"" FUNCTION_NAME "\\", \\"arguments\\": " tool_schema "}\\n"
tool_schema: %json {"type": "string"}
FUNCTION_NAME: ("search" | "fetch")
FREE: /[\\S\\s]*/
NL: /\\n/""")


def test_grammar_use_tools_custom_regex():
    grammar = build_grammar([UseTools(json_schema={"type": "string"}, tool_name_regex="/[a-z]+/")])
    assert grammar == snapshot("""\
start: <tool_call> tool_call </tool_call>
tool_call: "{\\"name\\": \\"" FUNCTION_NAME "\\", \\"arguments\\": " tool_schema "}\\n"
tool_schema: %json {"type": "string"}
FUNCTION_NAME: /[a-z]+/
FREE: /[\\S\\s]*/
NL: /\\n/""")


def test_grammar_use_tools_custom_tokens():
    grammar = build_grammar([UseTools(json_schema={"type": "string"}, start_token="[TOOL]", stop_token="[/TOOL]")])
    assert grammar == snapshot("""\
start: [TOOL] tool_call [/TOOL]
tool_call: "{\\"name\\": \\"" FUNCTION_NAME "\\", \\"arguments\\": " tool_schema "}\\n"
tool_schema: %json {"type": "string"}
FUNCTION_NAME: /[a-zA-Z0-9_]+/
FREE: /[\\S\\s]*/
NL: /\\n/""")


# ── make_guided_extra_body ─────────────────────────────────────────────────────


def test_extra_body_structure():
    extra_body = make_guided_extra_body([Anchor("hi ")])
    assert extra_body["chat_template_kwargs"] == {"add_generation_prompt": False, "enable_thinking": False}
    assert "grammar" in extra_body["structured_outputs"]


def test_extra_body_grammar_matches_build_grammar():
    seq = [Anchor("test "), Free()]
    extra_body = make_guided_extra_body(seq)
    assert extra_body["structured_outputs"]["grammar"] == build_grammar(seq)


def test_extra_body_disable_thinking_and_prompt():
    extra_body = make_guided_extra_body([Free()])
    assert extra_body["chat_template_kwargs"]["add_generation_prompt"] is False
    assert extra_body["chat_template_kwargs"]["enable_thinking"] is False
