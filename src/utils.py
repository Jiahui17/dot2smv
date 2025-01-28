import sys, subprocess, re, ast
from typing import Tuple
from networkx import nx_agraph

""" convert port from dot format to hdl format """

COLOR_RED = "\033[0;31m"
COLOR_GREEN = "\033[0;32m"
COLOR_CYAN = "\033[0;36m"
COLOR_NC = "\033[0m"

def print_msg(*args, **kwargs):
    print(COLOR_GREEN + "[INFO]", *args, COLOR_NC, file=sys.stderr, **kwargs)

def print_err(*args, **kwargs):
    print(COLOR_RED + "[ERROR]", *args, COLOR_NC, file=sys.stderr, **kwargs)

def parse_port(portid):
    return int(re.sub(r"(out|in)", "", portid)) - 1


def nx2dot(dfg, name: str = "test.dot"):
    nx_agraph.to_agraph(dfg).write(f"./reports/{name}")


def run(*args, **kwargs):
    sys.stdout.flush()
    return subprocess.run(*args, **kwargs, check=True)


PATTERN_OPERATOR = r"(add|ashr|shl|sub|lshr|fneg|sext|zext|getelementptr|mul|fmul|udiv|urem|sdiv|srem|fadd|fsub|fdiv|sitofp|trunc)_op"

PATTREN_DECIDER = r"(and|or|icmp_\w*|fcmp_\w*)_op"

MLIR_OPERATOR_TYPES=r"handshake.(add|ashr|shl|sub|lshr|fneg|extsi|extui|getelementptr|mul|fmul|udiv|urem|sdiv|srem|addf|subf|divf|sitofp|trunci)"

MLIR_DECIDER_TYPES = r"handshake.(cmpi[<>!=]*|cmpf[<>!=]*)"

# returns true if the op is an operator or decider
def is_operator_or_decider(attr) -> bool:
    return re.match(MLIR_OPERATOR_TYPES, attr["mlir_op"]) or re.match(MLIR_DECIDER_TYPES, attr["mlir_op"])

def parse_buffer_attr(attr : dict) -> Tuple[str, str]:
    m = re.search(r"(tehb|oehb) \[(\d+)\]", attr["label"])
    if m:
        # print_err("Matched buffer type line", m.string)
        transparent = "true" if m.group(1) == "tehb" else "false"
        slots = int(m.group(2))
        return transparent, slots
    else:
        raise ValueError

# NOTE: Very hacky way of parsing the constant value, which only works when we
# abstract the data
def parse_constant_value(attr : dict) -> str:
    value = attr["label"]
    if value == "false" or ast.literal_eval(value) == 0:
        return "FALSE"
    else:
        return "TRUE"


def get_op_type(attr):
    latency = int(attr.get("latency", 0))
    if "latency" not in attr:
        # print_err("Be careful! The latency is not specified!")
        pass

    if re.match(MLIR_OPERATOR_TYPES, attr["mlir_op"]):
        # print_msg("Matching operator!")
        return f"operator{latency}c"
    elif re.match(MLIR_DECIDER_TYPES, attr["mlir_op"]):
        if attr["mlir_op"] == "handshake.cmpi==":
            pass
            return "eq"
        else:
            # print_msg("Matching decider!")
            return f"decider{latency}c"
    else:
        raise ValueError(f'error - unknown Operator {attr["op"]}')

def remove_indent(string):
    indents = []
    for line in string.split("\n"):
        if line == "":
            continue
        indents.append(len(re.findall(r"\t", line)))

    min_indent = min(indents)
    print(min_indent)
    cleaned_up_strings = []
    for line in string.split("\n"):
        line = re.sub("\t", "", line, count=min_indent)
        cleaned_up_strings.append(line)
    return "\n".join(cleaned_up_strings)


def include_guard(func):
    def wrapper_include_guard(*args, **kwargs):
        name = "_".join(
            [str(func.__name__)]
            + list(str(n) for n in args)
            + [str(key) + "_" + str(value) for key, value in kwargs.items()]
        )
        name = re.sub(r"\W", "_", name.upper())
        to_return = f"""
		#ifndef {name}
		#define {name}
		"""
        type_ = type(func(*args, **kwargs))
        if type_ != str:
            print(f"error - type of the return value {type_} is not str!")
            raise TypeError
        to_return += func(*args, **kwargs)
        to_return += f"""
		#endif // {name}
		"""
        return to_return

    return wrapper_include_guard
