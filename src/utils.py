import sys, subprocess, re
from networkx import nx_agraph

""" convert port from dot format to hdl format """


def parse_port(portid):
    return int(re.sub(r"(out|in)", "", portid)) - 1


def nx2dot(dfg, name: str = "test.dot"):
    nx_agraph.to_agraph(dfg).write(f"./reports/{name}")


def run(*args, **kwargs):
    sys.stdout.flush()
    return subprocess.run(*args, **kwargs, check=True)


PATTERN_OPERATOR = r"(add|ashr|shl|sub|lshr|fneg|sext|zext|getelementptr|mul|fmul|udiv|urem|sdiv|srem|fadd|fsub|fdiv|sitofp|trunc)_op"

PATTREN_DECIDER = r"(and|or|icmp_\w*|fcmp_\w*)_op"


def get_op_type(attr):
    if not (type(attr) == dict and "type" in attr and attr["type"] == "Operator"):
        print(attr)
        assert False
    latency = int(attr.get("latency", 0))

    if re.match(PATTERN_OPERATOR, attr["op"]):
        return f"operator{latency}c"
    elif re.match(PATTREN_DECIDER, attr["op"]):
        return f"decider{latency}c"
    elif re.match(r"(lsq|mc)_(load|store)_op", attr["op"]) or re.match(
        r"select_op", attr["op"]
    ):
        return attr["op"]
    elif re.match(r"ret_op", attr["op"]):
        return "tehb"
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
