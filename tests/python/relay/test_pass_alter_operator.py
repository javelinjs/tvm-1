from tvm import relay
from tvm.relay.op import register_alter_operator
from tvm.relay.ir_pass import *


def assert_alpha_equal(expected, actual):
    assert alpha_equal(expected, actual), "\nExpected:\n" + str(expected) \
                                          + "\nActual:\n" + str(actual)

def test_alter_split():
    shape = (1, 64, 56, 56)
    def before():
        data = relay.var("data", shape=shape)
        splits = relay.split(data, indices_or_sections=2, axis=2)
        outputs = []
        for i in range(len(splits)):
            outputs.append(relay.nn.relu(splits[i]))
        ret = relay.Tuple(outputs)
        return relay.Function(free_vars(ret), ret)

    @register_alter_operator("split", level=112)
    def alter_split(attrs, inputs, tinfos, in_layouts, out_layouts):
        outputs = []
        splits = relay.split(inputs[0], indices_or_sections=2, axis=2)
        for i in range(len(splits)):
            outputs.append(relay.nn.relu(splits[i]))
        return relay.Tuple(outputs)

    def expected():
        outputs = []
        data = relay.var("data", shape=shape)
        splits = relay.split(data, indices_or_sections=2, axis=2)
        for i in range(len(splits)):
            branch = relay.nn.relu(splits[i])
            branch = relay.nn.relu(branch)
            outputs.append(branch)
        ret = relay.Tuple(outputs)
        return relay.Function(free_vars(ret), ret)

    a = before()
    a = infer_type(a)
    a = alter_operator(a)

    b = expected()
    b = infer_type(b)

    assert_alpha_equal(b, a)


def test_infer_layout():
    x = relay.var("x", shape=(1, 64, 56, 56))
    weight = relay.var('weight', shape=(64, 64, 3, 3))
    y = relay.nn.conv2d(x, weight,
                        channels=64,
                        kernel_size=(3, 3),
                        padding=(1, 1))
    # y = relay.nn.relu(y)
    y = relay.Function([x, weight], y)

    y = infer_type(y)

    @register_alter_operator("nn.conv2d", level=112)
    def alter_conv2d(attrs, inputs, tinfos, in_layouts, out_layouts):
        for i in range(len(in_layouts)):
            print("in layout " + str(i), in_layouts[i])
        data, kernel = inputs
        new_attrs = {k : attrs[k] for k in attrs.keys()}
        new_attrs["data_layout"] = "NCHW16c"
        new_attrs["out_layout"] = "NCHW16c"
        new_attrs["kernel_layout"] = "OIHW16i16o"
        data = relay.layout_transform(data, src_layout="NCHW", dst_layout="NCHW16c")
        conv = relay.nn.contrib_conv2d_nchwc(data, kernel, **new_attrs)
        return relay.layout_transform(conv, src_layout="NCHW16c", dst_layout="NCHW")

    y = alter_operator(y)
    print(y)

if __name__ == "__main__":
    test_alter_split()
    test_infer_layout()