package ml.dmlc.tvm.contrib;

import ml.dmlc.tvm.Function;
import ml.dmlc.tvm.Module;
import ml.dmlc.tvm.NDArray;
import ml.dmlc.tvm.TVMContext;

public class GraphRuntime {
  public static GraphModule create(String graphJson, Module libmod, TVMContext ctx) {
    Function fcreate = Function.getFunction("tvm.graph_runtime.create");
    if (fcreate == null) {
      throw new RuntimeException("Cannot find global function tvm.graph_runtime.create." +
          "Did you compile tvm_runtime with correct version?");
    }
    Module module = fcreate.pushArg(graphJson)
        .pushArg(libmod).pushArg(ctx.deviceType).pushArg(ctx.deviceId)
        .invoke().asModule();
    return new GraphModule(module, ctx);
  }

  public static class GraphModule {
    private Module module;
    private TVMContext ctx;

    private Function fSetInput;
    private Function fRun;
    private Function fGetOutput;
    private Function fGetInput;
    private Function fDebugGetOutput;
    private Function fLoadParams;

    private GraphModule(Module module, TVMContext ctx) {
      this.module = module;
      this.ctx = ctx;
      fSetInput = module.getFunction("set_input");
      fRun = module.getFunction("run");
      fGetInput = module.getFunction("get_input");
      fGetOutput = module.getFunction("get_output");
      try {
        fDebugGetOutput = module.getFunction("debug_get_output");
      } catch (IllegalArgumentException ignored) {}
      fLoadParams = module.getFunction("load_params");
    }

    public GraphModule setInput(String key, NDArray value) {
      NDArray v = value;
      if (!value.ctx().equals(ctx)) {
        v = NDArray.empty(value.shape(), ctx);
        value.copyTo(v);
      }
      fSetInput.pushArg(key).pushArg(v).invoke();
      return this;
    }

    public GraphModule setInput(int key, NDArray value) {
      NDArray v = value;
      if (!value.ctx().equals(ctx)) {
        v = NDArray.empty(value.shape(), ctx);
        value.copyTo(v);
      }
      fSetInput.pushArg(key).pushArg(v).invoke();
      return this;
    }

    public GraphModule run() {
      fRun.invoke();
      return this;
    }

    public NDArray getInput(int index, NDArray out) {
      fGetInput.pushArg(index).pushArg(out).invoke();
      return out;
    }

    public NDArray getOutput(int index, NDArray out) {
      fGetOutput.pushArg(index).pushArg(out).invoke();
      return out;
    }

    public NDArray debugGetOutput(String node, NDArray out) {
      if (fDebugGetOutput != null) {
        fDebugGetOutput.pushArg(node).pushArg(out).invoke();
      } else {
        throw new RuntimeException("Please compile runtime with USE_GRAPH_RUNTIME_DEBUG = 0");
      }
      return out;
    }

    public NDArray debugGetOutput(int node, NDArray out) {
      if (fDebugGetOutput != null) {
        fDebugGetOutput.pushArg(node).pushArg(out).invoke();
      } else {
        throw new RuntimeException("Please compile runtime with USE_GRAPH_RUNTIME_DEBUG = 0");
      }
      return out;
    }

    public GraphModule loadParams(byte[] params) {
      fLoadParams.pushArg(params).invoke();
      return this;
    }

    public Function getFunction(String key) {
      return module.getFunction(key);
    }
  }
}
