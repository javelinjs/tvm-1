package ml.dmlc.tvm.contrib;

import ml.dmlc.tvm.Function;
import ml.dmlc.tvm.Module;
import ml.dmlc.tvm.NDArray;
import ml.dmlc.tvm.TVMContext;
import ml.dmlc.tvm.TVMValue;
import ml.dmlc.tvm.rpc.RPC;
import ml.dmlc.tvm.rpc.RPCSession;

public class GraphRuntime {
  /**
   * Create a runtime executor module given a graph and module.
   * @param graphJson The graph deployed in json format output by nnvm graph.
   * @param libmod The module of the corresponding function.
   * @param ctx The local context to deploy the module.
   * @return Runtime graph module that can be used to execute the graph.
   */
  public static GraphModule create(String graphJson, Module libmod, TVMContext ctx) {
    return create(graphJson, libmod, ctx, null);
  }

  /**
   * Create a runtime executor module given a graph and module.
   * @param graphJson The graph deployed in json format output by nnvm graph.
   * @param libmod The module of the corresponding function.
   * @param ctx The context to deploy the module.
   *            If it is remote, the related rpcSession must be provided.
   * @param rpcSession RPCSession related to the module and context.
   * @return Runtime graph module that can be used to execute the graph.
   */
  public static GraphModule create(String graphJson, Module libmod,
                                   TVMContext ctx, RPCSession rpcSession) {
    Module graphModule = null;
    if (rpcSession != null && ctx.deviceType >= RPC.RPC_SESS_MASK) {
      // check arguments
      if (!"rpc".equals(libmod.typeKey())) {
        throw new IllegalArgumentException("libmod.typeKey != rpc");
      }
      final int sessIndex = (int) RPC.getApi("_SessTableIndex").pushArg(libmod).invoke().asLong();
      if (sessIndex != rpcSession.tblIndex) {
        throw new IllegalArgumentException(String.format(
            "libmod SessTableIndex=%d mismatch rpcSession.tblIndex=%d",
            sessIndex, rpcSession.tblIndex));
      }

      Function rpcModuleHandle = RPC.getApi("_ModuleHandle");
      if (rpcModuleHandle == null) {
        throw new RuntimeException("Cannot find global function tvm.rpc._ModuleHandle."
            + "Did you compile tvm_runtime with the correct version?");
      }
      TVMValue hmod = rpcModuleHandle.pushArg(libmod).invoke();
      Function fcreate = Function.getFunction("tvm.graph_runtime.remote_create");
      if (fcreate == null) {
        throw new RuntimeException("Cannot find global function tvm.graph_runtime.remote_create."
            + "Did you compile tvm_runtime with correct version?");
      }
      graphModule = fcreate.call(graphJson, hmod,
          ctx.deviceType % RPC.RPC_SESS_MASK, ctx.deviceId).asModule();
    } else if (rpcSession == null && ctx.deviceType < RPC.RPC_SESS_MASK) {
      Function fcreate = Function.getFunction("tvm.graph_runtime.create");
      if (fcreate == null) {
        throw new RuntimeException("Cannot find global function tvm.graph_runtime.create."
            + "Did you compile tvm_runtime with correct version?");
      }
      graphModule = fcreate.pushArg(graphJson)
          .pushArg(libmod).pushArg(ctx.deviceType).pushArg(ctx.deviceId)
          .invoke().asModule();
    } else {
      throw new IllegalArgumentException("libmod and rpcSession do not match.");
    }

    return new GraphModule(graphModule, ctx);
  }

  /**
   * Wrapper runtime module.
   * This is a thin wrapper of the underlying TVM module.
   * you can also directly call set_input, run, and get_output
   * of underlying module functions.
   */
  public static class GraphModule {
    private Module module;
    private TVMContext ctx;

    private Function fsetInput;
    private Function frun;
    private Function fgetOutput;
    private Function fgetInput;
    private Function fdebugGetOutput;
    private Function floadParams;

    private GraphModule(Module module, TVMContext ctx) {
      this.module = module;
      this.ctx = ctx;
      fsetInput = module.getFunction("set_input");
      frun = module.getFunction("run");
      fgetInput = module.getFunction("get_input");
      fgetOutput = module.getFunction("get_output");
      try {
        fdebugGetOutput = module.getFunction("debug_get_output");
      } catch (IllegalArgumentException ignored) {
        // ignore
      }
      floadParams = module.getFunction("load_params");
    }

    /**
     * Set inputs to the module.
     * @param key The input key.
     * @param value The input value
     * @return self.
     */
    public GraphModule setInput(String key, NDArray value) {
      NDArray input = value;
      if (!value.ctx().equals(ctx)) {
        input = NDArray.empty(value.shape(), ctx);
        value.copyTo(input);
      }
      fsetInput.pushArg(key).pushArg(input).invoke();
      return this;
    }

    /**
     * Set inputs to the module
     * @param key The input key.
     * @param value The input value.
     * @return self.
     */
    public GraphModule setInput(int key, NDArray value) {
      NDArray input = value;
      if (!value.ctx().equals(ctx)) {
        input = NDArray.empty(value.shape(), ctx);
        value.copyTo(input);
      }
      fsetInput.pushArg(key).pushArg(input).invoke();
      return this;
    }

    /**
     * Run forward execution of the graph.
     * @return self.
     */
    public GraphModule run() {
      frun.invoke();
      return this;
    }

    /**
     * Get index-th input to out.
     * @param index The input index.
     * @param out The output array container.
     * @return out.
     */
    public NDArray getInput(int index, NDArray out) {
      fgetInput.pushArg(index).pushArg(out).invoke();
      return out;
    }

    /**
     * Get index-th output to out.
     * @param index The output index.
     * @param out The output array container.
     * @return out.
     */
    public NDArray getOutput(int index, NDArray out) {
      fgetOutput.pushArg(index).pushArg(out).invoke();
      return out;
    }

    /**
     * Run graph up to node and get the output to out.
     * @param node The node name.
     * @param out The output array container.
     * @return out.
     */
    public NDArray debugGetOutput(String node, NDArray out) {
      if (fdebugGetOutput != null) {
        fdebugGetOutput.pushArg(node).pushArg(out).invoke();
      } else {
        throw new RuntimeException("Please compile runtime with USE_GRAPH_RUNTIME_DEBUG = 0");
      }
      return out;
    }

    /**
     * Run graph up to node and get the output to out.
     * @param node The node index.
     * @param out The output array container.
     * @return out.
     */
    public NDArray debugGetOutput(int node, NDArray out) {
      if (fdebugGetOutput != null) {
        fdebugGetOutput.pushArg(node).pushArg(out).invoke();
      } else {
        throw new RuntimeException("Please compile runtime with USE_GRAPH_RUNTIME_DEBUG = 0");
      }
      return out;
    }

    /**
     * Load parameters from serialized byte array of parameter dict.
     * @param params The serialized parameter.
     * @return self.
     */
    public GraphModule loadParams(byte[] params) {
      floadParams.pushArg(params).invoke();
      return this;
    }

    /**
     * Get internal module function.
     * @param key The key to the module.
     * @return The function.
     * @throws IllegalArgumentException if function does not exist.
     */
    public Function getFunction(String key) {
      return module.getFunction(key);
    }
  }
}
