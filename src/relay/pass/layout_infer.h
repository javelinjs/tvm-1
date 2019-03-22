#ifndef TVM_LAYOUT_INFER_H
#define TVM_LAYOUT_INFER_H

#include <tvm/data_layout.h>

namespace tvm {
namespace relay {

class RelayBaseLayoutNode : public RelayNode {
 public:
  static constexpr const char* _type_key = "relay.BaseLayout";
  TVM_DECLARE_BASE_NODE_INFO(RelayBaseLayoutNode, Node);
};

class RelayBaseLayout : public NodeRef {
 public:
  RelayBaseLayout() {}
  explicit RelayBaseLayout(NodePtr<tvm::Node> p) : NodeRef(p) {}

  Array<Layout> Flatten() const;
  using ContainerType = RelayBaseLayoutNode;
};

class RelayLayout;
class RelayLayoutNode : public RelayBaseLayoutNode {
 public:
  /*! \brief The type of each field in the tuple. */
  Layout layout;

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("layout", &layout);
  }

  TVM_DLL static RelayLayout make(Layout layout);

  static constexpr const char* _type_key = "relay.Layout";
  TVM_DECLARE_NODE_TYPE_INFO(RelayLayoutNode, RelayBaseLayoutNode);
};

class RelayLayout : public RelayBaseLayout {
 public:
  explicit RelayLayout(NodePtr<Node> n) : RelayBaseLayout(n) {}
  using ContainerType = RelayLayoutNode;
};

class RelayTupleLayout;
class RelayTupleLayoutNode : public RelayBaseLayoutNode {
 public:
  /*! \brief The type of each field in the tuple. */
  tvm::Array<RelayBaseLayout> fields;

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("fields", &fields);
  }

  TVM_DLL static RelayTupleLayout make(tvm::Array<RelayBaseLayout> fields);

  static constexpr const char* _type_key = "relay.TupleLayout";
  TVM_DECLARE_NODE_TYPE_INFO(RelayTupleLayoutNode, RelayBaseLayoutNode);
};

class RelayTupleLayout : public RelayBaseLayout {
 public:
  explicit RelayTupleLayout(NodePtr<Node> n) : RelayBaseLayout(n) {}
  using ContainerType = RelayTupleLayoutNode;
};


class LayoutInferencer : private ExprFunctor<RelayBaseLayout(const Expr&)> {
 public:
  // inference the type of expr.
  Map<Expr, RelayBaseLayout> Infer(Expr expr);

 private:
  // map from expression to checked type
  // type inferencer will populate it up
  std::unordered_map<Expr, RelayBaseLayout, NodeHash, NodeEqual> layout_map_;

  // Lazily get type for expr
  // expression, we will populate it now, and return the result.
  RelayBaseLayout GetLayout(const Expr& expr) {
    auto it = layout_map_.find(expr);
    if (it != layout_map_.end()) {
      return it->second;
    }
    RelayBaseLayout layout = this->VisitExpr(expr);
    layout_map_[expr] = layout;
    return layout;
  }

  // Visitor Logic
  RelayBaseLayout VisitExpr_(const VarNode* op) final {
    return RelayLayoutNode::make(Layout::Undef());
  }

  RelayBaseLayout VisitExpr_(const GlobalVarNode* op) final {
    GlobalVar var = GetRef<GlobalVar>(op);
    LOG(FATAL) << "GlobalVarNode not supported.";
  }

  RelayBaseLayout VisitExpr_(const ConstantNode* op) final {
    return RelayLayoutNode::make(Layout::Undef());
  }

  RelayBaseLayout VisitExpr_(const TupleNode* op) final {
    Array<RelayBaseLayout> layouts;
    for (const Expr& field : op->fields) {
      layouts.push_back(GetLayout(field));
    }
    return RelayTupleLayoutNode::make(layouts);
  }

  RelayBaseLayout VisitExpr_(const TupleGetItemNode* op) final {
    RelayBaseLayout tuple_layout = GetLayout(op->tuple);
    const auto* layouts = tuple_layout.as<RelayTupleLayoutNode>();
    CHECK(layouts);
    return layouts->fields[op->index];
  }

  RelayBaseLayout VisitExpr_(const MatchNode* op) final {
    LOG(FATAL) << "MatchNode not supported.";
  }

  RelayBaseLayout VisitExpr_(const OpNode* op) final {
    LOG(FATAL) << "OpNode not supported.";
  }

  RelayBaseLayout VisitExpr_(const LetNode* let) final {
    LOG(FATAL) << "LetNode not supported.";
  }

  RelayBaseLayout VisitExpr_(const IfNode* ite) final {
    LOG(FATAL) << "IfNode not supported.";
  }

  RelayBaseLayout VisitExpr_(const CallNode* call) final;

  RelayBaseLayout VisitExpr_(const FunctionNode* f) final {
    Array<Type> arg_types;
    for (auto param : f->params) {
      GetLayout(param);
    }
    return GetLayout(f->body);
  }

  RelayBaseLayout VisitExpr_(const RefCreateNode* op) final {
    LOG(FATAL) << "RefCreateNode not supported.";
  }

  RelayBaseLayout VisitExpr_(const RefReadNode* op) final {
    LOG(FATAL) << "RefReadNode not supported.";
  }

  RelayBaseLayout VisitExpr_(const RefWriteNode* op) final {
    LOG(FATAL) << "RefWriteNode not supported.";
  }

  RelayBaseLayout VisitExpr_(const ConstructorNode* c) final {
    LOG(FATAL) << "ConstructorNode not supported.";
  }
};

}  //  namespace relay
}  //  namespace tvm

#endif //TVM_LAYOUT_INFER_H
