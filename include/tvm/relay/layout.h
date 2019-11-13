#ifndef TVM_RELAY_LAYOUT_H_
#define TVM_RELAY_LAYOUT_H_

#include <tvm/data_layout.h>
#include "./expr.h"

namespace tvm {
namespace relay {

/*! \brief Base type of the Relay Layout hierarchy. */
class RelayLayoutNode : public RelayNode {
 public:
  static constexpr const char* _type_key = "relay.Layout";
  TVM_DECLARE_BASE_NODE_INFO(RelayLayoutNode, Node);
};

class RelayLayout : public NodeRef {
 public:
  RelayLayout() {}
  explicit RelayLayout(ObjectPtr<tvm::Object> p) : NodeRef(p) {}
  /*!
   * \brief Whether the two layouts are equal.
   * \param rhs Another layout.
   * \return whether the two layouts are equal.
   */
  bool Equals(const RelayLayout& rhs) const;
  using ContainerType = RelayLayoutNode;
};

class TensorLayout;
/*! \brief TensorType container node */
class TensorLayoutNode : public RelayLayoutNode {
 public:
  Layout layout;

  void VisitAttrs(tvm::AttrVisitor *v) {
    v->Visit("layout", &layout);
  }

  TVM_DLL static TensorLayout make(const Layout& layout);

  static constexpr const char* _type_key = "relay.TensorLayout";
  TVM_DECLARE_NODE_TYPE_INFO(TensorLayoutNode, RelayLayoutNode);
};

RELAY_DEFINE_NODE_REF(TensorLayout, TensorLayoutNode, RelayLayout);

class TupleLayout;
/*!
 * \brief TupleType container.
 */
class TupleLayoutNode : public RelayLayoutNode {
 public:
  /*! \brief The type of each field in the tuple. */
  tvm::Array<Layout> fields;

  TupleLayoutNode() {}

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("fields", &fields);
  }

  TVM_DLL static TupleLayout make(tvm::Array<Layout> fields);

  static constexpr const char* _type_key = "relay.TupleLayout";
  TVM_DECLARE_NODE_TYPE_INFO(TupleLayoutNode, RelayLayoutNode);
};

RELAY_DEFINE_NODE_REF(TupleLayout, TupleLayoutNode, RelayLayout);

class LayoutReporter;

/*!
 * \brief reporter that reports back to the
 *  type resolution information.
 */
class LayoutReporterNode : public Node {
 public:
  tvm::Array<Expr> args;
  tvm::Array<RelayLayout> args_layout;
  tvm::Map<Expr, RelayLayout> results;
  /*!
   * \brief Create a type equality constraint.
   *
   *  The "assign direction" acts as a hint to the solver
   *  showing that it is more likely to resolve dst by src.
   *  But it is possible for the solver to resolve src by dst as well.
   */
  TVM_DLL void Assign(size_t index, const RelayLayout& layout);
  TVM_DLL static LayoutReporter make(tvm::Array<Expr> args, tvm::Array<RelayLayout> args_layout);

  // solver is not serializable.
  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("args", &args);
    v->Visit("args_layout", &args_layout);
    v->Visit("results", &results);
  }

  static constexpr const char* _type_key = "relay.LayoutReporter";
  TVM_DECLARE_NODE_TYPE_INFO(LayoutReporterNode, Node);
};

/*!
 * \brief Container class of LayoutReporter.
 * \sa LayoutReporterNode
 */
class LayoutReporter : public NodeRef {
 public:
  LayoutReporter() {}
  explicit LayoutReporter(::tvm::ObjectPtr<::tvm::Object> n) : NodeRef(n) {
  }
  LayoutReporterNode* operator->() const {
    return const_cast<LayoutReporterNode*>(
        static_cast<const LayoutReporterNode*>(get()));
  }
  using ContainerType = LayoutReporterNode;
};

/*!
 * \brief Infer & correct function of node layout. See \p Layout for layout convention
 * \param attrs The attribute of the node.
 * \param new_in_layouts The layouts of input arguments after alter_op_layout.
 *                       This can be undefined, which means we call this function before alternating
 *                       any operators.
 * \param old_in_layouts The layouts of input arguments before alter_op_layout.
 * \param old_in_shapes The shapes of old input arguments.
 * \return infered_layout An array of two elements that are inferred input layouts and
 *                        inferred output layouts.
 */
using FInferLayout = runtime::TypedPackedFunc <
    bool(const Array<RelayLayout>& layouts,
         const Array<Type>& types,
         int num_inputs,
         const Attrs& attrs,
         const LayoutReporter& reporter)>;

}  // namespace relay
}  // namespace tvm

#endif // TVM_RELAY_LAYOUT_H_

