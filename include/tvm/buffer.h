/*!
 *  Copyright (c) 2016 by Contributors
 * \file tvm/buffer.h
 * \brief Symbolic n-dimensional array, to represent a memory buffer.
 */
#ifndef TVM_BUFFER_H_
#define TVM_BUFFER_H_

#include <string>

#include "base.h"
#include "expr.h"
#include "ir_operator.h"
#include "tvm/node/container.h"

namespace tvm {

// Internal node container Buffer
class BufferNode;
// Internal node container DataLayout
class DataLayoutNode;

/*! \brief memory access kind */
enum class AccessMask : int {
  kRead = 1,
  kWrite = 2
};

/*!
 * \brief Buffer is a symbolic n-darray structure.
 *  It is a composition of primitive symbolic types,
 *  used to specify the memory layout of the Tensor used in program input.
 */
class Buffer : public NodeRef {
 public:
  Buffer() {}
  explicit Buffer(NodePtr<Node> n) : NodeRef(n) {}
  /*!
   * \brief Return a new buffer that is equivalent with current one
   *  but always add stride field.
   * \return The strided version of the buffer.
   */
  TVM_DLL Buffer MakeStrideView() const;
  /*!
   * \brief Make a new symbolic buffer representing a slice of the buffer.
   * \param begins The beginning position of each dimension.
   * \param extents The extent of each dimension.
   * \note This function will make target buffer as compact as possible.
   *  If stride is not needed in the slice, it won't be presented
   * \return the result buffer.
   */
  TVM_DLL Buffer MakeSlice(Array<Expr> begins, Array<Expr> extents) const;
  /*!
   * \brief Get access ptr to the entire buffer.
   * \param access_mask The access mask
   * \param ptr_type The type of the pointer.
   * \param content_lanes The number of lanes for the (data) type.
   * \param offset The offset of ptr.
   */
  TVM_DLL Expr access_ptr(int access_mask, Type ptr_type = Handle(),
                          int content_lanes = 1, Expr offset = make_const(Int(32), 0)) const;
  /*!
   * \brief Create an Expr that does a vector load at begin index.
   * \param begin The beginning index
   * \param dtype The data type to be loaded.
   */
  TVM_DLL Expr vload(Array<Expr> begin, Type dtype) const;
  /*!
   * \brief Create a Stmt that does a vector store at begin index.
   * \param begin The beginning index
   * \param value The value to be stored.
   */
  TVM_DLL Stmt vstore(Array<Expr> begin, Expr value) const;
  /*!
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  inline const BufferNode* operator->() const;

  /*! \brief specify container node */
  using ContainerType = BufferNode;
};

/*! \brief Node to represent a buffer */
class BufferNode : public Node {
 public:
  // Data fields.
  /*!
   * \brief The pointer to the head of the data
   * \sa data_alignment The alignment of data in bytes.
   */
  Var data;
  /*! \brief data type in the content of the tensor */
  Type dtype;
  /*! \brief The shape of the buffer */
  Array<Expr> shape;
  /*!
   * \brief The strides of each dimension
   *  This can be an empty array, indicating array is contiguous
   */
  Array<Expr> strides;
  /*! \brief The offset in terms of number of dtype elements (including lanes) */
  Expr elem_offset;
  // Meta data
  /*! \brief optional name of the buffer */
  std::string name;
  /*! \brief storage scope of the buffer, if other than global */
  std::string scope;
  /*! \brief Alignment requirement of data pointer in bytes. */
  int data_alignment;
  /*!
   * \brief Factor of elem_offset field,
   *  elem_offset is guaranteed to be multiple of offset_factor.
   */
  int offset_factor;
  /*! \brief constructor */
  BufferNode() {}

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("data", &data);
    v->Visit("dtype", &dtype);
    v->Visit("shape", &shape);
    v->Visit("strides", &strides);
    v->Visit("elem_offset", &elem_offset);
    v->Visit("name", &name);
    v->Visit("scope", &scope);
    v->Visit("data_alignment", &data_alignment);
    v->Visit("offset_factor", &offset_factor);
  }

  /*! \return preferred index type for this buffer node */
  Type DefaultIndexType() const {
    return shape.size() != 0 ? shape[0].type() : Int(32);
  }

  // User can specify data_alignment and offset_factor to be 0
  // A default value will be picked.
  TVM_DLL static Buffer make(Var ptr,
                             Type dtype,
                             Array<Expr> shape,
                             Array<Expr> strides,
                             Expr elem_offset,
                             std::string name,
                             std::string scope,
                             int data_alignment,
                             int offset_factor);

  static constexpr const char* _type_key = "Buffer";
  TVM_DECLARE_NODE_TYPE_INFO(BufferNode, Node);
};

inline const BufferNode* Buffer::operator->() const {
  return static_cast<const BufferNode*>(node_.get());
}

/*!
 * \brief Construct a new buffer given shape, and dtype.
 * \param shape The shape of the buffer,
 * \param dtype The content data type.
 * \param name The name of the buffer
 * \return The created buffer.
 * \sa BufferNode::make for complete constructor.
 */
TVM_DLL Buffer decl_buffer(Array<Expr> shape,
                           Type dtype = Float(32),
                           std::string name = "buffer");


class DataLayout : public NodeRef {
 public:
  DataLayout() {}
  explicit DataLayout(NodePtr<Node> n) : NodeRef(n) {}

  // Final shape of the underlying array, given the shape of the normal layout
  TVM_DLL Array<Expr> ForwardShape(const Array<Expr>& shape) const;
  // Given final shape, recover the original shape.
  TVM_DLL Array<Expr> BackwardShape(const Array<Expr>& shape) const;
  // Final index of the underlying array, given the normal layout.
  TVM_DLL Array<Expr> ForwardIndex(const Array<Expr>& index) const;
  // Given store index, recover the original representation space index.
  TVM_DLL Array<Expr> BackwardIndex(const Array<Expr>& store_index) const;

  /*!
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  inline const DataLayoutNode* operator->() const;

  /*! \brief specify container node */
  using ContainerType = DataLayoutNode;
};

class DataLayoutNode : public Node {
 public:
  // The original axis, with symbolic shape
  Array<IterVar> orig_axis;
  Array<IterVar> store_axis;
  // The shape of the stored array
//  Array<Expr> shape;
  // expression of each location, on how original location can be mapped
  // to the store location, example
  // [i0 / 16, i1, i0 % 16]
  Array<Expr> forward_rule;
  Array<Expr> backward_rule;

  std::string orig_layout;
  std::string store_layout;

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("orig_axis", &orig_axis);
//    v->Visit("shape", &shape);
    v->Visit("store_axis", &store_axis);
    v->Visit("orig_layout", &orig_layout);
    v->Visit("store_layout", &store_layout);
  }

  TVM_DLL static DataLayout make(const std::string& orig_layout,
                                 const std::string& store_layout);

  static constexpr const char* _type_key = "DataLayout";
  TVM_DECLARE_NODE_TYPE_INFO(DataLayoutNode, Node);

  inline static char GetAxisName(const IterVar& axis) {
    return axis->var.get()->name_hint.at(0);
  }
  inline static bool IsMajorAxis(const IterVar& axis) {
    return GetAxisName(axis) >= 'A' && GetAxisName(axis) <= 'Z';
  }
  inline static bool Match(const IterVar& x, const IterVar& y) {
    const char x_name = IsMajorAxis(x) ? GetAxisName(x) : GetAxisName(x) - 'a' + 'A';
    const char y_name = IsMajorAxis(y) ? GetAxisName(y) : GetAxisName(y) - 'a' + 'A';
    return x_name == y_name;
  }

 private:
  inline static bool GetStoreRule(Array<Expr>& rule,
                                  const Array<IterVar>& orig_axes,
                                  const Array<IterVar>& store_axes) {
    for (const IterVar& axis : store_axes) {
      Expr store(0);
      for (const IterVar& orig_axis : orig_axes) {
        if (Match(axis, orig_axis)) {
          if (IsMajorAxis(orig_axis)) {
            Expr orig_var = orig_axis->var;
            // TODO: avoid for loop
            for (const IterVar& temp_axis : orig_axes) {
              if (!IsMajorAxis(temp_axis) && Match(temp_axis, orig_axis)) {
                orig_var = orig_var * temp_axis->dom->extent;
              }
            }
            store = store + orig_var;
          } else {
            store = store + orig_axis->var;
          }
        }
      }
      if (is_zero(store)) {
        // Not convertible
        return false;
      }
      if (IsMajorAxis(axis)) {
        // TODO: avoid for loop
        for (const IterVar& temp_axis : store_axes) {
          if (!IsMajorAxis(temp_axis) && Match(temp_axis, axis)) {
            store = store / temp_axis->dom->extent;
          }
        }
      } else {
        store = store % axis->dom->extent;
      }
      rule.push_back(store);
    }
    return true;
  }
  /*
  inline static bool GetShapeRule(Array<Expr>& rule,
                                  const Array<IterVar>& orig_axes,
                                  const Array<IterVar>& store_axes) {
    for (const IterVar& axis : store_axes) {
      if (IsMajorAxis(axis)) {
        Expr store(1);
        for (const IterVar &orig_axis : orig_axes) {
          if (Match(axis, orig_axis)) {
            store = store * orig_axis->dom->extent;
          }
        }
        if (is_one(store)) {
          // Not convertible
          return false;
        }
        for (const IterVar& temp_axis : store_axes) {
          if (!IsMajorAxis(temp_axis) && Match(temp_axis, axis)) {
            store = store / temp_axis->dom->extent;
          }
        }
      } else {
      }
      rule.push_back(store);
    }
  }
  */
};

inline const DataLayoutNode* DataLayout::operator->() const {
  return static_cast<const DataLayoutNode*>(node_.get());
}

}  // namespace tvm
#endif  // TVM_BUFFER_H_
