/*!
 *  Copyright (c) 2017 by Contributors
 * \file loop_partition.cc
 */
#include <tvm/ir.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <tvm/arithmetic.h>
#include <unordered_map>
#include <unordered_set>
#include "../arithmetic/int_set_internal.h"
#include "../runtime/thread_storage_scope.h"

namespace tvm {
namespace ir {

using arith::IntSet;
using arith::DeduceBound;
using arith::Intersect;

class IntrinsicEmitter : public IRMutator {
 public:
  explicit IntrinsicEmitter() {}

  Stmt Mutate_(const AttrStmt* op, const Stmt& stmt) {
    LOG(INFO) << "IntrinsicEmitter";
    return stmt;
  }

 private:
};

Stmt EmitIntrinsic(Stmt stmt) {
  stmt = IntrinsicEmitter().Mutate(stmt);
  return stmt;
}

}  // namespace ir
}  // namespace tvm
