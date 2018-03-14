/*!
 *  Copyright (c) 2017 by Contributors
 * \file Use external generic C library function
 */

#ifndef NNVM_ALGORITHM_H
#define NNVM_ALGORITHM_H

#include <tvm/runtime/registry.h>

namespace tvm {
namespace contrib {

template<typename DType>
struct SortElem {
  DType value;
  int32_t index;
  static bool is_descend;

  SortElem(DType v, int32_t i) {
    value = v;
    index = i;
  }

  bool operator<(const SortElem &other) const {
    return is_descend ? value > other.value : value < other.value;
  }
};

} // namespace contrib
} // namespace tvm

#endif //NNVM_ALGORITHM_H
