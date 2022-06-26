/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <iostream>

#include "velox/common/memory/Memory.h"
#include "velox/functions/Udf.h"
#include "velox/type/Type.h"
#include "velox/vector/BaseVector.h"

using namespace facebook::velox;

/// This file contains a step-by-step usage example of Velox's expression
/// evaluation engine.
///
/// It shows how to register a simple function and describes all the steps
/// required to create the appropriate query and expression structures, a simple
/// expression tree, an input batch of data, and execute the expression over it.
///
/// 它展示了如何注册一个简单的函数，并描述了创建适当的查询和表达式结构、
/// 一个简单的表达式树、一批输入数据以及在其上执行表达式所需的所有步骤。

// First, define a toy function that multiplies the input argument by two.
//
// Check `velox/docs/develop/scalar-functions.rst` for more documentation on how
// to build scalar functions.
// 定义一个简单的标量函数，实现 * 2计算
template <typename T>
struct TimesTwoFunction {
  FOLLY_ALWAYS_INLINE bool call(int64_t& out, const int64_t& a) {
    out = a * 2;
    return true; // True if result is not null.
  }
};

int main(int argc, char** argv) {
  // Register the function defined above. The first template parameter is the
  // class that implements the `call()` function (or one of its variations), the
  // second template parameter is the function return type, followed by the list
  // of function input parameters.
  //
  // This function takes as an argument a list of aliases for the function being
  // registered.
  // template <template <class> typename Func, typename TReturn, typename... TArgs>
  // 注册上面定义的函数。 第一个模板参数是实现 `call()` 函数（或其变体）的类，
  // 第二个模板参数是函数返回类型，然后是函数输入参数列表。
  //
  // 这个函数将一个被注册函数的别名列表作为参数
  registerFunction<TimesTwoFunction, int64_t, int64_t>({"times_two"});

  // First of all, executing an expression in Velox will require us to create a
  // query context, a memory pool, and an execution context.
  //
  // QueryCtx holds the metadata and configuration associated with a
  // particular query. This is shared between all threads of execution
  // for the same query (one object per query).
  //
  // 对于查询来讲，需要有一个context来管理内存池等一些执行状态，所以需要先创建好一个query context.
  // QueryContext 持有查询时的metadata和配置， 可以多个线程共享
  auto queryCtx = core::QueryCtx::createForTest();

  // ExecCtx holds structures associated with a single thread of execution
  // (one per thread). Each thread of execution requires a scoped memory pool,
  // which is where allocations from this thread will be made. When required, a
  // pointer to this pool can be obtained using execCtx.pool().
  //
  // Optionally, one can control the per-thread memory cap by passing it as an
  // argument to getDefaultScopedMemoryPool() - no limit by default.
  //
  // ExecCtx 保存与单个执行线程相关的结构
  //（每个线程一个）。 每个执行线程都需要一个作用域内存池，
  // 这是从这个线程进行分配的地方。 需要时，一个
  // 可以使用 execCtx.pool() 获得指向该池的指针。
  //
  // 可选地，可以通过将每个线程的内存上限作为
  // getDefaultScopedMemoryPool() 的参数 - 默认没有限制。
  auto pool = memory::getDefaultScopedMemoryPool();
  core::ExecCtx execCtx{pool.get(), queryCtx.get()};

  // Next, let's create an expression tree to be executed in this example. On a
  // high-level, our expression tree will have the following structure:
  //
  // -----------------------------
  // | CallTypedExpr (times_two) |  => root
  // -----------------------------
  //            /\
  //            ||
  // ---------------------------------
  // | FieldAccessTypedExpr (my_col) |
  // ---------------------------------
  //
  // Let's first define a type for the input dataset used in this example. In
  // this case, a single input column called "my_col", typed as bigint:
  // 创建一个RowSchema, 只有一个int列
  auto inputRowType = ROW({{"my_col", BIGINT()}});

  // FieldAccessTypedExpr let us choose a particular field/column from the input
  // dataset(s). The first parameter defines the return type of this field, the
  // second is the field name. In this case we're interested in the "my_col"
  // field:
  // FieldAccessTypedExpr 让我们从输入中选择一个特定的字段/列 数据集。 第一个参数定义了这个字段的返回类型，
  // 第二个是字段名。 在这种情况下，我们对“my_col”感兴趣的字段：
  auto fieldAccessExprNode =
      std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "my_col");

  // CallTypedExpr will be the root of our expression tree, and defines a
  // function call. The first parameter is the expected return type (bigint in
  // this case), the second is the list of parameter the function takes, and the
  // third is the function name.
  //
  // This will be the root of our expression tree. In a realistic use case, this
  // would be automatically and recursively generated based on some input IDL
  // (or by a SQL string parser).
  //
  // CallTypedExpr 将是我们表达式树的根，并定义了一个函数调用。
  // 第一个参数是预期的返回类型（本例中为 bigint），第二个是函数采用的参数列表，第三个是函数名。
  auto exprTree = std::make_shared<core::CallTypedExpr>(
      BIGINT(),
      std::vector<core::TypedExprPtr>{fieldAccessExprNode},
      "times_two");

  // Lastly, ExprSet contains the main expression evaluation logic. It takes a
  // vector of expression trees (if there are multiple expressions to be
  // evaluated). ExprSet will output one column per input exprTree. It also
  // takes the execution context associated with the current thread of
  // execution.
  exec::ExprSet exprSet({exprTree}, &execCtx);

  // Print the exprs tree.
  std::cout << "Expres tree" << std::endl;
  std::cout << exprSet.toString(/*compact=*/false);

  // Generate input batch.
  //
  // The next step is to generate an input batch of data to exercise the
  // expression evaluation code. Expressions are always evaluated using
  // RowVectors as input, which are named collections of vectors.
  //
  // Let's first create a single flat vector to represent the input
  // "my_col" bigint column, and manually fill some data in it:
  const size_t vectorSize = 10;
  // 创建一个int列的数据桶
  auto flatVector = std::dynamic_pointer_cast<FlatVector<int64_t>>(
      BaseVector::create(BIGINT(), vectorSize, execCtx.pool()));
  auto rawValues = flatVector->mutableRawValues();
  // 放10个数进去
  std::iota(rawValues, rawValues + vectorSize, 0); // 0, 1, 2, 3, ...

  // Then, let's wrap the generated flatVector in a RowVector:
  auto rowVector = std::make_shared<RowVector>(
      execCtx.pool(), // pool where allocations will be made.
      inputRowType, // input row type (defined above).
      BufferPtr(nullptr), // no nulls for this example.
      vectorSize, // length of the vectors.
      std::vector<VectorPtr>{flatVector}); // the input vector data.

  // Now we move to the actual execution.
  //
  // We first create a vector of VectorPtrs to hold the expression results.
  // (ExprSet outputs one vector per input expression - in this case, 1). The
  // output vector will be allocated internally by ExprSet, so we just need to
  // have a single null VectorPtr in this std::vector.
  std::vector<VectorPtr> result{nullptr};

  // Next, we create an input selectivity vector that controls the visibility
  // of records from the input RowVector. In this case we don't want to filter
  // out any rows, so just create a selectivity vector with all bits set.
  // 创建一个Select vector，用来进行数据过滤，在这个例子里面不需要进行数据过滤
  SelectivityVector rows{vectorSize};

  // Before execution we need to create one last structure - EvalCtx - which
  // holds context about the expression evaluation of this particular batch.
  // ExprSets can be reused by the same expression over multiple batches, but we
  // need one EvalCtx per RowVector.
  // 在执行之前，我们需要创建最后一个结构 - EvalCtx - 它保存有关此特定批次的表达式执行的上下文。
  // ExprSets 可以在多个批次中被同一个表达式重用，但是我们每个 RowVector 需要一个 EvalCtx。
  exec::EvalCtx evalCtx(&execCtx, &exprSet, rowVector.get());

  // Voila! Here we do the actual evaluation. When this function returns, the
  // output vectors will be available in the results vector. Note that ExprSet's
  // logic is synchronous and single threaded.
  // 执行表达式：输入数据、执行上下文、计算结果
  exprSet.eval(rows, &evalCtx, &result);

  // Print the output vector, just for fun:
  // 打印输出执行结果
  const auto& outputVector = result.front();
  for (size_t i = 0; i < outputVector->size(); ++i) {
    LOG(INFO) << outputVector->toString(i);
  }

  // Lastly, remember that all allocations are associated with the scoped pool
  // created in the beginning, and moved to ExecCtx. Once ExecCtx dies, it
  // destructs the pool which will deallocate all memory associated with it, so
  // be mindful about the object lifetime!
  //
  // (in this example this is safe since ExecCtx will be destructed last).
  return 0;
}
