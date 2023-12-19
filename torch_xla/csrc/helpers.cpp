#include "torch_xla/csrc/helpers.h"

#include <torch/csrc/lazy/core/helpers.h>
#include <torch/csrc/lazy/core/util.h>

#include <iterator>
#include <limits>

#include "absl/strings/str_join.h"
#include "torch_xla/csrc/convert_ops.h"
#include "torch_xla/csrc/dtype.h"
#include "torch_xla/csrc/runtime/debug_macros.h"
#include "torch_xla/csrc/runtime/tf_logging.h"
#include "torch_xla/csrc/runtime/util.h"
#include "torch_xla/csrc/shape_helper.h"
#include "torch_xla/csrc/tensor_util.h"
#include "xla/client/lib/constants.h"
#include "xla/primitive_util.h"
#include "xla/shape_util.h"

namespace torch_xla {
namespace {

xla::XlaOp ConvertBinaryOpResult(xla::XlaOp op1, xla::XlaOp op2,
                                 xla::XlaOp result) {
  xla::PrimitiveType type1 = XlaHelpers::TypeOfXlaOp(op1);
  xla::PrimitiveType type2 = XlaHelpers::TypeOfXlaOp(op2);
  xla::PrimitiveType result_type = XlaHelpers::TypeOfXlaOp(result);
  if (type1 == type2 && type1 != result_type) {
    return ConvertTo(result, result_type, type1);
  }
  return result;
}

xla::XlaComputation CreateComputation(
    const std::string& name, xla::PrimitiveType type,
    const std::function<xla::XlaOp(xla::XlaOp, xla::XlaOp)>& op) {
  xla::XlaBuilder builder(name);
  xla::XlaOp x =
      xla::Parameter(&builder, 0, xla::ShapeUtil::MakeShape(type, {}), "x");
  xla::XlaOp y =
      xla::Parameter(&builder, 1, xla::ShapeUtil::MakeShape(type, {}), "y");
  return ConsumeValue(builder.Build(op(x, y)));
}

}  // namespace

xla::PrecisionConfig::Precision XlaHelpers::s_mat_mul_precision =
    xla::PrecisionConfig::DEFAULT;

xla::PrecisionConfig XlaHelpers::BuildPrecisionConfig(
    xla::PrecisionConfig::Precision conv_precision, int num_arguments) {
  xla::PrecisionConfig precision_config;
  precision_config.mutable_operand_precision()->Resize(num_arguments,
                                                       conv_precision);
  return precision_config;
}

xla::XlaOp XlaHelpers::BroadcastDimensions(xla::XlaOp input,
                                           absl::Span<const int64_t> dimensions,
                                           absl::Span<const int64_t> sizes) {
  XLA_CHECK_EQ(dimensions.size(), sizes.size());
  std::vector<int64_t> bcast_sizes = SizesOfXlaOp(input);
  for (size_t i = 0; i < dimensions.size(); ++i) {
    bcast_sizes.at(dimensions[i]) = sizes[i];
    if (XlaHelpers::IsUnboundedDynamismEnabled()) {
      XLA_CHECK(sizes[i] != xla::Shape::kUnboundedSize);
    }
  }
  return xla::BroadcastInDim(input, bcast_sizes,
                             GetAllDimensions(bcast_sizes.size()));
}

xla::XlaOp XlaHelpers::CreateReturnValue(
    xla::XlaBuilder* builder, const std::vector<xla::XlaOp>& outputs) {
  if (outputs.size() > 1) {
    return xla::Tuple(builder, outputs);
  } else if (!outputs.empty()) {
    return xla::GetTupleElement(xla::Tuple(builder, {outputs[0]}), 0);
  } else {
    return xla::Tuple(builder, {});
  }
}

int64_t XlaHelpers::GetDynamicDimension(const xla::Shape& shape) {
  int64_t dynamic_dimension = -1;
  for (int64_t i = 0; i < shape.rank(); ++i) {
    if (shape.is_dynamic_dimension(i)) {
      XLA_CHECK(dynamic_dimension < 0)
          << "Only one dynamic dimension is supported: " << i << " and "
          << dynamic_dimension << " in " << shape;
      dynamic_dimension = i;
    }
  }
  return dynamic_dimension;
}

XlaHelpers::DynamicSize XlaHelpers::GetDimensionsSize(
    absl::Span<const xla::XlaOp> inputs, absl::Span<const int64_t> dimensions) {
  XLA_CHECK(!inputs.empty());
  xla::PrimitiveType size_type = GetShapeDimensionType(/*device=*/nullptr);
  xla::XlaOp size;
  int64_t size_scalar = 1;
  for (auto& input : inputs) {
    const xla::Shape& shape = ShapeHelper::ShapeOfXlaOp(input);
    for (auto dim : dimensions) {
      if (size_scalar >= 0) {
        if (!shape.is_dynamic_dimension(dim)) {
          size_scalar *= shape.dimensions(dim);
          continue;
        } else {
          if (size_scalar != 1) {
            size = ScalarValue(size_scalar, size_type, input.builder());
          }
          size_scalar = -1;
        }
      }
      if (size.valid()) {
        size = size * xla::GetDimensionSize(input, dim);
      } else {
        size = xla::GetDimensionSize(input, dim);
      }
    }
  }
  absl::optional<int64_t> scalar_size;
  if (size_scalar >= 0) {
    scalar_size = size_scalar;
  }
  if (!size.valid()) {
    size = ScalarValue(size_scalar, size_type, inputs[0].builder());
  }
  return {size, scalar_size};
}

XlaHelpers::MinMax XlaHelpers::MinMaxValues(xla::PrimitiveType type) {
  switch (type) {
    case xla::PrimitiveType::S8:
      return {std::numeric_limits<int8_t>::lowest(),
              std::numeric_limits<int8_t>::max()};
    case xla::PrimitiveType::U8:
      return {std::numeric_limits<uint8_t>::lowest(),
              std::numeric_limits<uint8_t>::max()};
    case xla::PrimitiveType::S16:
      return {std::numeric_limits<int16_t>::lowest(),
              std::numeric_limits<int16_t>::max()};
    case xla::PrimitiveType::U16:
      return {std::numeric_limits<uint16_t>::lowest(),
              std::numeric_limits<uint16_t>::max()};
    case xla::PrimitiveType::S32:
      return {static_cast<int64_t>(std::numeric_limits<int32_t>::lowest()),
              static_cast<int64_t>(std::numeric_limits<int32_t>::max())};
    case xla::PrimitiveType::U32:
      return {static_cast<int64_t>(std::numeric_limits<uint32_t>::lowest()),
              static_cast<int64_t>(std::numeric_limits<uint32_t>::max())};
    case xla::PrimitiveType::S64:
      return {static_cast<int64_t>(std::numeric_limits<int64_t>::lowest()),
              static_cast<int64_t>(std::numeric_limits<int64_t>::max())};
    case xla::PrimitiveType::U64:
      return {static_cast<int64_t>(std::numeric_limits<uint64_t>::lowest()),
              static_cast<int64_t>(std::numeric_limits<uint64_t>::max())};
    case xla::PrimitiveType::F16:
      return {static_cast<float>(std::numeric_limits<xla::half>::lowest()),
              static_cast<float>(std::numeric_limits<xla::half>::max())};
    case xla::PrimitiveType::BF16:
    case xla::PrimitiveType::F32:
      return {std::numeric_limits<float>::lowest(),
              std::numeric_limits<float>::max()};
    case xla::PrimitiveType::F64:
      return {std::numeric_limits<double>::lowest(),
              std::numeric_limits<double>::max()};
    case xla::PrimitiveType::PRED:
      return {0, 1};
    default:
      XLA_ERROR() << "Unsupported XLA type " << type;
  }
}

xla::PaddingConfig XlaHelpers::MakeXlaPaddingConfigFromNdPadding(
    absl::Span<const int64_t> padding) {
  XLA_CHECK_EQ(padding.size() % 2, 0)
      << "Padding specification must have even length";
  XLA_CHECK(!padding.empty()) << "Padding specification cannot be empty";
  xla::PaddingConfig padding_config;
  for (int i = 0; i < padding.size(); i += 2) {
    xla::PaddingConfig::PaddingConfigDimension* dims =
        padding_config.add_dimensions();
    dims->set_edge_padding_low(padding[padding.size() - i - 2]);
    dims->set_edge_padding_high(padding[padding.size() - i - 1]);
  }
  return padding_config;
}

xla::XlaComputation XlaHelpers::CreateAddComputation(xla::PrimitiveType type) {
  return CreateComputation(
      "AddComputation", type, [&](xla::XlaOp x, xla::XlaOp y) {
        return type == xla::PrimitiveType::PRED ? xla::Or(x, y)
                                                : xla::Add(x, y);
      });
}

xla::XlaComputation XlaHelpers::CreateMulComputation(xla::PrimitiveType type) {
  return CreateComputation(
      "MulComputation", type,
      [&](xla::XlaOp x, xla::XlaOp y) { return xla::Mul(x, y); });
}

xla::XlaComputation XlaHelpers::CreateMaxComputation(xla::PrimitiveType type) {
  return CreateComputation(
      "MaxComputation", type,
      [&](xla::XlaOp x, xla::XlaOp y) { return xla::Max(x, y); });
}

xla::XlaComputation XlaHelpers::CreateMinComputation(xla::PrimitiveType type) {
  return CreateComputation(
      "MinComputation", type,
      [&](xla::XlaOp x, xla::XlaOp y) { return xla::Min(x, y); });
}

xla::XlaComputation XlaHelpers::CreateAndComputation(xla::PrimitiveType type) {
  return CreateComputation(
      "AndComputation", type,
      [&](xla::XlaOp x, xla::XlaOp y) { return xla::And(x, y); });
}

xla::XlaComputation XlaHelpers::CreateOrComputation(xla::PrimitiveType type) {
  return CreateComputation(
      "OrComputation", type,
      [&](xla::XlaOp x, xla::XlaOp y) { return xla::Or(x, y); });
}

std::vector<int64_t> XlaHelpers::SizesOfXlaOp(xla::XlaOp op) {
  const xla::Shape& op_shape = ShapeHelper::ShapeOfXlaOp(op);
  return std::vector<int64_t>(op_shape.dimensions().begin(),
                              op_shape.dimensions().end());
}

xla::PrimitiveType XlaHelpers::TypeOfXlaOp(xla::XlaOp op) {
  return ShapeHelper::ShapeOfXlaOp(op).element_type();
}

xla::XlaOp XlaHelpers::ReshapeToRank(xla::XlaOp input, int64_t expected_rank,
                                     int64_t offset) {
  const xla::Shape& shape = ShapeHelper::ShapeOfXlaOp(input);
  XLA_CHECK_LE(offset + shape.rank(), expected_rank);
  if (shape.rank() == expected_rank) {
    return input;
  }
  std::vector<int64_t> dimensions(expected_rank - offset - shape.rank(), 1);
  dimensions.insert(dimensions.end(), shape.dimensions().begin(),
                    shape.dimensions().end());
  dimensions.insert(dimensions.end(), offset, 1);
  return xla::Reshape(input, dimensions);
}

absl::optional<XlaHelpers::DynamicReshapeInfo>
XlaHelpers::GetDynamicReshapeInfo(const xla::Shape& input_shape,
                                  absl::Span<const int64_t> output_sizes) {
  int64_t input_dyndim_idx = GetDynamicDimension(input_shape);
  if (input_dyndim_idx < 0) {
    return absl::nullopt;
  }
  DynamicReshapeInfo info;
  info.output_shape =
      xla::ShapeUtil::MakeShape(input_shape.element_type(), output_sizes);
  if (info.output_shape.rank() > 0) {
    int64_t size_prod_until_dyndim = 1;
    for (int64_t i = 0; i <= input_dyndim_idx; ++i) {
      size_prod_until_dyndim *= input_shape.dimensions(i);
    }
    int64_t dynamic_dimension = -1;
    int64_t out_size = 1;
    for (int64_t i = 0; i < output_sizes.size(); ++i) {
      XLA_CHECK_LE(out_size, size_prod_until_dyndim /
                                 input_shape.dimensions(input_dyndim_idx))
          << "Unable to map dynamic dimension of shape " << input_shape
          << " to output sizes (" << absl::StrJoin(output_sizes, ", ") << ")";
      out_size *= output_sizes[i];
      if (out_size >= size_prod_until_dyndim) {
        dynamic_dimension = i;
        break;
      }
    }
    XLA_CHECK(dynamic_dimension >= 0)
        << "Unable to map dynamic dimension of shape " << input_shape
        << " to output sizes (" << absl::StrJoin(output_sizes, ", ") << ")";
    info.dynamic_dimension = dynamic_dimension;
    info.output_shape.set_dynamic_dimension(info.dynamic_dimension, true);
  }
  return std::move(info);
}

xla::Shape XlaHelpers::GetDynamicReshape(
    const xla::Shape& input_shape, absl::Span<const int64_t> output_sizes) {
  auto info = GetDynamicReshapeInfo(input_shape, output_sizes);
  if (info) {
    return info->output_shape;
  }
  return xla::ShapeUtil::MakeShape(input_shape.element_type(), output_sizes);
}

xla::XlaOp XlaHelpers::DynamicReshape(xla::XlaOp input,
                                      absl::Span<const int64_t> output_sizes) {
  const xla::Shape& input_shape = ShapeHelper::ShapeOfXlaOp(input);
  if (output_sizes == input_shape.dimensions()) {
    return input;
  }
  auto info = GetDynamicReshapeInfo(input_shape, output_sizes);
  if (info) {
    return xla::ReshapeWithInferredDimension(input, output_sizes,
                                             info->dynamic_dimension);
  }
  return xla::Reshape(input, output_sizes);
}

xla::XlaOp XlaHelpers::DynamicReshapeAs(xla::XlaOp input,
                                        const xla::Shape& shape) {
  const xla::Shape& input_shape = ShapeHelper::ShapeOfXlaOp(input);
  int64_t dynamic_dimension = GetDynamicDimension(shape);
  if (dynamic_dimension >= 0) {
    return xla::ReshapeWithInferredDimension(input, shape.dimensions(),
                                             dynamic_dimension);
  }
  return shape.dimensions() == input_shape.dimensions()
             ? input
             : xla::Reshape(input, shape.dimensions());
}

bool XlaHelpers::IsUnboundedDynamic(const xla::Shape& shape) {
  XLA_CHECK(XlaHelpers::IsUnboundedDynamismEnabled())
      << "set EXPERIMENTAL_XLA_UNBOUNDED_DYNAMISM=1 to run any unbounded "
         "dynamism workload.";
  const absl::Span<const int64_t> dims = shape.dimensions();
  return std::any_of(dims.begin(), dims.end(), [](int64_t size) {
    return size == xla::Shape::kUnboundedSize;
  });
}

xla::XlaOp XlaHelpers::DynamicUnboundedReshape(
    xla::XlaOp input, xla::XlaOp aux_input,
    absl::Span<const int64_t> output_sizes) {
  XLA_CHECK(XlaHelpers::IsUnboundedDynamismEnabled())
      << "set EXPERIMENTAL_XLA_UNBOUNDED_DYNAMISM=1 to run any unbounded "
         "dynamism workload.";
  const xla::Shape& aux_input_shape = ShapeHelper::ShapeOfXlaOp(aux_input);
  XLA_CHECK(output_sizes.size() == aux_input_shape.rank())
      << "XlaHelpers::DynamicUnboundedReshape constrainled failed!";
  std::vector<xla::XlaOp> get_dim_ops;
  std::vector<xla::XlaOp> reshaped_ops;
  bool all_static = true;
  std::vector<bool> output_dynamic(output_sizes.size(), false);

  for (int i = 0; i < output_sizes.size(); i++) {
    if (output_sizes[i] == xla::Shape::kUnboundedSize) {
      output_dynamic[i] = true;
      get_dim_ops.push_back(xla::GetDimensionSize(aux_input, i));
      all_static = false;
    } else {
      get_dim_ops.push_back(XlaHelpers::ScalarValue<int32_t>(
          output_sizes[i], aux_input.builder()));
    }
  }

  if (all_static) {
    return xla::Reshape(input, output_sizes);
  }

  // Create the reshape from scalar to 1-D vector
  for (auto get_dim_op : get_dim_ops) {
    reshaped_ops.push_back(xla::Reshape(get_dim_op, {1}));
  }

  // Create Concatenate op
  auto concat_op = xla::ConcatInDim(input.builder(), reshaped_ops, {0});
  return xla::CustomCall(
      aux_input.builder(), "stablehlo.dynamic_reshape", {input, concat_op},
      xla::ShapeUtil::MakeShape(aux_input_shape.element_type(), output_sizes,
                                output_dynamic));

  return input;
}

bool XlaHelpers::SameStaticDimensions(const xla::Shape& shape1,
                                      const xla::Shape& shape2) {
  return shape1.is_static() && shape2.is_static() &&
         shape1.dimensions() == shape2.dimensions();
}

xla::XlaOp XlaHelpers::Flatten(xla::XlaOp input, xla::Shape* input_shape) {
  runtime::util::MaybePtr<xla::Shape> input_shape_tmp(input_shape);
  *input_shape_tmp = ShapeHelper::ShapeOfXlaOp(input);
  if (input_shape_tmp->rank() == 1) {
    return input;
  }
  int64_t input_elements = xla::ShapeUtil::ElementsIn(*input_shape_tmp);
  return DynamicReshape(input, {input_elements});
}

xla::XlaOp XlaHelpers::FlattenDimRange(xla::XlaOp input, int64_t start,
                                       int64_t range, xla::Shape* input_shape) {
  runtime::util::MaybePtr<xla::Shape> input_shape_tmp(input_shape);
  *input_shape_tmp = ShapeHelper::ShapeOfXlaOp(input);

  std::vector<int64_t> sizes;
  int64_t flat_size = -1;
  for (int64_t dim = 0; dim < input_shape_tmp->rank(); ++dim) {
    if (dim < start || dim >= start + range) {
      if (flat_size >= 0) {
        sizes.push_back(flat_size);
        flat_size = -1;
      }
      sizes.push_back(input_shape_tmp->dimensions(dim));
    } else {
      flat_size =
          (flat_size < 0 ? 1 : flat_size) * input_shape_tmp->dimensions(dim);
    }
  }
  if (flat_size >= 0) {
    sizes.push_back(flat_size);
  }
  return DynamicReshape(input, sizes);
}

xla::XlaOp XlaHelpers::LinearInterpolation(xla::XlaOp value0, xla::XlaOp value1,
                                           double alpha) {
  const xla::Shape& shape = ShapeHelper::ShapeOfXlaOp(value0);
  xla::XlaOp one = xla::One(value0.builder(), shape.element_type());
  xla::XlaOp alpha_value =
      ScalarValue(alpha, shape.element_type(), value0.builder());
  return value0 * alpha_value + value1 * (one - alpha_value);
}

xla::PrimitiveType XlaHelpers::PromoteType(xla::PrimitiveType type1,
                                           xla::PrimitiveType type2) {
  if (type1 == type2) {
    return type1;
  }
  int64_t size1 = xla::ShapeUtil::ByteSizeOfPrimitiveType(type1);
  int64_t size2 = xla::ShapeUtil::ByteSizeOfPrimitiveType(type2);
  if (xla::primitive_util::IsComplexType(type1)) {
    return (!xla::primitive_util::IsComplexType(type2) || size1 >= size2)
               ? type1
               : type2;
  }
  if (xla::primitive_util::IsComplexType(type2)) {
    return type2;
  }
  if (xla::primitive_util::IsFloatingPointType(type1)) {
    return (!xla::primitive_util::IsFloatingPointType(type2) || size1 >= size2)
               ? type1
               : type2;
  }
  if (xla::primitive_util::IsFloatingPointType(type2) || size2 > size1) {
    return type2;
  }
  if (xla::primitive_util::IsIntegralType(type1) &&
      xla::primitive_util::IsIntegralType(type2)) {
    if (size1 > size2) {
      return type1;
    }
    if (size2 > size1) {
      return type2;
    }
    // At this point, they are not the same type, they are both integers, and
    // they have the same size. One of them must be unsigned and the other
    // signed, convert to unsigned.
    return xla::primitive_util::UnsignedIntegralTypeForBitWidth(
        xla::primitive_util::BitWidth(type1));
  }
  if (type1 == xla::PrimitiveType::PRED) {
    return type2;
  }
  if (type2 == xla::PrimitiveType::PRED) {
    return type1;
  }
  // If nothing matches the above logic, first operand wins.
  return type1;
}

xla::PrimitiveType XlaHelpers::PromoteType(xla::PrimitiveType type1,
                                           xla::PrimitiveType type2,
                                           xla::PrimitiveType type3) {
  return PromoteType(PromoteType(type1, type2), type3);
}

std::pair<xla::XlaOp, xla::XlaOp> XlaHelpers::PromoteValues(xla::XlaOp op1,
                                                            xla::XlaOp op2) {
  xla::PrimitiveType type1 = TypeOfXlaOp(op1);
  xla::PrimitiveType type2 = TypeOfXlaOp(op2);
  xla::PrimitiveType result_type = PromoteType(type1, type2);
  if (type1 != result_type) {
    op1 = ConvertTo(op1, type1, result_type);
  }
  if (type2 != result_type) {
    op2 = ConvertTo(op2, type2, result_type);
  }
  return std::pair<xla::XlaOp, xla::XlaOp>(op1, op2);
}

std::tuple<xla::XlaOp, xla::XlaOp, xla::XlaOp> XlaHelpers::PromoteValues(
    xla::XlaOp op1, xla::XlaOp op2, xla::XlaOp op3) {
  xla::PrimitiveType type1 = TypeOfXlaOp(op1);
  xla::PrimitiveType type2 = TypeOfXlaOp(op2);
  xla::PrimitiveType type3 = TypeOfXlaOp(op3);
  xla::PrimitiveType result_type = PromoteType(type1, type2, type3);
  if (type1 != result_type) {
    op1 = ConvertTo(op1, type1, result_type);
  }
  if (type2 != result_type) {
    op2 = ConvertTo(op2, type2, result_type);
  }
  if (type3 != result_type) {
    op3 = ConvertTo(op3, type3, result_type);
  }
  return std::tuple<xla::XlaOp, xla::XlaOp, xla::XlaOp>(op1, op2, op3);
}

std::pair<xla::XlaOp, xla::XlaOp> XlaHelpers::PromoteSecondValue(
    xla::XlaOp op1, xla::XlaOp op2) {
  xla::PrimitiveType type1 = TypeOfXlaOp(op1);
  xla::PrimitiveType type2 = TypeOfXlaOp(op2);
  return type1 == type2 ? std::pair<xla::XlaOp, xla::XlaOp>(op1, op2)
                        : std::pair<xla::XlaOp, xla::XlaOp>(
                              op1, ConvertTo(op2, type2, type1));
}

xla::Shape XlaHelpers::GetPromotedShape(const xla::Shape& shape1,
                                        const xla::Shape& shape2) {
  std::vector<int64_t> dimensions;
  std::vector<bool> dynamic_dimensions;

  // If the rank of a shape is bigger than then other, fill up the first
  // dimensions with the ones of the bigger.
  // Example:
  //   shape1 = [9, 7, 6, 5, 2]
  //   shape2 =       [6, 1, 2]
  // Insert [9, 7] into the dimensions vector.
  if (shape1.dimensions().size() > shape2.dimensions().size()) {
    dimensions.insert(
        dimensions.end(), shape1.dimensions().begin(),
        shape1.dimensions().begin() +
            (shape1.dimensions().size() - shape2.dimensions().size()));
    dynamic_dimensions.insert(dynamic_dimensions.end(),
                              shape1.dynamic_dimensions().begin(),
                              shape1.dynamic_dimensions().begin() +
                                  (shape1.dynamic_dimensions().size() -
                                   shape2.dynamic_dimensions().size()));
  } else if (shape2.dimensions().size() > shape1.dimensions().size()) {
    dimensions.insert(
        dimensions.end(), shape2.dimensions().begin(),
        shape2.dimensions().begin() +
            (shape2.dimensions().size() - shape1.dimensions().size()));
    dynamic_dimensions.insert(dynamic_dimensions.end(),
                              shape2.dynamic_dimensions().begin(),
                              shape2.dynamic_dimensions().begin() +
                                  (shape2.dynamic_dimensions().size() -
                                   shape1.dynamic_dimensions().size()));
  }
  // For the common dimensions, they must match, or one of them be 1.
  size_t min_size =
      std::min(shape1.dimensions().size(), shape2.dimensions().size());
  for (size_t i = 0; i < min_size; i++) {
    int64_t dim1 =
        shape1.dimensions()[shape1.dimensions().size() - min_size + i];
    int64_t dynamic_dim1 =
        shape1.dynamic_dimensions()[shape1.dynamic_dimensions().size() -
                                    min_size + i];
    int64_t dim2 =
        shape2.dimensions()[shape2.dimensions().size() - min_size + i];
    int64_t dynamic_dim2 =
        shape2.dynamic_dimensions()[shape2.dynamic_dimensions().size() -
                                    min_size + i];

    XLA_CHECK(dim1 == dim2 || dim1 == 1 || dim2 == 1 ||
              dim1 == xla::Shape::kUnboundedSize ||
              dim2 == xla::Shape::kUnboundedSize);
    if (dim1 == 0 || dim2 == 0) {
      dimensions.push_back(0);
      dynamic_dimensions.push_back(dynamic_dim1 || dynamic_dim2);
    } else if (dim1 == xla::Shape::kUnboundedSize ||
               dim2 == xla::Shape::kUnboundedSize) {
      dimensions.push_back(xla::Shape::kUnboundedSize);
      dynamic_dimensions.push_back(true);
    } else {
      dimensions.push_back(std::max<int64_t>(dim1, dim2));
      dynamic_dimensions.push_back(dynamic_dim1 || dynamic_dim2);
    }
  }

  return xla::ShapeUtil::MakeShape(shape1.element_type(), dimensions,
                                   dynamic_dimensions);
}

std::vector<int64_t> XlaHelpers::getBroadcastDimensions(xla::XlaOp op1,
                                                        xla::XlaOp op2) {
  const xla::Shape& shape1 = ShapeHelper::ShapeOfXlaOp(op1);
  const xla::Shape& shape2 = ShapeHelper::ShapeOfXlaOp(op2);
  if (shape1.rank() == 0 || shape2.rank() == 0 ||
      shape1.rank() == shape2.rank())
    return {};

  std::vector<int64_t> broadcast_dimensions(
      shape1.rank() <= shape2.rank() ? shape1.rank() : shape2.rank());
  std::iota(broadcast_dimensions.begin(), broadcast_dimensions.end(),
            std::abs(shape1.rank() - shape2.rank()));
  return broadcast_dimensions;
}

xla::Shape XlaHelpers::GetPromotedBinaryOpShape(const xla::Shape& shape1,
                                                const xla::Shape& shape2) {
  if (!shape1.is_dynamic() && !shape2.is_dynamic()) {
    auto promoted_shape = GetPromotedShape(shape1, shape2);
    return xla::ShapeUtil::MakeShape(
        PromoteType(shape1.element_type(), shape2.element_type()),
        promoted_shape.dimensions());
  }
  if (XlaHelpers::IsUnboundedDynamismEnabled()) {
    XLA_CHECK(!XlaHelpers::IsUnboundedDynamic(shape1) &&
              !XlaHelpers::IsUnboundedDynamic(shape2))
        << "Unreachable for unbounded dynamic code\n";
  }
  return GetPromotedDynamicShape(shape1, shape2);
}

xla::Shape XlaHelpers::GetPromotedDynamicShape(const xla::Shape& shape1,
                                               const xla::Shape& shape2) {
  std::vector<int64_t> upper_bounds1 =
      runtime::util::ToVector<int64_t>(shape1.dimensions());
  std::vector<int64_t> upper_bounds2 =
      runtime::util::ToVector<int64_t>(shape2.dimensions());
  absl::Span<const bool> dyn_dims1 = shape1.dynamic_dimensions();
  absl::Span<const bool> dyn_dims2 = shape2.dynamic_dimensions();
  std::vector<int64_t> upper_bounds;
  std::vector<bool> dyn_dims;

  // See
  // https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics
  if (upper_bounds1.size() > upper_bounds2.size()) {
    upper_bounds.insert(
        upper_bounds.end(), upper_bounds1.begin(),
        upper_bounds1.begin() + (upper_bounds1.size() - upper_bounds2.size()));
    dyn_dims.insert(dyn_dims.end(), dyn_dims1.begin(),
                    dyn_dims1.begin() + (dyn_dims1.size() - dyn_dims2.size()));
  } else {
    upper_bounds.insert(
        upper_bounds.end(), upper_bounds2.begin(),
        upper_bounds2.begin() + (upper_bounds2.size() - upper_bounds1.size()));
    dyn_dims.insert(dyn_dims.end(), dyn_dims2.begin(),
                    dyn_dims2.begin() + (dyn_dims2.size() - dyn_dims1.size()));
  }
  size_t min_size = std::min(upper_bounds1.size(), upper_bounds2.size());
  for (const auto i : c10::irange(min_size)) {
    int64_t ubound1 = upper_bounds1[upper_bounds1.size() - min_size + i];
    int64_t ubound2 = upper_bounds2[upper_bounds2.size() - min_size + i];
    bool is_dim1_dynamic = dyn_dims1[dyn_dims1.size() - min_size + i];
    bool is_dim2_dynamic = dyn_dims2[dyn_dims2.size() - min_size + i];
    if (!is_dim1_dynamic && !is_dim2_dynamic) {
      XLA_CHECK(ubound1 == 1 || ubound2 == 1 || ubound1 == ubound2)
          << "At dimension " << i
          << ", both dimension are static with real size " << ubound1 << " and "
          << ubound2;
    } else {
      // For now, if both dimension are dynamic and has the same upper bound, we
      // regard this dimension to be broadcastable.
      XLA_CHECK((is_dim1_dynamic && !is_dim2_dynamic && ubound2 == 1) ||
                (is_dim2_dynamic && !is_dim1_dynamic && ubound1 == 1) ||
                (is_dim1_dynamic && is_dim2_dynamic && ubound1 == ubound2))
          << "At dimension " << i << ", operand1 has dimension size " << ubound1
          << " isDynamic=" << is_dim1_dynamic
          << " vs operand2 has dimension size " << ubound2
          << " isDynamic=" << is_dim2_dynamic;
    }

    int64_t ubound = std::max<int64_t>(ubound1, ubound2);
    upper_bounds.push_back(ubound);
    bool is_dim_dynamic = is_dim1_dynamic || is_dim2_dynamic;
    dyn_dims.push_back(is_dim_dynamic);
  }
  const xla::Shape& promoted_shape = xla::ShapeUtil::MakeShape(
      PromoteType(shape1.element_type(), shape2.element_type()), upper_bounds,
      dyn_dims);

  std::cout << xla::ShapeUtil::HumanString(promoted_shape);
  return promoted_shape;
}

std::pair<xla::XlaOp, xla::XlaOp> XlaHelpers::PromoteShapes(xla::XlaOp op1,
                                                            xla::XlaOp op2) {
  const xla::Shape& shape1 = ShapeHelper::ShapeOfXlaOp(op1);
  const xla::Shape& shape2 = ShapeHelper::ShapeOfXlaOp(op2);

  xla::Shape shape = GetPromotedShape(shape1, shape2);
  if (shape.is_unbounded_dynamic()) {
    return ImplicitBroadcastWithUnboundedDynamicShapes(op1, op2, shape);
  }

  if (xla::ShapeUtil::Compatible(shape1, shape2)) {
    // Fast path shortcut if the shapes already matches in dimensions.
    return std::pair<xla::XlaOp, xla::XlaOp>(op1, op2);
  }
  XLA_CHECK(xla::ShapeUtil::SameElementType(shape1, shape2))
      << shape1 << " and " << shape2;

  return std::pair<xla::XlaOp, xla::XlaOp>(op1, op2);
}

std::pair<xla::XlaOp, xla::XlaOp> XlaHelpers::Promote(xla::XlaOp op1,
                                                      xla::XlaOp op2) {
  std::pair<xla::XlaOp, xla::XlaOp> vops = PromoteValues(op1, op2);
  return PromoteShapes(vops.first, vops.second);
}

std::pair<xla::XlaOp, xla::XlaOp> XlaHelpers::PromoteSecond(xla::XlaOp op1,
                                                            xla::XlaOp op2) {
  std::pair<xla::XlaOp, xla::XlaOp> vops = PromoteSecondValue(op1, op2);
  return PromoteShapes(vops.first, vops.second);
}

xla::XlaOp XlaHelpers::ImplicitBroadcast(xla::XlaOp op,
                                         const xla::Shape& op_shape,
                                         const xla::Shape& shape) {
  const auto& op_shape_dims = op_shape.dimensions();
  const auto& shape_dims = shape.dimensions();
  XLA_CHECK_GE(shape_dims.size(), op_shape_dims.size())
      << shape << " vs " << op_shape;
  int64_t size_delta = shape_dims.size() - op_shape_dims.size();
  xla::XlaOp new_op = op;
  if (!std::equal(op_shape_dims.begin(), op_shape_dims.end(),
                  shape_dims.begin() + size_delta)) {
    // If the base N dimensions do not match, broadcast the original op.
    // Example:
    //   op_shape =       [3, 1, 5]
    //   shape    = [6, 8, 3, 4, 5]
    // After this operation we will have:
    //   op_shape =       [3, 4, 5]
    std::vector<int64_t> common_shape_dims(shape_dims.begin() + size_delta,
                                           shape_dims.end());
    std::vector<int64_t> broadcast_dimensions(op_shape_dims.size());
    std::iota(broadcast_dimensions.begin(), broadcast_dimensions.end(), 0);
    new_op =
        xla::BroadcastInDim(new_op, common_shape_dims, broadcast_dimensions);
  }
  if (size_delta > 0) {
    // Add the major dimensions if necessary:
    // Example:
    //   op_shape =       [3, 4, 5]
    //   shape    = [6, 8, 3, 4, 5]
    // After this operation we will have (added [6, 8]):
    //   op_shape = [6, 8, 3, 4, 5]
    std::vector<int64_t> broadcast_sizes(shape_dims.begin(),
                                         shape_dims.begin() + size_delta);
    new_op = xla::Broadcast(new_op, broadcast_sizes);
  }
  return new_op;
}

std::pair<xla::XlaOp, xla::XlaOp>
XlaHelpers::ImplicitBroadcastWithUnboundedDynamicShapes(
    xla::XlaOp op1, xla::XlaOp op2, const xla::Shape& shape) {
  XLA_CHECK(shape.is_unbounded_dynamic());

  const xla::Shape& shape1 = ShapeHelper::ShapeOfXlaOp(op1);
  const xla::Shape& shape2 = ShapeHelper::ShapeOfXlaOp(op2);
  const auto& shape1_dims = shape1.dimensions();
  const auto& shape2_dims = shape2.dimensions();
  const auto& shape_dims = shape.dimensions();

  XLA_CHECK((shape1_dims.size() == shape2_dims.size() &&
             shape_dims.size() == shape1_dims.size()) ||
            (shape1_dims.size() > shape2_dims.size() &&
             shape_dims.size() == shape1_dims.size()) ||
            (shape1_dims.size() < shape2_dims.size() &&
             shape_dims.size() == shape2_dims.size()));

  int64_t size_delta = shape2_dims.size() - shape1_dims.size();

  // Extract the 'num_dims' counts of dimension sizes from the 'op' and append
  // them at 'op_dims'.
  auto extract_dim_size = [&](const xla::XlaOp op, const xla::Shape& op_shape,
                              int num_dims, std::vector<xla::XlaOp>& op_dims) {
    for (int i = 0; i < num_dims; i++) {
      op_dims.push_back(xla::Reshape(xla::GetDimensionSize(op, i), {1}));
    }
  };

  // Collect the dimension sizes of the broadcasted 'op1'.
  // Example:
  //   shape1 = [     1,    ?_1]
  //   shape2 = [?_2, ?_2', 5]
  //     ?: represents unbounded dynamic size and the prefix '_n' is used to
  //        differentiate instances of
  // We compute the followings:
  //   op1_dims = [?_2, 1, ?_1]
  //   op1_broadcast_dims = [1, 2]
  std::vector<xla::XlaOp> op1_dims;
  std::vector<int64_t> op1_broadcast_dims(shape1_dims.size());
  int64_t op1_broadcast_dim_start = 0;
  if (size_delta > 0) {
    extract_dim_size(op2, shape2, size_delta, op1_dims);
    op1_broadcast_dim_start = size_delta;
  }
  extract_dim_size(op1, shape1, shape1_dims.size(), op1_dims);
  std::iota(op1_broadcast_dims.begin(), op1_broadcast_dims.end(),
            op1_broadcast_dim_start);

  // Collect the dimension sizes of the broadcasted 'op2'.
  // Example:
  //   shape1 = [     1,    ?_1]
  //   shape2 = [?_2, ?_2', 5]
  // We compute the followings:
  //   op2_dims = [?_2, ?_2', 5]
  //   op2_broadcast_dims = [0, 1, 2]
  std::vector<xla::XlaOp> op2_dims;
  std::vector<int64_t> op2_broadcast_dims(shape2_dims.size());
  int64_t op2_broadcast_dim_start = 0;
  if (size_delta < 0) {
    extract_dim_size(op1, shape1, std::abs(size_delta), op2_dims);
    op2_broadcast_dim_start = std::abs(size_delta);
  }
  extract_dim_size(op2, shape2, shape2_dims.size(), op2_dims);
  std::iota(op2_broadcast_dims.begin(), op2_broadcast_dims.end(),
            op2_broadcast_dim_start);

  // The broadcasted shape is the max of the individual broadcasted shapes.
  // Example:
  //   shape1 = [     1,    ?_1]
  //   shape2 = [?_2, ?_2', 5]
  // We compute the followings:
  //   op1_dims = [?_2, 1, ?_1]
  //   op2_dims = [?_2, ?_2', 5]
  //   output_dimensions = max(op1_dims, op2_dims);
  auto output_dimensions =
      xla::Max(xla::ConcatInDim(op1.builder(), op1_dims, {0}),
               xla::ConcatInDim(op2.builder(), op2_dims, {0}));

  // Stringify the broadcast dimensions to provide for the 'backend_config'
  // attribute of the generated custom_call.
  auto stringify_broadcast_dimensions =
      [&](std::vector<int64_t> broadcast_dims) -> std::string {
    std::string str("{broadcast_dimensions=[");
    if (broadcast_dims.size() >= 1) {
      str += std::to_string(broadcast_dims[0]);
    }
    for (size_t i = 1; i < broadcast_dims.size(); i++) {
      str += ", " + std::to_string(broadcast_dims[i]);
    }
    str += "]}";
    return str;
  };

  auto broadcasted_op1 = xla::CustomCall(
      op1.builder(), "mhlo.dynamic_broadcast_in_dim",
      /*operands=*/{op1, output_dimensions}, /*shape*/ shape,
      /*opaque=*/stringify_broadcast_dimensions(op1_broadcast_dims));

  auto broadcasted_op2 = xla::CustomCall(
      op2.builder(), "mhlo.dynamic_broadcast_in_dim",
      /*operands=*/{op2, output_dimensions}, /*shape*/ shape,
      /*opaque=*/stringify_broadcast_dimensions(op2_broadcast_dims));

  return std::pair<xla::XlaOp, xla::XlaOp>(broadcasted_op1, broadcasted_op2);
}

xla::XlaOp XlaHelpers::PromotedBinaryOp(
    xla::XlaOp op1, xla::XlaOp op2,
    const std::function<xla::XlaOp(xla::XlaOp, xla::XlaOp)>& bin_op) {
  xla::XlaOp numeric_op1 = ConvertToNumeric(op1);
  xla::XlaOp numeric_op2 = ConvertToNumeric(op2);
  std::pair<xla::XlaOp, xla::XlaOp> vops = Promote(numeric_op1, numeric_op2);
  xla::XlaOp result = bin_op(vops.first, vops.second);
  return ConvertBinaryOpResult(op1, op2, result);
}

xla::XlaOp XlaHelpers::PromotedLogicalBinaryOp(
    xla::XlaOp op1, xla::XlaOp op2,
    const std::function<xla::XlaOp(xla::XlaOp, xla::XlaOp)>& bin_op) {
  // XLA only supports bitwise_and/or/xor so we need to cast inputs to
  // PRED first.
  op1 = xla::ConvertElementType(op1, xla::PrimitiveType::PRED);
  op2 = xla::ConvertElementType(op2, xla::PrimitiveType::PRED);
  return bin_op(op1, op2);
}

xla::XlaOp XlaHelpers::PromotedLogicalUnaryOp(
    xla::XlaOp op, const std::function<xla::XlaOp(xla::XlaOp)>& unary_op) {
  // XLA only supports bitwise_not so we need to cast inputs to
  // PRED first.
  op = xla::ConvertElementType(op, xla::PrimitiveType::PRED);
  return unary_op(op);
}

xla::StatusOr<xla::XlaComputation> XlaHelpers::WrapXlaComputation(
    const xla::XlaComputation& computation,
    const std::vector<xla::Shape>& parameter_shapes,
    std::vector<std::pair<int64_t, int64_t>> input_output_alias_pair) {
  xla::XlaBuilder builder(computation.proto().name());

  // Construct a single tuple parameter.
  xla::Shape input_tuple_shape;
  input_tuple_shape.set_element_type(xla::PrimitiveType::TUPLE);
  input_tuple_shape.mutable_tuple_shapes()->reserve(parameter_shapes.size());
  for (int i = 0; i < parameter_shapes.size(); ++i) {
    *input_tuple_shape.add_tuple_shapes() = parameter_shapes[i];
  }
  xla::XlaOp input_tuple = xla::Parameter(&builder, 0, input_tuple_shape, "in");

  // Handle the results of the original computation.
  std::vector<xla::XlaOp> inner_params;
  inner_params.reserve(parameter_shapes.size());
  for (int i = 0; i < parameter_shapes.size(); ++i) {
    inner_params.push_back(xla::GetTupleElement(input_tuple, i));
  }

  // Call the original computation.
  xla::XlaOp orig_result = xla::Call(&builder, computation, inner_params);

  // Rebuild aliasing.
  for (const auto& [input_index, output_index] : input_output_alias_pair) {
    // Both input and output will be a tuple so parameter_number will always be
    // 0
    builder.SetUpAlias(/*output_index=*/xla::ShapeIndex({output_index}),
                       /*param_number=*/0,
                       /*param_index=*/xla::ShapeIndex({input_index}));
  }

  return builder.Build(orig_result);
}

torch::lazy::Shape XlaHelpers::ConvertXlaShapeToLazy(const xla::Shape& shape) {
  at::ScalarType scalar_type = MaybeUpcastToHostTorchType(shape.element_type());
  c10::optional<std::vector<bool>> is_symbolic = c10::nullopt;
  if (shape.is_dynamic()) {
    std::vector<bool> xla_dynamic_dimensions =
        runtime::util::ToVector<bool>(shape.dynamic_dimensions());
    is_symbolic = c10::make_optional(xla_dynamic_dimensions);
  }

  return torch::lazy::Shape(
      scalar_type, runtime::util::ToVector<int64_t>(shape.dimensions()),
      std::move(is_symbolic));
}

}  // namespace torch_xla
