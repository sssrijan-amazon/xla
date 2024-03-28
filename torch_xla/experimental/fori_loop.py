import numpy as np
import torch
import torch_xla
import torch_xla.core.xla_builder as xb
import torch_xla.core.xla_model as xm
import torch_xla.utils.utils as xu
import torch_xla.core.xla_op_registry as xor

from torch._C import DispatchKey
from torch._ops import HigherOrderOperator
import torch._higher_order_ops.while_loop
from torch._higher_order_ops.while_loop import while_loop_op

# lower, upper, body_fun, init_val, one_value
# def fori_loop(upper, body_fun, lowers):#  upper, body_fun, *init_vals): # *init_val):
def fori_loop(lower, upper, body_fun, one_value, *init_val):
  print("init_val: ", init_val)

  # print("lower: ", lower) # tensor([1], device='xla:0', dtype=torch.int32)
  # print("upper: ", upper) # tensor([20], device='xla:0', dtype=torch.int32)
  # print("body_fun: ", body_fun) # <function WhileLoopTest.test_fori_loop_tpu_addition.<locals>.body_fun at 0x7f69c40ce320>
  # print("init_val: ", init_val) # tensor([1], device='xla:0', dtype=torch.int32)
  # print("one_value: ", one_value) # tensor([1], device='xla:0', dtype=torch.int32)

  device = xm.xla_device()

  # def cond_fn(lower, upper, x):
  def cond_fn(upper, lower, *x):
    return lower[0] < upper[0]

  # def body_fn(lower, upper, x):
  def body_fn(upper, lower, *x):
    one_value = torch.ones(1, dtype=torch.int32, device=device)
    # two_value = upper.clone()
    return (torch.sub(upper, one_value), lower, body_fun(one_value, *x)) # , one_value))

  # upper, lower, one_value, init_val
  # real(ov, lower, upper, x)
  # def cond_fn(upper, lower, one_value, init_val): #one_value, lower, upper, init_val): # loop_carry): # iter, upper, one_value): # lower, *init_vals):
  def old_cond_fn(one_value, lower, upper, init_val): 
    # lower, upper, one_value, init_val = loop_carry
    lower_compare = torch.add(lower, one_value)
    # upper_compare = torch.add(upper, one_value)
    # return lower[0] <= upper[0] # while stop when cond fail
    return lower_compare[0] <= upper[0] # upper_compare[0]

  # def body_fn(upper, lowers): # , *init_vals):
  # def body_fn(upper, lower, one_value, init_val): # one_value, lower, upper, init_val): # loop_carry): # iter, upper, one_value):
  def old_body_fn(one_value, lower, upper, init_val):
    # lower, upper, one_value, init_val = loop_carry
    # return (torch.add(iter, one_value).clone(), upper.clone(), one_value.clone(), body_fun(x, one_value).clone())
    # one_value = torch.tensor([1], dtype=torch.int32, device=device)
    # new_upper = torch.sub(upper, one_value)
    new_lower = torch.add(lower, one_value)
    new_init_val = body_fun(init_val, one_value)
    # return (new_lower, upper, one_value, new_init_val)
    # return (new_upper, lower, one_value, new_init_val) # one_value, lower, new_upper, new_init_val)
    # return (upper, new_lower, one_value, new_init_val)
    return (one_value, new_lower, upper, new_init_val)

  # loop_carruy_print = (lower, upper, one_value, init_val)
  # print("loop_carruy_print[0]: ", loop_carruy_print[0]) # tensor([1], device='xla:0', dtype=torch.int32)
  # print("loop_carruy_print[1]: ", loop_carruy_print[1]) # tensor([20], device='xla:0', dtype=torch.int32)
  # print("loop_carruy_print[2]: ", loop_carruy_print[2]) # tensor([1], device='xla:0', dtype=torch.int32)
  # print("loop_carruy_print[3]: ", loop_carruy_print[3]) # tensor([1], device='xla:0', dtype=torch.int32)

  # res = _xla_while_loop(cond_fn, body_fn, upper, lower, one_value, init_val) # one_value, lower, upper, init_val) # upper, lower, one_value, init_val)
  # res = _xla_while_loop(cond_fn, body_fn, one_value, lower, upper, init_val)
  res = _xla_while_loop(cond_fn, body_fn, lower, upper, *init_val)
  return res

@while_loop_op.py_impl(DispatchKey.XLA)
def while_loop(cond_fn, body_fn, operands):
  # cond_fn&body_fn: callable
  # operands: (Tuple of possibly nested dict/list/tuple of tensors)
  return _xla_while_loop(cond_fn, body_fn, operands)


# fori_loop: original_operands==(lower, upper, init_val)
# def _xla_while_loop(cond_fn, body_fn, original_operands):
# (lower, upper, one_value, init_val)
def _xla_while_loop(cond_fn, body_fn, *operands):
  # print("!!! arguments: original_operands: ", original_operands)
  # fake operands to split formal code
  # operands = [] # fake_operands
  # for original_operand in original_operands:
  #   device = original_operand.device
  #   operands.append(torch.randint(10, original_operand.size(), dtype=torch.int32).to(device))
  # operands = tuple(operands)
  # print("!!! operands: ", operands) # (tensor([0], device='xla:0', dtype=torch.int32), tensor([30], device='xla:0', dtype=torch.int32), tensor([1], device='xla:0', dtype=torch.int32))

  # print("!!! arguments: cond_fn: ", cond_fn, ", body_fn: ", body_fn, ", operands: ", operands)
  # cond_fn: <function fori_loop.<locals>.cond_fn at 0x7f469149e710>
  # body_fn: <function fori_loop.<locals>.body_fn at 0x7f469149e680>
  # operands: (tensor([1], device='xla:0', dtype=torch.int32),
  #            tensor([20], device='xla:0', dtype=torch.int32),
  #            tensor([1], device='xla:0', dtype=torch.int32),
  #            tensor([1], device='xla:0', dtype=torch.int32))

  # create inputs placeholder
  # operands_tuple = tuple(operands)
  kwargs = {}
  shapes = xb.tensor_shape((operands)) # _tuple)
  builder = xb.create_builder('test_while')
  params = []
  for shape in shapes:
    p = xb.mkparam(builder, len(params), shape)
    params.append(p)

  # lower, upper, init_val = operands
  # lower, upper, one_value, init_val = operands
  # print("arrive here!!!")
  # generate cond_fn xlacomputation
  # print("!!! operands: ", operands)
  # !!! operands:  
  #     (tensor([1], device='xla:0', dtype=torch.int32),
  #     tensor([20], device='xla:0', dtype=torch.int32),
  #     tensor([1], device='xla:0', dtype=torch.int32),
  #     tensor([1], device='xla:0', dtype=torch.int32))

  cond_result = cond_fn(*operands) # lower, upper, init_val) # operands) # *operands)
  cond_ctx = torch_xla._XLAC.lowering.LoweringContext()
  cond_ctx.set_name_string("condctx")
  # print("arrive here!!!")
  # print("cond_result: ", cond_result)
  # print("init_val: ", init_val)
  # TODO(@manfei) to reduce to operands[2:]
  cond_ctx.build([cond_result], list(operands[2:]))# operands[:1], operands[3:])) # [one_value, init_val]) # , init_val) # [operands[2]])
  # print("arrive here!!!")
  cond_hlo = cond_ctx.hlo()
  cond_computation = xb.computation_from_module_proto("condcomputation",
                                                      cond_hlo)
  cond_hlo_print = xb.get_computation_hlo(cond_computation)
  print("cond computation: !!!!!!!!!")
  print(cond_hlo_print)

  # generate body_fn xlacomputation
  body_result = body_fn(*operands) # lower, upper, init_val) # operands) # *operands)
  body_ctx = torch_xla._XLAC.lowering.LoweringContext()
  body_ctx.set_name_string("bodyctx")
  # body_ctx.build(list(body_result))
  body_ctx.build(list(body_result), []) # [one_value, init_val]) # , [init_val])
  body_hlo = body_ctx.hlo()
  body_computation = xb.computation_from_module_proto("bodycomputation",
                                                      body_hlo)
  # body_hlo_print = xb.get_computation_hlo(body_computation)
  # print("body computation: !!!!!!!!!")
  # print(body_hlo_print)

  # generate while xlacomputation
  input_tuple = xb.Op.tuple(tuple(params))
  w = xb.mkop(
      'While', (input_tuple.op,),
      condition_computation=cond_computation,
      body_computation=body_computation)
  name = 'fori_loop_ed_torch_func'
  computation = w.build(name)
  # hlo_print = xb.get_computation_hlo(computation)
  # print("while computation: !!!!!!!!!")
  # print(hlo_print)

  # gain final result with generated while xlacomputation
  result = torch_xla._XLAC._xla_user_computation('xla::_op_test_while', (operands), computation)

  # print("operands: ", operands)
  # print("upper: ", operands[0])
  # print("lower: ", operands[1])
  # print("init: ", operands[2])
  # print("in _xla_while_loop result: ", result)
  return result


##################### old tried code
 # limit_value = upper
  # init = lower
  # iterator = lower

  # one_value is actually not used here, but actually redefined in body_fn to avoid introduce new argument in body_xlacomputation
  # lower == init_val
  # assert(lower == init_val)
  # init = lower # = init_val
  # limit_value = upper

  # one_value_original = torch.tensor([1], dtype=torch.int32, device=device)
  # (a, b) = init_vals

    # def cond_fn(init, limit_value):
    #   return limit_value[0] >= init[0]

    # def body_fn(init, limit_value):
    #   one_value = torch.ones(1, dtype=torch.int32, device=device)
    #   return (torch.add(init, one_value), limit_value.clone())

  # cond_fn
  # def _fori_cond_fun(loop_carry):
  #   i, upper, _ = loop_carry
  #   return torch.lt(i, upper)

  # def _fori_body_fun(body_fun):
  #   # body_fun = weakref.ref(body_fun)
  #   def while_body_fun(loop_carry):
  #     i, upper, x = loop_carry
  #     one_value = torch.ones(1, dtype=torch.int32, device=device)
  #     # return torch.add(i, one_value), upper, body_fun(i, x) # body_fun()(i, x)
  #     return torch.add(i, one_value), upper, body_fun(x) # body_fun()(i, x)
  #   return while_body_fun

  # # def cond_fn(upper, lowers): # lower, *init_vals):
  # def cond_fn(init, limit_value): # lower, *init_vals):
  #   # init_val_compy = init_val.clone()
  #   # one_value1 = torch.tensor([0], dtype=torch.int32, device=device)
  #   # one_value2 = torch.tensor([0], dtype=torch.int32, device=device)
  #   # lower = torch.add(lower, one_value1[0])
  #   # lower = torch.sub(lower, one_value2[0])
  #   # assert isinstance(init_vals[0], torch.Tensor)
  #   # assert isinstance(init_vals[1], torch.Tensor)
  #   # bool_value = isinstance(init_vals[0], torch.Tensor) and isinstance(init_vals[1], torch.Tensor)
  #   # body_fun(*init_vals)
  #   # result = True
  #   # if (lower[0] <= upper[0]) and bool_value:
  #   #   return True
  #   # return False
  #   # bool_result = ((lower[0] <= upper[0]) and bool_value)
  #   # bool_tensor = torch.tensor(bool_result, dtype=torch.bool)
  #   # return bool_tensor # (lower[0] <= upper[0]) and bool_tensor
  #   # return lower[0] <= upper[0]
  #   # return lowers[0] <= upper[0]
  #   return limit_value[0] >= init[0]


  #   # one_value_original = torch.tensor(1, dtype=torch.int32, device=device)
  #   # (a, b) = init_vals
  #   # return (upper, torch.add(lower, 1), body_fun(a, b), b.clone())
  #   # return (upper.clone(), (torch.add(lower.clone(), init_vals[1].clone())).clone(), (body_fun(*init_vals)).clone(), init_vals[1].clone()) # init_vals[1:])
  #   # return (upper, (torch.add(lowers[0], lowers[2]), body_fun(lowers[1], lowers[2]), lowers[2])) # init_vals[1:])
  #   # (body_fun(*init_vals)).clone(), init_vals[1].clone())
  #   # body_fun(one_value_original, init_val)) # body_fun(lower, init_val))
  #   one_value = torch.ones(1, dtype=torch.int32, device=device)


  # res = while_loop(cond_fn, body_fn, (upper, lower, *init_vals))
  # lowers = (lower, *init_vals)
  # res = _xla_while_loop(cond_fn, body_fn, (upper, lowers)) # , *init_vals))
  # lower, upper, body_fun, init_val, one_value
  # res = _xla_while_loop(cond_fn, body_fn, (init, limit_value))
  # res = _xla_while_loop(cond_fn, body_fn, (lower, upper, init_val))


  # inits): # init_val, one_value):
  # _, _, result = _xla_while_loop(_fori_cond_fun, _fori_body_fun(body_fun),
  #                           (lower, upper, inits))
  # _, _, result = _xla_while_loop(_fori_cond_fun, _fori_body_fun(body_fun),
  #                           (lower, upper, init_val))
  # print("upper: ", upper)
  # print("lower: ", lower)