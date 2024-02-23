import unittest

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
from torch_xla.core.dynamo_bridge import AliasWithBufferDonorContext


class TestBufferDonationUtil(unittest.TestCase):

  def test_hash_with_buffer_donor(self):
    device = xm.xla_device()
    input = torch.randn(5, 5).to(device)
    res = torch.cos(input)
    hash_no_donor = torch_xla._XLAC._get_graph_hash([res])
    self.assertTrue(torch_xla._XLAC._set_buffer_donation(input, True))
    # without the AliasWithBufferDonorContext, buffer donor will be ignored,
    # so we still expect the hash to be the same.
    hash_with_donor = torch_xla._XLAC._get_graph_hash([res])
    self.assertEqual(hash_no_donor, hash_with_donor)

    with AliasWithBufferDonorContext(True) as context:
      hash_with_donor_and_context = torch_xla._XLAC._get_graph_hash([res])
    self.assertNotEqual(hash_no_donor, hash_with_donor_and_context)


class TestDynamoBufferDonationAliasing(unittest.TestCase):

  def dummy_inplace_add(self, input):
    input += 1
    return

  def dummy_add(self, input):
    return input + 1

  def test_manual_buffer_donation(self):
    device = xm.xla_device()
    input = torch.randn(5, 5).to(device)
    input_cloned = torch.clone(input)
    dummy_inplace_add_compiled = torch.compile(
        self.dummy_inplace_add, backend='openxla')

    met.clear_all()
    # input is a device_data, we should be able to set the buffer donation field.
    self.assertTrue(torch_xla._XLAC._set_buffer_donation(input, True))
    # make sure buffer donation setting is correctly updated
    self.assertTrue(torch_xla._XLAC._get_buffer_donation(input))
    self.assertIn('XlaSetBufferDonation', met.counter_names())
    self.assertEqual(met.counter_value('XlaSetBufferDonation'), 1)
    dummy_inplace_add_compiled(input)
    torch.allclose(input_cloned.cpu() + 1, input.cpu())

  def test_manual_buffer_donation_for_non_inplce_op(self):
    device = xm.xla_device()
    input = torch.randn(5, 5).to(device)
    input_cloned = torch.clone(input)
    dummy_add_compiled = torch.compile(self.dummy_add, backend='openxla')

    met.clear_all()
    # input is a device_data, we should be able to set the buffer donation field.
    self.assertTrue(torch_xla._XLAC._set_buffer_donation(input, True))
    # make sure buffer donation setting is correctly updated
    self.assertTrue(torch_xla._XLAC._get_buffer_donation(input))
    self.assertIn('XlaSetBufferDonation', met.counter_names())
    self.assertEqual(met.counter_value('XlaSetBufferDonation'), 1)

    res = dummy_add_compiled(input)
    # check input's buffer has been aliased.
    xm.wait_device_ops()
    self.assertIn('Data Handle: Deleted',
                  torch_xla._XLAC._get_xla_tensor_debug_info(input))
    torch.allclose(input_cloned.cpu() + 1, res.cpu())

  def test_buffer_donation_on_non_data_tensor(self):
    device = xm.xla_device()
    input = torch.randn(5, 5).to(device)
    res = input + 1

    met.clear_all()
    # res now points to a `Add` IR, only data's buffer can be aliased
    self.assertFalse(torch_xla._XLAC._set_buffer_donation(res, True))
    self.assertFalse(torch_xla._XLAC._get_buffer_donation(res))
    self.assertNotIn('XlaSetBufferDonation', met.counter_names())


class TestNonDynamoBufferDonationAliasing(unittest.TestCase):

  def dummy_fn(self, input):
    return torch.cos(torch.sin(input))

  # Currently let's skip buffer donation api for the non-dynamo use case
  def test_buffer_donation_skip_for_non_dynamo(self):
    device = xm.xla_device()
    input = torch.randn(5, 5).to(device)
    xm.mark_step()
    met.clear_all()

    # We should be able to set buffer donation for input tensor, but when mark_step
    # triggered, the buffer donation should be ignored.
    self.assertTrue(torch_xla._XLAC._set_buffer_donation(input, True))
    res = self.dummy_fn(input)
    xm.mark_step()
    # Make sure that input buffer is not aliased and can be used for other compuations.
    # Also make sure that buffer_donation will not trigger recompilation in non-dynamo.
    self.assertTrue(torch_xla._XLAC._set_buffer_donation(input, False))
    res2 = self.dummy_fn(input)
    xm.mark_step()
    torch.allclose(res.cpu(), res2.cpu())
    self.assertEqual(met.metric_data('CompileTime')[0], 1)

  def test_no_op_mark_step_keep_buffer_donation(self):
    device = xm.xla_device()
    input = torch.randn(5, 5).to(device)
    self.assertTrue(torch_xla._XLAC._set_buffer_donation(input, True))
    xm.mark_step()
    self.assertTrue(torch_xla._XLAC._get_buffer_donation(input))
    xm.mark_step()
    self.assertTrue(torch_xla._XLAC._get_buffer_donation(input))


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
