import logging
import os
import unittest

import torch
from torch import nn as nn

import torch_xla
from torch_xla import runtime as xr


class PallasTest(unittest.TestCase):

  @unittest.skipIf(xr.device_type() != 'TPU', "This test only works on TPU.")
  def test_tpu_custom_call_pallas_add(self):
    # This payload is generated by the following Pallas code:
    # def add_vectors_kernel(x_ref, y_ref, o_ref):
    #   x, y = x_ref[...], y_ref[...]
    #   o_ref[...] = x + y
    payload = "{\"custom_call_config\": {\"body\": \"TUzvUgFNTElSMTguMC4wZ2l0AAErCwEDBQcJAQMLAwUDDQcFDxEJBRMVA2lNDQFLBw8LEw8PDwsPMwsLCwtlCwsLCwsPCw8PEwsTDwsTDwsPDxMLDwUDYQENGwcTDxsPAsICHx0rLQUXAwMnKRURNx1HSRELAQUZHTM1AwsVFxkbHw0hDSMlBRsBAQUdDQlhZmZpbmVfbWFwPChkMCkgLT4gKGQwKT4ABR8FIQUjBSUFJxEDAQUpFS8JHQ8xFwUTAQUrFwUdAR05OwUtFwUlAR0/QQUvFUMJHQ9FFwUVAQUxFREJI3RwdS5tZW1vcnlfc3BhY2U8dm1lbT4AF0sDIQcdAycDIQcBAgIFBwEBAQEBAgQEpwUBEAEHAwEFAxEBEwcDFScHAQEBAQEBBwMDBwMDCwYDAwUFAQcHAwMHAwMLBgMDBQUDCwkGPQMFBQkNBwMLBwMDCwYLAwUFBRENBAsHDwURBQABBgMBBQEAdgcz2wsTGdkNCxMjIR0pJ0MNCwsTDw8PDQkLEWJ1aWx0aW4AZnVuYwB0cHUAYXJpdGgAdmVjdG9yAG1vZHVsZQByZXR1cm4AY29uc3RhbnQAYWRkaQBsb2FkAHN0b3JlAC9ob21lL2p3dGFuL3BhbGxhcy9wYWxsYXNfYWRkLnB5AGFkZF92ZWN0b3JzX2tlcm5lbABkaW1lbnNpb25fc2VtYW50aWNzAGZ1bmN0aW9uX3R5cGUAc2NhbGFyX3ByZWZldGNoAHNjcmF0Y2hfb3BlcmFuZHMAc3ltX25hbWUAbWFpbgB2YWx1ZQAvZ2V0W3RyZWU9UHlUcmVlRGVmKChDdXN0b21Ob2RlKE5ESW5kZXhlclsoUHlUcmVlRGVmKChDdXN0b21Ob2RlKFNsaWNlWygwLCA4KV0sIFtdKSwpKSwgKDgsKSwgKCkpXSwgW10pLCkpXQBhZGRfdmVjdG9ycwA8bW9kdWxlPgAvYWRkAC9zd2FwW3RyZWU9UHlUcmVlRGVmKChDdXN0b21Ob2RlKE5ESW5kZXhlclsoUHlUcmVlRGVmKChDdXN0b21Ob2RlKFNsaWNlWygwLCA4KV0sIFtdKSwpKSwgKDgsKSwgKCkpXSwgW10pLCkpXQA=\", \"needs_layout_passes\": true}}"

    x = torch.arange(8, dtype=torch.int).to("xla")
    y = torch.arange(8, dtype=torch.int).to("xla")
    expected_output = x + y
    output = torch.arange(8, dtype=torch.int).to("xla")

    torch_xla._XLAC._xla_tpu_custom_call_(output, [x, y], payload)
    self.assertTrue(torch.allclose(output.cpu(), expected_output.cpu()))

  @unittest.skipIf(xr.device_type() != 'TPU', "This test only works on TPU.")
  def test_tpu_custom_call_pallas_add_one(self):
    # This payload is generated by the following Pallas code:
    # def add_vectors_kernel(x_ref, o_ref):
    #   o_ref[...] = x_ref[...] + 1
    payload = "{\"custom_call_config\": {\"body\": \"TUzvUgFNTElSMTguMC4wZ2l0AAEtCwEDBQcJAQMLAwUDDQcFDxEJBxMVFwNlSQ0BRwcPCw8PDxMLDzMLCwsLZQsLCwsPCw8LEw8PCxMPCxMTDwsLBQNhAQ0bDxMHFw8CpgIfFSsxBRkdQwMdRQMRCwEDAw8nBRsdKQMDCxUXGRsfCyELIyUFHQEBBR8NCWFmZmluZV9tYXA8KGQwKSAtPiAoZDApPgAFIQUjBSUFJxEHAQUpHS0vBSsXBRsBFTM5HTU3BS0XBS8BHTs9BS8XBUUBAwMPQREDBQUxBTMjdHB1Lm1lbW9yeV9zcGFjZTx2bWVtPgAXRwMhAx0BAgInAyEDAwUFAQEBAQIEBKEFARABBwMBBQMRARMHAxMnBQEBAQEHAxENAwcLBhEDBQUBBQcDBz8DAw0GBwMFAwkJBgcDBQUHCwcDCQ0DBwsGCQMFBQMPDwQJBw0DDwUAAQYDAQUBAMIHNdsLEyEv2QsTIyEdKQ1DDRULCxMPDw8NCQsRYnVpbHRpbgBmdW5jAHRwdQBhcml0aAB2ZWN0b3IAbW9kdWxlAHJldHVybgBjb25zdGFudABhZGRpAGxvYWQAYnJvYWRjYXN0AHN0b3JlAC9ob21lL2p3dGFuL3BhbGxhcy9wYWxsYXNfYWRkLnB5AHZhbHVlAGRpbWVuc2lvbl9zZW1hbnRpY3MAZnVuY3Rpb25fdHlwZQBzY2FsYXJfcHJlZmV0Y2gAc2NyYXRjaF9vcGVyYW5kcwBzeW1fbmFtZQBtYWluAC9nZXRbdHJlZT1QeVRyZWVEZWYoKEN1c3RvbU5vZGUoTkRJbmRleGVyWyhQeVRyZWVEZWYoKEN1c3RvbU5vZGUoU2xpY2VbKDAsIDgpXSwgW10pLCkpLCAoOCwpLCAoKSldLCBbXSksKSldAGFkZF9vbmVfdmVjdG9yc19rZXJuZWwAYWRkX3ZlY3RvcnNfb25lADxtb2R1bGU+AC9hZGQAL3N3YXBbdHJlZT1QeVRyZWVEZWYoKEN1c3RvbU5vZGUoTkRJbmRleGVyWyhQeVRyZWVEZWYoKEN1c3RvbU5vZGUoU2xpY2VbKDAsIDgpXSwgW10pLCkpLCAoOCwpLCAoKSldLCBbXSksKSldAA==\", \"needs_layout_passes\": true}}"

    x = torch.arange(8, dtype=torch.int).to("xla")
    expected_output = x + 1
    output = torch.arange(8, dtype=torch.int).to("xla")

    torch_xla._XLAC._xla_tpu_custom_call_(output, [x], payload)
    self.assertTrue(torch.allclose(output.cpu(), expected_output.cpu()))

  @unittest.skipIf(xr.device_type() != 'TPU', "This test only works on TPU.")
  def test_tpu_custom_call_pallas_raise(self):
    # This payload is generated by the following Pallas code:
    # def add_vectors_kernel(x_ref, o_ref):
    #   o_ref[...] = x_ref[...] + 1
    payload = "{\"custom_call_config\": {\"body\": \"TUzvUgFNTElSMTguMC4wZ2l0AAEtCwEDBQcJAQMLAwUDDQcFDxEJBxMVFwNlSQ0BRwcPCw8PDxMLDzMLCwsLZQsLCwsPCw8LEw8PCxMPCxMTDwsLBQNhAQ0bDxMHFw8CpgIfFSsxBRkdQwMdRQMRCwEDAw8nBRsdKQMDCxUXGRsfCyELIyUFHQEBBR8NCWFmZmluZV9tYXA8KGQwKSAtPiAoZDApPgAFIQUjBSUFJxEHAQUpHS0vBSsXBRsBFTM5HTU3BS0XBS8BHTs9BS8XBUUBAwMPQREDBQUxBTMjdHB1Lm1lbW9yeV9zcGFjZTx2bWVtPgAXRwMhAx0BAgInAyEDAwUFAQEBAQIEBKEFARABBwMBBQMRARMHAxMnBQEBAQEHAxENAwcLBhEDBQUBBQcDBz8DAw0GBwMFAwkJBgcDBQUHCwcDCQ0DBwsGCQMFBQMPDwQJBw0DDwUAAQYDAQUBAMIHNdsLEyEv2QsTIyEdKQ1DDRULCxMPDw8NCQsRYnVpbHRpbgBmdW5jAHRwdQBhcml0aAB2ZWN0b3IAbW9kdWxlAHJldHVybgBjb25zdGFudABhZGRpAGxvYWQAYnJvYWRjYXN0AHN0b3JlAC9ob21lL2p3dGFuL3BhbGxhcy9wYWxsYXNfYWRkLnB5AHZhbHVlAGRpbWVuc2lvbl9zZW1hbnRpY3MAZnVuY3Rpb25fdHlwZQBzY2FsYXJfcHJlZmV0Y2gAc2NyYXRjaF9vcGVyYW5kcwBzeW1fbmFtZQBtYWluAC9nZXRbdHJlZT1QeVRyZWVEZWYoKEN1c3RvbU5vZGUoTkRJbmRleGVyWyhQeVRyZWVEZWYoKEN1c3RvbU5vZGUoU2xpY2VbKDAsIDgpXSwgW10pLCkpLCAoOCwpLCAoKSldLCBbXSksKSldAGFkZF9vbmVfdmVjdG9yc19rZXJuZWwAYWRkX3ZlY3RvcnNfb25lADxtb2R1bGU+AC9hZGQAL3N3YXBbdHJlZT1QeVRyZWVEZWYoKEN1c3RvbU5vZGUoTkRJbmRleGVyWyhQeVRyZWVEZWYoKEN1c3RvbU5vZGUoU2xpY2VbKDAsIDgpXSwgW10pLCkpLCAoOCwpLCAoKSldLCBbXSksKSldAA==\", \"needs_layout_passes\": true}}"

    output = torch.arange(8, dtype=torch.int).to("xla")

    # _xla_tpu_custom_call_ requires at least one input.
    with self.assertRaises(RuntimeError):
      torch_xla._XLAC._xla_tpu_custom_call_(output, [], payload)
      output.cpu()

  @unittest.skipIf(xr.device_type() != 'TPU', "This test only works on TPU.")
  # Mosiac is not compatible with our sorted layout that boosts performance for dim > 2 tensor input applications, like resnet.
  # For LLM, it should be fine since all inputs are 2D.
  @unittest.mock.patch.dict(os.environ, {"XLA_TPU_LAYOUT": "0"})
  def test_tpu_custom_call_pallas_flash_attention(self):
    # This payload is generated by the following Pallas code:
    # https://github.com/google/jax/blob/b2058d72b7e1693a41303d5411572aabf99b7981/jax/experimental/pallas/ops/tpu/flash_attention.py#L139
    # To be noted, set `jax.config.update('jax_default_matmul_precision', jax.lax.Precision.HIGHEST)`` before generating the payload.
    payload = "{\"custom_call_config\": {\"body\": \"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eC9leHBlcmltZW50YWwvcGFsbGFzL29wcy90cHUvZmxhc2hfYXR0ZW50aW9uLnB5AF9mbGFzaF9hdHRlbnRpb25fa2VybmVsX3NpbmdsZV9iYXRjaF9zaW5nbGVfc3RlcAB2YWx1ZQBmdW5jdGlvbl90eXBlAHN5bV9uYW1lAHRyYW5zZm9ybV9pbmRpY2VzAHdpbmRvd19ib3VuZHMAL2dldFt0cmVlPVB5VHJlZURlZigoQ3VzdG9tTm9kZShOREluZGV4ZXJbKFB5VHJlZURlZigoKiwgKiwgQ3VzdG9tTm9kZShTbGljZVsoMCwgMTI4KV0sIFtdKSwgQ3VzdG9tTm9kZShTbGljZVsoMCwgNCldLCBbXSkpKSwgKDEsIDEsIDEyOCwgNCksICgpKV0sIFsqLCAqXSksKSldAHRyYW5zZm9ybV8wAHRyYW5zZm9ybV8xAHRyYW5zZm9ybV8yAHRyYW5zZm9ybV8zAHByZWNpc2lvbgB0cmFuc3Bvc2VfbGhzAHRyYW5zcG9zZV9yaHMAa2luZAByZWR1Y3Rpb25fZGltcwAvYnJvYWRjYXN0X2luX2RpbVtzaGFwZT0oMTI4LCAxKSBicm9hZGNhc3RfZGltZW5zaW9ucz0oMCwpXQBkaW1lbnNpb25fc2VtYW50aWNzAGl0ZXJhdGlvbl9ib3VuZHMAc2NhbGFyX3ByZWZldGNoAHNjcmF0Y2hfb3BlcmFuZHMAbWFpbgB3aW5kb3dfcGFyYW1zAF9mbGFzaF9hdHRlbnRpb25fa2VybmVsAF9mbGFzaF9hdHRlbnRpb25faW1wbABfZmxhc2hfYXR0ZW50aW9uAGZsYXNoX2F0dGVudGlvbgA8bW9kdWxlPgAvbW50L2Rpc2tzL3NzZC93b3JrL3BhbGxhcy9wYWxsYXNfYWRkLnB5AC9kb3RfZ2VuZXJhbFtkaW1lbnNpb25fbnVtYmVycz0oKCgxLCksICgxLCkpLCAoKCksICgpKSkgcHJlY2lzaW9uPSg8UHJlY2lzaW9uLkhJR0hFU1Q6IDI+LCA8UHJlY2lzaW9uLkhJR0hFU1Q6IDI+KSBwcmVmZXJyZWRfZWxlbWVudF90eXBlPWZsb2F0MzJdAC9yZWR1Y2VfbWF4W2F4ZXM9KDEsKV0AL3N1YgBmYXN0bWF0aAAvZXhwAC9yZWR1Y2Vfc3VtW2F4ZXM9KDEsKV0AL2RpdgAvZG90X2dlbmVyYWxbZGltZW5zaW9uX251bWJlcnM9KCgoMSwpLCAoMCwpKSwgKCgpLCAoKSkpIHByZWNpc2lvbj0oPFByZWNpc2lvbi5ISUdIRVNUOiAyPiwgPFByZWNpc2lvbi5ISUdIRVNUOiAyPikgcHJlZmVycmVkX2VsZW1lbnRfdHlwZT1mbG9hdDMyXQAvc3dhcFt0cmVlPVB5VHJlZURlZigoQ3VzdG9tTm9kZShOREluZGV4ZXJbKFB5VHJlZURlZigoKiwgKiwgQ3VzdG9tTm9kZShTbGljZVsoMCwgMTI4KV0sIFtdKSwgQ3VzdG9tTm9kZShTbGljZVsoMCwgNCldLCBbXSkpKSwgKDEsIDEsIDEyOCwgNCksICgpKV0sIFsqLCAqXSksKSldAA==\", \"needs_layout_passes\": true}}"

    # The division is to cause potential precision issue on TPU.
    q_mini = torch.arange(128 * 4, dtype=torch.float32).reshape(128, 4) / 13
    k_mini = torch.arange(
        1000, 1000 + 128 * 4, dtype=torch.float32).reshape(128, 4) / 13
    q = q_mini.broadcast_to(3, 2, 128, 4).to("xla")
    k = k_mini.broadcast_to(3, 2, 128, 4).to("xla")
    v = torch.ones(3, 2, 128, 4).to("xla")
    o = torch.zeros(3, 2, 128, 4).to("xla")

    def attention(q, k, v):
      attn_weight = q @ k.transpose(-2, -1)
      attn_weight = nn.functional.softmax(attn_weight, dim=-1)
      attn_output = attn_weight @ v
      return attn_output

    expected_o = attention(q, k, v)

    torch_xla._XLAC._xla_tpu_custom_call_(o, [q, k, v], payload)
    self.assertTrue(torch.allclose(o.cpu(), expected_o.cpu()))

  @unittest.skipIf(xr.device_type() != 'TPU', "This test only works on TPU.")
  # TODO: Make the tpu_custom_call_ as functional.
  @unittest.mock.patch.dict(os.environ, {"XLA_DISABLE_FUNCTIONALIZATION": "1"})
  def test_tpu_custom_call_pallas_add_one_dynamo(self):
    # This payload is generated by the following Pallas code:
    # def add_vectors_kernel(x_ref, o_ref):
    #   o_ref[...] = x_ref[...] + 1
    payload = "{\"custom_call_config\": {\"body\": \"TUzvUgFNTElSMTguMC4wZ2l0AAEtCwEDBQcJAQMLAwUDDQcFDxEJBxMVFwNlSQ0BRwcPCw8PDxMLDzMLCwsLZQsLCwsPCw8LEw8PCxMPCxMTDwsLBQNhAQ0bDxMHFw8CpgIfFSsxBRkdQwMdRQMRCwEDAw8nBRsdKQMDCxUXGRsfCyELIyUFHQEBBR8NCWFmZmluZV9tYXA8KGQwKSAtPiAoZDApPgAFIQUjBSUFJxEHAQUpHS0vBSsXBRsBFTM5HTU3BS0XBS8BHTs9BS8XBUUBAwMPQREDBQUxBTMjdHB1Lm1lbW9yeV9zcGFjZTx2bWVtPgAXRwMhAx0BAgInAyEDAwUFAQEBAQIEBKEFARABBwMBBQMRARMHAxMnBQEBAQEHAxENAwcLBhEDBQUBBQcDBz8DAw0GBwMFAwkJBgcDBQUHCwcDCQ0DBwsGCQMFBQMPDwQJBw0DDwUAAQYDAQUBAMIHNdsLEyEv2QsTIyEdKQ1DDRULCxMPDw8NCQsRYnVpbHRpbgBmdW5jAHRwdQBhcml0aAB2ZWN0b3IAbW9kdWxlAHJldHVybgBjb25zdGFudABhZGRpAGxvYWQAYnJvYWRjYXN0AHN0b3JlAC9ob21lL2p3dGFuL3BhbGxhcy9wYWxsYXNfYWRkLnB5AHZhbHVlAGRpbWVuc2lvbl9zZW1hbnRpY3MAZnVuY3Rpb25fdHlwZQBzY2FsYXJfcHJlZmV0Y2gAc2NyYXRjaF9vcGVyYW5kcwBzeW1fbmFtZQBtYWluAC9nZXRbdHJlZT1QeVRyZWVEZWYoKEN1c3RvbU5vZGUoTkRJbmRleGVyWyhQeVRyZWVEZWYoKEN1c3RvbU5vZGUoU2xpY2VbKDAsIDgpXSwgW10pLCkpLCAoOCwpLCAoKSldLCBbXSksKSldAGFkZF9vbmVfdmVjdG9yc19rZXJuZWwAYWRkX3ZlY3RvcnNfb25lADxtb2R1bGU+AC9hZGQAL3N3YXBbdHJlZT1QeVRyZWVEZWYoKEN1c3RvbU5vZGUoTkRJbmRleGVyWyhQeVRyZWVEZWYoKEN1c3RvbU5vZGUoU2xpY2VbKDAsIDgpXSwgW10pLCkpLCAoOCwpLCAoKSldLCBbXSksKSldAA==\", \"needs_layout_passes\": true}}"

    x = torch.arange(8, dtype=torch.int).to("xla")
    expected_output = x + 1
    output = torch.arange(8, dtype=torch.int).to("xla")

    import torch_xla.experimental.custom_kernel

    def add_one_pallas(output, inputs, payload):
      torch.ops.xla.tpu_custom_call_(output, inputs, payload)

    compiled_add_one_pallas = torch.compile(
        add_one_pallas, backend='openxla', fullgraph=True)

    compiled_add_one_pallas(output, [x], payload)
    self.assertTrue(torch.allclose(output.cpu(), expected_output.cpu()))

  @unittest.skipIf(xr.device_type() != 'TPU', "This test only works on TPU.")
  def test_tpu_custom_call_pallas_extract_add_payload(self):
    import jax
    import jax.numpy as jnp
    import jax._src.pallas.mosaic.pallas_call_registration

    from jax.experimental import pallas as pl

    def add_vectors_kernel(x_ref, y_ref, o_ref):
      x, y = x_ref[...], y_ref[...]
      o_ref[...] = x + y

    @jax.jit
    def add_vectors(x: jax.Array, y: jax.Array) -> jax.Array:
      return pl.pallas_call(
          add_vectors_kernel, out_shape=jax.ShapeDtypeStruct(x.shape,
                                                             x.dtype))(x, y)

    import torch_xla.experimental.custom_kernel as custom_kernel

    ir = jax.jit(add_vectors).lower(jnp.arange(8), jnp.arange(8)).compiler_ir()
    payload = custom_kernel._extract_backend_config(ir)
    # The payload being generated could vary each time. We just want to make sure
    # the most important fields are present.
    self.assertIn("custom_call_config", payload)


if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  # TODO: do we want to set the following flags?
  torch.set_default_dtype(torch.float32)
  torch.manual_seed(42)
  torch_xla._XLAC._xla_set_use_full_mat_mul_precision(
      use_full_mat_mul_precision=True)
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
