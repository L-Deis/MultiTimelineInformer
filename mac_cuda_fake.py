"""
mac_cuda_fake.py

Monkey-patch PyTorch so:
  1) torch.cuda.* is faked if real CUDA is unavailable but MPS is available.
  2) torch.device("cuda") -> "mps"/"cpu" as appropriate.
  3) os.environ["CUDA_VISIBLE_DEVICES"] is "overridden" if we only have MPS.

Usage:
    import mac_cuda_fake   # Hack is applied immediately
    import torch

    # The rest of your code can just pretend it's on CUDA.
    print_flush(torch.cuda.is_available())         # True if MPS is available
    os.environ["CUDA_VISIBLE_DEVICES"] = "2" # Actually sets "0" if MPS is forced.
    x = torch.randn(2).to("cuda")            # Goes to MPS under the hood.

Use at your own risk. This is highly unofficial.
"""

import os
import torch
import builtins
from utils.logger import print_flush

builtins.print = print_flush

# ------------------------------------------------------------------------
# (A) Monkey-patch os.environ so that setting CUDA_VISIBLE_DEVICES is overridden.
# ------------------------------------------------------------------------
_original_environ = os.environ

class _FakeEnviron(dict):
    def __init__(self, original):
        # Copy the real environment into our dict
        super().__init__(original)

    def __setitem__(self, key, value):
        if key == "CUDA_VISIBLE_DEVICES":
            # If MPS is available but real CUDA is not, override whatever the user sets.
            if torch.backends.mps.is_available() and not torch.cuda.is_available():
                print_flush(f"[mac_cuda_fake] Overriding CUDA_VISIBLE_DEVICES -> '0' (Faking single GPU with MPS).")
                value = "0"
        return super().__setitem__(key, value)

# Replace os.environ with our wrapper
os.environ = _FakeEnviron(_original_environ)


# ------------------------------------------------------------------------
# (B) Monkey-patch torch.cuda
# ------------------------------------------------------------------------
_original_cuda = torch.cuda

def _fake_is_available():
    """
    Return True if real CUDA is available OR if MPS is available.
    This convinces code that "some GPU" is present.
    """
    return _original_cuda.is_available() or torch.backends.mps.is_available()

def _fake_device_count():
    """
    If real CUDA is available, return that count.
    Else if MPS is available, pretend there's 1 GPU.
    """
    if _original_cuda.is_available():
        return _original_cuda.device_count()
    elif torch.backends.mps.is_available():
        return 1
    return 0

def _fake_current_device():
    """
    Pretend device 0 is active if MPS is available.
    """
    if _original_cuda.is_available():
        return _original_cuda.current_device()
    elif torch.backends.mps.is_available():
        return 0
    return 0

def _fake_get_device_name(device=None):
    """
    Return the real device name if CUDA is real,
    else return "MPS" if MPS is available.
    """
    if _original_cuda.is_available():
        if device is not None:
            return _original_cuda.get_device_name(device)
        return _original_cuda.get_device_name()
    return "MPS"

class _FakeCudaModule:
    """
    Fake cuda module that pretends MPS is CUDA if real CUDA is unavailable.
    """

    def is_available(self):
        return _fake_is_available()

    def device_count(self):
        return _fake_device_count()

    def current_device(self):
        return _fake_current_device()

    def get_device_name(self, device=None):
        return _fake_get_device_name(device)

    def synchronize(self, device=None):
        """
        If real CUDA is available, synchronize that.
        If not, but MPS is available, try torch.mps.synchronize().
        """
        if _original_cuda.is_available():
            _original_cuda.synchronize(device)
        elif torch.backends.mps.is_available():
            if hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()

    def __getattr__(self, name):
        """
        Fallback for any other cuda.* attributes or functions.
        If real CUDA is available, pass through to real CUDA.
        If we only have MPS, some functions won't exist or won't behave as expected.
        """
        return getattr(_original_cuda, name)

# Replace torch.cuda with the fake
torch.cuda = _FakeCudaModule()


# ------------------------------------------------------------------------
# (C) Monkey-patch torch.device so "cuda" -> "mps" (if available) or "cpu"
# ------------------------------------------------------------------------
_original_device = torch.device

def _custom_device(device_str):
    if "cuda" in device_str:
        if torch.backends.mps.is_available():
            print_flush("[mac_cuda_fake] Overriding 'cuda' -> 'mps'")
            return _original_device("mps")
        else:
            print_flush("[mac_cuda_fake] No MPS, overriding 'cuda' -> 'cpu'")
            return _original_device("cpu")
    return _original_device(device_str)

torch.device = _custom_device


# ------------------------------------------------------------------------
# (D) Optional quick test if run as script
# ------------------------------------------------------------------------
if __name__ == "__main__":
    print_flush("[mac_cuda_fake] Testing monkey patch...")

    print_flush("torch.cuda.is_available() =", torch.cuda.is_available())
    print_flush("torch.cuda.device_count() =", torch.cuda.device_count())

    # Force setting CUDA_VISIBLE_DEVICES
    print_flush("Setting os.environ['CUDA_VISIBLE_DEVICES'] = '2'")
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    print_flush("Actual CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", None))

    dev = torch.device("cuda:0")
    print_flush("Requested device: 'cuda:0' ->", dev)

    x = torch.randn(3, 3).to(dev)
    print_flush("x is on device:", x.device)
